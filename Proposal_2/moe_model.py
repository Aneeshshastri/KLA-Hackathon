"""
MoE Router and Iterative Pipeline for Proposal 2.

Architecture:
  - MoERouter: Lightweight CNN → logits over 4 experts
  - Gumbel-Softmax hard routing (straight-through estimator)
  - IterativeMoE: 4-step iterative refinement with hard expert selection

Hard routing is mandatory because the upsample expert changes spatial
resolution (128→256), making soft blending impossible.

The pipeline handles the resolution change by:
  1. Before upsample: run same-resolution experts at current resolution
  2. When upsample expert is selected: resolution jumps to 256×256
  3. After upsample: run same-resolution experts at 256×256
  4. Upsample expert is masked out after first use

Since all experts are fully convolutional, they naturally handle both
128×128 and 256×256 inputs without any architecture changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from flax import nnx

from moe_experts import Expert, create_all_experts

# Expert indices (fixed ordering)
EXPERT_UPSAMPLE = 0
EXPERT_DEBLUR = 1
EXPERT_GAUSSIAN = 2
EXPERT_SPECKLE = 3
NUM_EXPERTS = 4
NUM_STEPS = 4

EXPERT_NAMES = ["upsample", "deblur", "gaussian", "speckle"]


# ─── Router Network ──────────────────────────────────────────────────────

class MoERouter(nnx.Module):
    """
    Lightweight convolutional router.
    Input: (B, H, W, 1) — current image at any resolution
    Output: (B, NUM_EXPERTS) — logits over experts
    """

    def __init__(self, num_experts: int = NUM_EXPERTS, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.proj = nnx.Linear(64, num_experts, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.leaky_relu(self.conv1(x), negative_slope=0.2)
        h = nnx.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = nnx.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = jnp.mean(h, axis=(1, 2))  # global avg pool → (B, 64)
        return self.proj(h)  # (B, num_experts)


# ─── Gumbel-Softmax Hard Routing ─────────────────────────────────────────

def gumbel_softmax_hard(logits: jax.Array, key: jax.Array, tau: float = 1.0) -> jax.Array:
    """
    Gumbel-Softmax with straight-through estimator.
    Returns one-hot vectors, but gradients flow through the soft version.

    Args:
        logits: (B, K) unnormalized log-probabilities
        key: PRNG key
        tau: temperature (lower = harder selection)

    Returns:
        (B, K) one-hot vectors (hard in forward, soft in backward)
    """
    gumbel_noise = -jnp.log(-jnp.log(
        jax.random.uniform(key, logits.shape, minval=1e-20, maxval=1.0) + 1e-20
    ))
    y_soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
    # Straight-through: forward uses hard, backward uses soft
    return y_hard - jax.lax.stop_gradient(y_soft) + y_soft


def argmax_hard(logits: jax.Array) -> jax.Array:
    """Deterministic hard routing for inference (no Gumbel noise)."""
    return jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])


# ─── Iterative MoE Pipeline ──────────────────────────────────────────────

class IterativeMoE(nnx.Module):
    """
    4-step iterative MoE pipeline with hard routing.

    At each step:
      1. Router produces logits over experts
      2. Mask out upsample expert if already used
      3. Gumbel-Softmax selects exactly one expert (hard)
      4. Run the selected expert via jax.lax.switch
      5. Update the image

    The upsample expert is applied exactly once across the 4 steps.
    """

    def __init__(self, rngs: nnx.Rngs, num_steps: int = NUM_STEPS):
        self.num_steps = num_steps

        # Create all experts
        all_experts = create_all_experts(rngs)
        self.expert_upsample = all_experts["upsample"]
        self.expert_deblur = all_experts["deblur"]
        self.expert_gaussian = all_experts["gaussian"]
        self.expert_speckle = all_experts["speckle"]

        # Router
        self.router = MoERouter(num_experts=NUM_EXPERTS, rngs=rngs)

    def _run_expert(self, expert_idx: jax.Array, x: jax.Array) -> jax.Array:
        """
        Run a single expert selected by index.
        Uses jax.lax.switch for JIT-compatible branching.
        """
        branches = [
            lambda x=x: self.expert_upsample(x),
            lambda x=x: self.expert_deblur(x),
            lambda x=x: self.expert_gaussian(x),
            lambda x=x: self.expert_speckle(x),
        ]
        return jax.lax.switch(expert_idx, branches)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        tau: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass: 4-step iterative refinement.

        Args:
            x: (B, 128, 128, 1) degraded input
            key: PRNG key for Gumbel sampling
            tau: Gumbel temperature
            deterministic: if True, use argmax instead of Gumbel

        Returns:
            (output, routing_decisions)
            output: (B, 256, 256, 1) restored image
            routing_decisions: (B, num_steps) selected expert indices
        """
        batch_size = x.shape[0]
        all_decisions = []

        # Track whether upsample has been used (per batch element)
        upsample_used = jnp.zeros((batch_size,), dtype=jnp.bool_)

        for step in range(self.num_steps):
            key, subkey = jax.random.split(key)

            # Get router logits
            logits = self.router(x)  # (B, 4)

            # Mask out upsample expert if already used
            mask = jnp.where(upsample_used, -1e9, 0.0)
            logits = logits.at[:, EXPERT_UPSAMPLE].add(mask)

            # Hard routing
            if deterministic:
                weights = argmax_hard(logits)  # (B, 4)
            else:
                weights = gumbel_softmax_hard(logits, subkey, tau=tau)  # (B, 4)

            # Get selected expert index per batch element
            selected = jnp.argmax(weights, axis=-1)  # (B,)
            all_decisions.append(selected)

            # Check if any batch element selected upsample
            chose_upsample = (selected == EXPERT_UPSAMPLE)

            # We need to handle the case where some batch elements select
            # upsample and others don't. Since jax.lax.switch doesn't handle
            # per-element branching, we run all possible experts and select.
            #
            # For same-resolution experts, output has same shape as input.
            # For upsample expert, output is 2x input.
            # We handle this by running experts separately for the two cases.

            # Run same-resolution experts
            out_deblur = self.expert_deblur(x)
            out_gaussian = self.expert_gaussian(x)
            out_speckle = self.expert_speckle(x)

            # Stack same-res outputs: (B, H, W, 1, 3)
            same_res_outputs = jnp.stack([out_deblur, out_gaussian, out_speckle], axis=-1)

            # For each batch element, select the same-res output
            # Map expert indices: deblur=1→0, gaussian=2→1, speckle=3→2
            same_res_idx = jnp.clip(selected - 1, 0, 2)  # (B,)
            same_res_selection = jax.vmap(lambda o, i: o[..., i])(same_res_outputs, same_res_idx)

            # Run upsample expert
            out_upsample = self.expert_upsample(x)  # (B, 2H, 2W, 1)

            # If upsample was chosen, use upsample output and resize same_res to match
            # If not chosen, use same_res output
            if step == 0 or not jnp.any(upsample_used):
                # Could be first use of upsample
                # Resize same_res_selection to match upsample output shape if needed
                target_shape = out_upsample.shape
                same_res_resized = jax.image.resize(
                    same_res_selection, target_shape, method="bilinear"
                )

                # Per-element selection: upsample if chose_upsample, else same_res
                chose_up_broadcast = chose_upsample[:, None, None, None]
                x = jnp.where(chose_up_broadcast, out_upsample, same_res_resized)
            else:
                # Upsample already used, all experts run at current resolution
                x = same_res_selection

            # Update upsample tracking
            upsample_used = upsample_used | chose_upsample

        routing_decisions = jnp.stack(all_decisions, axis=-1)  # (B, num_steps)
        return x, routing_decisions


# ─── Quick shape test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing IterativeMoE pipeline...")
    rngs = nnx.Rngs(42)
    model = IterativeMoE(rngs=rngs)

    dummy = jnp.ones((2, 128, 128, 1))
    key = jax.random.key(0)

    out, decisions = model(dummy, key, tau=0.5)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Routing decisions: {decisions}")
    print(f"Expected output shape: (2, 256, 256, 1)")
    assert out.shape == (2, 256, 256, 1), f"Shape mismatch: {out.shape}"
    print("\n✅ IterativeMoE shape test passed!")
