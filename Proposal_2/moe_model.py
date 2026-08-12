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
        """
        batch_size = x.shape[0]

        # We must map over the batch dimension to use lax.cond effectively
        @jax.vmap
        def process_single(x_single, key_single):
            # State: (x_128, x_256, is_256)
            x_128 = x_single
            x_256 = jnp.zeros((256, 256, 1), dtype=x_single.dtype)
            is_256 = jnp.array(False, dtype=jnp.bool_)
            
            all_decisions = []

            for step in range(self.num_steps):
                key_single, subkey = jax.random.split(key_single)

                # Router accepts both shapes (fully convolutional + global pool)
                # We expand dims to (1, H, W, C) for the router
                logits_128 = self.router(x_128[None])[0]
                logits_256 = self.router(x_256[None])[0]
                logits = jnp.where(is_256, logits_256, logits_128)

                # Mask out upsample expert if already used
                mask = jnp.where(is_256, -1e9, 0.0)
                logits = logits.at[EXPERT_UPSAMPLE].add(mask)

                # Hard routing
                if deterministic:
                    weights = argmax_hard(logits)
                else:
                    weights = gumbel_softmax_hard(logits, subkey, tau=tau)
                    
                selected = jnp.argmax(weights, axis=-1)
                all_decisions.append(selected)

                def do_256():
                    # Run same-res experts on x_256 (expanded to batch=1)
                    out_deblur = self.expert_deblur(x_256[None])[0]
                    out_gaussian = self.expert_gaussian(x_256[None])[0]
                    out_speckle = self.expert_speckle(x_256[None])[0]
                    
                    same_res = jnp.stack([out_deblur, out_gaussian, out_speckle], axis=-1)
                    mapped_idx = jnp.clip(selected - 1, 0, 2)
                    new_x_256 = same_res[..., mapped_idx]
                    return x_128, new_x_256, jnp.array(True, dtype=jnp.bool_)

                def do_128():
                    # Run same-res experts on x_128
                    out_deblur = self.expert_deblur(x_128[None])[0]
                    out_gaussian = self.expert_gaussian(x_128[None])[0]
                    out_speckle = self.expert_speckle(x_128[None])[0]
                    
                    same_res = jnp.stack([out_deblur, out_gaussian, out_speckle], axis=-1)
                    mapped_idx = jnp.clip(selected - 1, 0, 2)
                    new_x_128_same = same_res[..., mapped_idx]
                    
                    # Run upsample expert
                    out_up = self.expert_upsample(x_128[None])[0]
                    
                    is_up = (selected == EXPERT_UPSAMPLE)
                    
                    new_x_128 = jnp.where(is_up, x_128, new_x_128_same)
                    new_x_256 = jnp.where(is_up, out_up, x_256)
                    return new_x_128, new_x_256, is_up

                x_128, x_256, is_256 = jax.lax.cond(is_256, do_256, do_128)

            decisions_stack = jnp.stack(all_decisions, axis=0)
            
            # The final result must be 256x256. 
            # If for some reason upsample was NEVER chosen, we fallback to bicubic 
            # (though training should heavily penalize this)
            final_out = jnp.where(
                is_256, 
                x_256, 
                jax.image.resize(x_128, (256, 256, 1), method="bicubic")
            )
            return final_out, decisions_stack

        keys = jax.random.split(key, batch_size)
        out, decisions = process_single(x, keys)
        return out, decisions


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
