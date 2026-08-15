"""
MoE Expert definitions for Proposal 2.

4 specialized experts, all based on RestorationPipeline_E6 from BaselineTesting_V2:
  1. UpsampleExpert  — 128×128 → 256×256 (upsample_scale=2)
  2. DeblurExpert    — resolution-preserving (upsample_scale=1)
  3. GaussianDenoiseExpert — resolution-preserving (upsample_scale=1)
  4. SpeckleDenoiseExpert  — resolution-preserving (upsample_scale=1)
"""

import sys
from pathlib import Path

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from flax import nnx

from BaselineTesting_v2.baseline_models import RestorationPipeline_E6

# ─── Expert Configuration ────────────────────────────────────────────────

EXPERT_CONFIGS = {
    "upsample": {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 32,
        "num_blocks": 4,      # smaller than Baseline_2 configuration
        "upsample_scale": 2,  # 128→256 (2x super-res)
        "deg_hidden_dim": 16,
        "deg_embed_dim": 8,
    },
    "deblur": {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 32,
        "num_blocks": 4,
        "upsample_scale": 1,  # resolution-preserving
        "deg_hidden_dim": 16,
        "deg_embed_dim": 8,
    },
    "gaussian": {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 32,
        "num_blocks": 4,
        "upsample_scale": 1,
        "deg_hidden_dim": 16,
        "deg_embed_dim": 8,
    },
    "speckle": {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 32,
        "num_blocks": 4,
        "upsample_scale": 1,
        "deg_hidden_dim": 16,
        "deg_embed_dim": 8,
    },
}

# ─── Expert Module ────────────────────────────────────────────────────────

class Expert(nnx.Module):
    """
    Wrapper around RestorationPipeline_E6 to return only the prediction (discarding z_d).
    """
    def __init__(self, rngs: nnx.Rngs, **kwargs):
        self.model = RestorationPipeline_E6(rngs=rngs, **kwargs)

    def __call__(self, x_norm: jax.Array) -> jax.Array:
        pred, _z_d = self.model(x_norm)
        return pred

def create_expert(expert_type: str, rngs: nnx.Rngs) -> Expert:
    """Factory function to create an expert by type name."""
    cfg = EXPERT_CONFIGS[expert_type]
    return Expert(rngs=rngs, **cfg)

def create_all_experts(rngs: nnx.Rngs) -> dict[str, Expert]:
    """Create all 4 experts."""
    return {name: create_expert(name, rngs) for name in EXPERT_CONFIGS}

# ─── Quick shape test ────────────────────────────────────────────────────

if __name__ == "__main__":
    rngs = nnx.Rngs(42)
    experts = create_all_experts(rngs)

    dummy_lr = jnp.ones((2, 128, 128, 1))
    dummy_hr = jnp.ones((2, 256, 256, 1))

    for name, expert in experts.items():
        if name == "upsample":
            out = expert(dummy_lr)
            print(f"{name}: input {dummy_lr.shape} → output {out.shape}")
            assert out.shape == (2, 256, 256, 1), f"Expected (2,256,256,1), got {out.shape}"
        else:
            # Test at 128
            out_lr = expert(dummy_lr)
            print(f"{name} @128: input {dummy_lr.shape} → output {out_lr.shape}")
            assert out_lr.shape == (2, 128, 128, 1), f"Expected (2,128,128,1), got {out_lr.shape}"

            # Test at 256 (post-upsample steps)
            out_hr = expert(dummy_hr)
            print(f"{name} @256: input {dummy_hr.shape} → output {out_hr.shape}")
            assert out_hr.shape == (2, 256, 256, 1), f"Expected (2,256,256,1), got {out_hr.shape}"

    print("\n✅ All E6 expert shape tests passed!")
