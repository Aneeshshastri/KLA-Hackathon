import jax
import jax.numpy as jnp
from flax import nnx
import optax

from model_util import RestorationPipeline_2


# ── Loss ─────────────────────────────────────────────────────────────────

def charbonnier_loss_mean(
    pred: jax.Array, target: jax.Array, eps: float = 1e-3,
) -> jax.Array:
    diff = pred - target
    return jnp.mean(jnp.sqrt(diff * diff + eps * eps))


# ── Bicubic upsample helper ─────────────────────────────────────────────

def compute_bicubic_upsample(
    noisy: jax.Array, target_hw: tuple[int, int],
) -> jax.Array:
    """Resizes the degraded input to the target HR resolution via bicubic interpolation."""
    b, h, w, c = noisy.shape
    return jax.image.resize(noisy, (b, target_hw[0], target_hw[1], c), method="bicubic")


# ── Device helper ────────────────────────────────────────────────────────

def to_device(batch: dict) -> dict:
    return jax.tree.map(jax.device_put, batch)


# ── Training / validation steps ─────────────────────────────────────────
# Type conversion and normalisation are handled by the caller *outside*
# the JIT boundary so that the compiled functions see fixed dtypes/shapes.
# The overhead is negligible: these ops dispatch as single XLA primitives
# and are already cached after the first call.

@nnx.jit
def train_step(
    model: RestorationPipeline_2,
    optimizer: nnx.Optimizer,
    x_norm: jax.Array,  # (B, H, W, C) normalised float32 LR image
    gt: jax.Array,      # (B, H*s, W*s, C) float32 HR ground truth
) -> jax.Array:
    """Single training step with Charbonnier loss."""
    def loss_fn(model_ref):
        pred, _z_d = model_ref(x_norm)
        return charbonnier_loss_mean(pred, gt)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step(
    model: RestorationPipeline_2,
    x_norm: jax.Array,  # (B, H, W, C) normalised float32 LR image
    gt: jax.Array,      # (B, H*s, W*s, C) float32 HR ground truth
) -> tuple[jax.Array, jax.Array]:
    """Validation step (no gradient computation)."""
    pred, _z_d = model(x_norm)
    loss = charbonnier_loss_mean(pred, gt)
    return loss, pred