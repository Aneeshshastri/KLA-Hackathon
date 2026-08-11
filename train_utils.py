import jax
import jax.numpy as jnp
from flax import nnx
import optax

from evaluator import ModelEvaluator

# ── Loss ─────────────────────────────────────────────────────────────────

def charbonnier_loss_mean(
    pred: jax.Array, target: jax.Array, eps: float = 1e-3,
) -> jax.Array:
    diff = pred - target
    return jnp.mean(jnp.sqrt(diff * diff + eps * eps))

def FFT_loss_mean(
    pred: jax.Array, target: jax.Array, eps: float = 1e-3,
) -> jax.Array:
    pred_fft = jnp.fft.rfft2(pred, axes=(-3 , -2))
    target_fft = jnp.fft.rfft2(target, axes=(-3, -2))
    return jnp.mean(jnp.sqrt(jnp.abs(pred_fft - target_fft) ** 2 + eps ** 2))


    
def mixed_loss(
    pred: jax.Array, 
    target: jax.Array, 
    losses: tuple[callable, ...] = (FFT_loss_mean, charbonnier_loss_mean), 
    weights: tuple[float, ...] = (0.1, 0.9)
) -> jax.Array:
    
    total_loss = 0.0
    for loss_fn, weight in zip(losses, weights):
        total_loss += weight * loss_fn(pred, target)
        
    return jnp.asarray(total_loss)

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
    model,
    optimizer: nnx.Optimizer,
    x_norm: jax.Array,  # (B, H, W, C) normalised float32 LR image
    gt: jax.Array,      # (B, H*s, W*s, C) float32 HR ground truth
    loss_fn: callable = mixed_loss
) -> tuple[jax.Array, dict]:
    """Single training step with Charbonnier loss."""

    def compute_loss(model):
        pred, _z_d = model(x_norm)
        return loss_fn(pred, gt)

    loss, grads = nnx.value_and_grad(compute_loss)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step(
    model,
    x_norm: jax.Array,  # (B, H, W, C) normalised float32 LR image
    gt: jax.Array,      # (B, H*s, W*s, C) float32 HR ground truth
    loss_fn: callable = mixed_loss
) -> tuple[jax.Array, jax.Array]:
    """Validation step (no gradient computation)."""
    pred, _z_d = model(x_norm)
    loss = loss_fn(pred, gt)
    return loss, pred


@nnx.jit
def metric_val_step(
    model,
    x_norm: jax.Array,  # (B, H, W, C) normalised float32 LR image
    gt: jax.Array,    # (B, H*s, W*s, C) float32 HR ground truth
    evaluator: ModelEvaluator = ModelEvaluator(),
     
) -> tuple[dict, jax.Array]:
    """Metric-wise validation step (no gradient computation)."""
    
    pred, _z_d = model(x_norm)
    metrics = evaluator.validate(pred, gt)
    return metrics, pred