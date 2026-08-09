import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Optimizer Initialization (Contextual Example)
# The `wrt` argument is a mandatory keyword-only argument that filters the PyTree 
# to ensure only the specified variables (e.g., nnx.Param) are updated.
#
# optimizer = nnx.Optimizer(
#     model, 
#     optax.chain(
#         optax.clip_by_global_norm(1.0), 
#         optax.adamw(learning_rate=1e-3)
#     ), 
#     wrt=nnx.Param
# )

@nnx.jit
def train_step(model: BaselineNAFNet, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
    """Executes a single training step with Huber Loss."""
    def loss_fn(model_ref):
        preds = model_ref(batch['noisy_lr'])
        loss = optax.huber_loss(preds, batch['clean_hr'], delta=1.0).mean()
        return loss

    # Calculate loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Apply gradient clipping and updates.
    # The update method strictly requires the model instance to apply the computed
    # updates back to the parameters mapped by `wrt`.
    optimizer.update(model, grads)
    
    return loss

@nnx.jit
def val_step(model: BaselineNAFNet, batch: dict) -> tuple[jax.Array, jax.Array]:
    """Executes a validation step without calculating gradients."""
    preds = model(batch['noisy_lr'])
    val_loss = optax.huber_loss(preds, batch['clean_hr'], delta=1.0).mean()
    
    return val_loss, preds