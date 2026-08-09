import jax.numpy as jnp
from flax import nnx


# --- NAF Block Components ---
class SimpleGate(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        # Splits the channel dimension into two equal halves and multiplies them
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x1 * x2

class SimplifiedChannelAttention(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=channels, 
            out_features=channels, 
            kernel_size=(1, 1), 
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Global Average Pooling (B, H, W, C) -> (B, 1, 1, C)
        pool = jnp.mean(x, axis=(1, 2), keepdims=True)
        attn = self.conv(pool)
        return x * attn

class NAFBlock(nnx.Module):
    def __init__(self, c: int, dw_expansion: float = 2.0, ffn_expansion: float = 2.0, rngs: nnx.Rngs = None):
        expanded_dw_c = int(c * dw_expansion)
        expanded_ffn_c = int(c * ffn_expansion)
        
        # Spatial Block Components
        self.norm1 = nnx.LayerNorm(num_features=c, rngs=rngs)
        self.conv1 = nnx.Conv(c, expanded_dw_c, kernel_size=(1, 1), rngs=rngs)
        # Depthwise convolution maps channels 1:1
        self.conv2 = nnx.Conv(
            expanded_dw_c, expanded_dw_c, kernel_size=(3, 3), 
            feature_group_count=expanded_dw_c, padding='SAME', rngs=rngs
        )
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(expanded_dw_c // 2, rngs=rngs)
        self.conv3 = nnx.Conv(expanded_dw_c // 2, c, kernel_size=(1, 1), rngs=rngs)
        
        # FFN Block Components
        self.norm2 = nnx.LayerNorm(num_features=c, rngs=rngs)
        self.conv4 = nnx.Conv(c, expanded_ffn_c, kernel_size=(1, 1), rngs=rngs)
        self.sg2 = SimpleGate()
        self.conv5 = nnx.Conv(expanded_ffn_c // 2, c, kernel_size=(1, 1), rngs=rngs)
        
        # Learnable scaling parameters for residual connections initialized to 0
        self.beta1 = nnx.Param(jnp.zeros((1, 1, 1, c)))
        self.beta2 = nnx.Param(jnp.zeros((1, 1, 1, c)))

    def __call__(self, x: jax.Array) -> jax.Array:
        # Spatial Mixing
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = identity + x * self.beta1.value

        # Feed Forward Mixing
        identity = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = identity + x * self.beta2.value
        
        return x