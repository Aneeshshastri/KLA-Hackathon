import jax
import jax.numpy as jnp
from flax import nnx


class DegradationEncoder(nnx.Module):

    # Input:  (B, H, W, C) normalised degraded image.
    # Output: (B, embed_dim)

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        embed_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        self.embed_dim = embed_dim

        self.conv1 = nnx.Conv(in_channels, hidden_dim, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(hidden_dim, hidden_dim * 2, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.proj = nnx.Linear(hidden_dim * 2, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.leaky_relu(self.conv1(x), negative_slope=0.2)
        h = nnx.leaky_relu(self.conv2(h), negative_slope=0.2)
        pooled = jnp.mean(h, axis=(1, 2))
        z_d = self.proj(pooled)
        return z_d


"""
=========================================================
NAF Block Components
=========================================================
"""


class SimpleGate(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        # Splits the channel dimension into two equal halves and multiplies them
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x1 * x2


class SimplifiedChannelAttention(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs):

        # weird-- check optimal
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


class FiLM(nnx.Module):
    # Predicts a per-channel scale/shift from the degradation embedding and
    # applies it right after norm1, before spatial mixing starts — so the
    # block's first operation on the features is already degradation-aware.
    def __init__(self, embed_dim: int, channels: int, rngs: nnx.Rngs):
        self.to_scale_shift = nnx.Linear(embed_dim, channels * 2, rngs=rngs)

    def __call__(self, x: jax.Array, z_d: jax.Array) -> jax.Array:
        scale_shift = self.to_scale_shift(z_d)
        scale, shift = jnp.split(scale_shift[:, None, None, :], 2, axis=-1)
        return x * (1.0 + scale) + shift


class NAFBlock(nnx.Module):
    def __init__(self, c: int, embed_dim: int, dw_expansion: float = 2.0, ffn_expansion: float = 2.0, rngs: nnx.Rngs = None):
        expanded_dw_c = int(c * dw_expansion)
        expanded_ffn_c = int(c * ffn_expansion)

        # Spatial Block Components
        self.norm1 = nnx.LayerNorm(num_features=c, rngs=rngs)
        self.film = FiLM(embed_dim, c, rngs=rngs)
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

    def __call__(self, x: jax.Array, z_d: jax.Array) -> jax.Array:
        # Spatial Mixing
        res_x = x
        x = self.norm1(x)
        x = self.film(x, z_d)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = res_x + x * self.beta1.get_value()

        # Feed Forward Mixing
        res_x = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = res_x + x * self.beta2.get_value()

        return x


"""
==============================================================
NAFNet Model Architecture: (Primary Baseline, FiLM-conditioned)
==============================================================
"""


class PixelShuffle(nnx.Module):
    def __init__(self, scale: int):
        self.scale = scale

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Executes depth-to-space sub-pixel mapping.
        Input shape: (B, H, W, C * scale^2)
        Output shape: (B, H * scale, W * scale, C)
        """
        B, H, W, C_out_r2 = x.shape
        C_out = C_out_r2 // (self.scale ** 2)

        # Reshape to separate spatial and sub-pixel dimensions
        x = x.reshape((B, H, W, self.scale, self.scale, C_out))
        # Transpose to interleave the sub-pixels spatially
        x = x.transpose((0, 1, 3, 2, 4, 5))
        # Flatten the spatial dimensions
        x = x.reshape((B, H * self.scale, W * self.scale, C_out))
        return x


class BaselineNAFNet(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, num_blocks: int, upsample_scale: int, embed_dim: int, rngs: nnx.Rngs):
        # Initial feature extraction
        self.intro = nnx.Conv(in_channels, hidden_dim, kernel_size=(3, 3), padding='SAME', rngs=rngs)

        # Sequential NAFBlocks, each FiLM-conditioned on the degradation embedding
        self.blocks = nnx.List([NAFBlock(hidden_dim, embed_dim, rngs=rngs) for _ in range(num_blocks)])

        # Upsampling projection layer (Channels must equal out_channels * scale^2)
        self.up_conv = nnx.Conv(
            hidden_dim, out_channels * (upsample_scale ** 2),
            kernel_size=(3, 3), padding='SAME', rngs=rngs
        )
        self.pixel_shuffle = PixelShuffle(upsample_scale)

    def __call__(self, x: jax.Array, z_d: jax.Array) -> jax.Array:
        x = self.intro(x)

        for block in self.blocks:
            x = block(x, z_d)

        x = self.up_conv(x)
        x = self.pixel_shuffle(x)

        return x


class RestorationPipeline(nnx.Module):
    # Bundles the degradation encoder and NAFNet as one checkpointable unit.
    def __init__(self, in_channels, out_channels, hidden_dim, num_blocks, upsample_scale,
                 deg_hidden_dim, deg_embed_dim, rngs: nnx.Rngs):
        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels, hidden_dim=deg_hidden_dim, embed_dim=deg_embed_dim, rngs=rngs,
        )
        self.nafnet = BaselineNAFNet(
            in_channels=in_channels, out_channels=out_channels, hidden_dim=hidden_dim,
            num_blocks=num_blocks, upsample_scale=upsample_scale, embed_dim=deg_embed_dim, rngs=rngs,
        )

    def __call__(self, x_norm: jax.Array, y_upsampled: jax.Array) -> tuple[jax.Array, jax.Array]:
        z_d = self.degradation_encoder(x_norm)
        corruption = self.nafnet(x_norm, z_d)
        pred = y_upsampled - corruption
        return pred, z_d