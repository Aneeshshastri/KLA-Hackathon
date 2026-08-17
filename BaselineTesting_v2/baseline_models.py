import jax
import jax.numpy as jnp
from flax import nnx


class DegradationEncoder(nnx.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        embed_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = nnx.Conv(
            in_channels,
            hidden_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.proj = nnx.Linear(
            hidden_dim * 2,
            embed_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:

        h = nnx.leaky_relu(
            self.conv1(x),
            negative_slope=0.2,
        )

        h = nnx.leaky_relu(
            self.conv2(h),
            negative_slope=0.2,
        )

        pooled = jnp.mean(
            h,
            axis=(1, 2),
        )

        return self.proj(pooled)

class BlindDFCTokenEncoder(nnx.Module):

    # Input:  (B, H, W, C) normalised degraded image.
    # Output: (B, embed_dim)

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        embed_dim: int = 8,
        num_bands: int = 4,
        token_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.token_dim = token_dim

        # Band extractors and projectors – same as before
        self.band_extractors = nnx.List(
            [
                nnx.Conv(
                    in_features=in_channels,
                    out_features=hidden_dim,
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
                for _ in range(num_bands)
            ]
        )

        self.token_projections = nnx.List(
            [
                nnx.Linear(hidden_dim, token_dim, rngs=rngs)
                for _ in range(num_bands)
            ]
        )

        self.token_router = nnx.Linear(num_bands * token_dim, num_bands, rngs=rngs)

        # Final projection
        self.proj = nnx.Linear(token_dim, embed_dim, rngs=rngs)

        # a small MLP that takes a radial profile (e.g., 32 bins)
        # and predicts offsets for mask centres and bandwidths.
        profile_bins = 32
        self.profile_net = nnx.Sequential(
            nnx.Linear(profile_bins, 32, rngs=rngs),
            nnx.relu,
            nnx.Linear(32, 2 * num_bands, rngs=rngs),  # offsets for mu and sigma
        )
        # Initial (fixed) centres and bandwidths (normalised radius)
        self._init_centres = jnp.linspace(0.0, 1.0, num_bands)
        self._init_bandwidth = 1.0 / num_bands

    def _radial_profile(self, energy: jax.Array) -> jax.Array:
        """Compute a 1D radial profile (average energy per radial bin)."""
        h, w = energy.shape[-2], energy.shape[-1]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        radius = jnp.sqrt(((yy - cy) / jnp.maximum(cy, 1.0)) ** 2 +
                          ((xx - cx) / jnp.maximum(cx, 1.0)) ** 2)
        radius = radius / jnp.maximum(radius.max(), 1.0)  # normalise to [0,1]

        # Bin radii into 'profile_bins' and average energy in each bin
        bins = jnp.linspace(0, 1, 32)  # 32 bins
        indices = jnp.digitize(radius.ravel(), bins) - 1
        indices = jnp.clip(indices, 0, 31)
        # Use segment_sum for average
        energy_flat = energy.ravel()
        one_hot = jax.nn.one_hot(indices, num_classes=32)
        # average energy per bin: sum(energy * one_hot) / count
        counts = one_hot.sum(axis=0)
        sums = (energy_flat[:, None] * one_hot).sum(axis=0)
        profile = sums / (counts + 1e-8)
        return profile  # shape (32,)

    def _adaptive_masks(self, energy: jax.Array) -> jax.Array:
        """Generate Gaussian masks with adaptive centres/bandwidths."""

        h, w = energy.shape[-2], energy.shape[-1]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        radius = jnp.sqrt(((yy - cy) / jnp.maximum(cy, 1.0)) ** 2 +
                          ((xx - cx) / jnp.maximum(cx, 1.0)) ** 2)
        radius = radius / jnp.maximum(radius.max(), 1.0)

        # Compute radial profile
        profile = self._radial_profile(energy)  # (32,)

        # Predict offsets using the profile network
        offsets = self.profile_net(profile)  # (2*num_bands,)
        mu_offsets = offsets[:self.num_bands]
        sigma_offsets = offsets[self.num_bands:]

        # Adjust centres and bandwidths
        centres = self._init_centres + mu_offsets
        # Ensure centres stay within [0,1] and are sorted
        centres = jnp.clip(centres, 0.0, 1.0)
        centres = jnp.sort(centres)

        # Bandwidths: start from initial, add offset, clamp positive
        bandwidths = jnp.clip(self._init_bandwidth + sigma_offsets, 0.01, 0.5)

        # Create Gaussian masks
        masks = []
        for mu, sigma in zip(centres, bandwidths):
            mask = jnp.exp(-0.5 * ((radius - mu) / sigma) ** 2)
            masks.append(mask)
        masks = jnp.stack(masks, axis=0)  # (num_bands, H, W)
        return masks

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, H, W, C)
        x_gray = x[..., 0]  # Use grayscale for frequency analysis
        spectrum = jnp.fft.fftshift(jnp.fft.fft2(x_gray, axes=(-2, -1)), axes=(-2, -1))
        energy = jnp.log1p(jnp.abs(spectrum) ** 2)
        energy = energy / (jnp.mean(energy, axis=(-2, -1), keepdims=True) + 1e-6)

        # Generate adaptive masks per batch element
        # Use vmap to apply _adaptive_masks over batch
        masks = jax.vmap(self._adaptive_masks)(energy)  # (B, num_bands, H, W)

        tokens = []
        for band_idx in range(self.num_bands):
            # Extract band-wise energy map
            band_map = energy * masks[:, band_idx, :, :]  # (B, H, W)
            band_map = band_map[..., None]  # add channel dim -> (B, H, W, 1)
            features = self.band_extractors[band_idx](band_map)
            pooled = jnp.mean(features, axis=(1, 2))   # (B, hidden_dim)
            tokens.append(self.token_projections[band_idx](pooled))

        tokens = jnp.stack(tokens, axis=1)          # (B, num_bands, token_dim)
        flat_tokens = tokens.reshape(tokens.shape[0], -1)

        # Lightweight routing (similar to the original)
        routing_logits = self.token_router(flat_tokens)
        routing = jax.nn.softmax(routing_logits, axis=-1)
        z = jnp.sum(tokens * routing[..., None], axis=1)

        return self.proj(z)


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
        x = res_x + x * self.beta1

        # Feed Forward Mixing
        res_x = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = res_x + x * self.beta2

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



"""
==============================================================
Experiments E1-6 Utils and Pipelines
==============================================================
"""
class ZeroDegradationEncoder(nnx.Module):

    # Returns a zero conditioning vector.
    # Used for the unconditional bicubic-residual baseline.

    def __init__(
        self,
        embed_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        self.embed_dim = embed_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.zeros(
            (x.shape[0], self.embed_dim),
            dtype=x.dtype,
        )
class NAFTrunk(nnx.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_blocks: int,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        self.intro = nnx.Conv(
            in_channels,
            hidden_dim,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.blocks = nnx.List(
            [
                NAFBlock(
                    hidden_dim,
                    embed_dim,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )

    def __call__(
        self,
        x: jax.Array,
        z_d: jax.Array,
    ) -> jax.Array:
        x = self.intro(x)

        for block in self.blocks:
            x = block(x, z_d)

        return x



class HaarDWT(nnx.Module):

    def __call__(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        h = x.shape[1] - (x.shape[1] % 2)
        w = x.shape[2] - (x.shape[2] % 2)

        x = x[:, :h, :w, :]

        x00 = x[:, 0::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x11 = x[:, 1::2, 1::2, :]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)

        return ll, lh, hl, hh
class DetailFusion(nnx.Module):

    def __init__(
        self,
        channels: int,
        rngs: nnx.Rngs,
    ):
        self.proj = nnx.Conv(
            channels * 3,
            channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        self.gate = nnx.Conv(
            channels,
            channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(
        self,
        lh: jax.Array,
        hl: jax.Array,
        hh: jax.Array,
    ) -> jax.Array:

        detail = jnp.concatenate(
            [lh, hl, hh],
            axis=-1,
        )

        detail = self.proj(detail)
        gate = jax.nn.sigmoid(self.gate(detail))

        return detail * gate
class AdaptiveFrequencyBlock(nnx.Module):

    def __init__(
        self,
        channels: int,
        rngs: nnx.Rngs,
    ):
        self.boundary = nnx.Linear(
            channels,
            1,
            rngs=rngs,
        )

        self.low_proj = nnx.Conv(
            channels,
            channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        self.high_proj = nnx.Conv(
            channels,
            channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        self.merge = nnx.Conv(
            channels * 2,
            channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        self.scale = nnx.Param(
            jnp.zeros((1, 1, 1, channels))
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:

        b, h, w, c = x.shape

        pooled = jnp.mean(
            x,
            axis=(1, 2),
        )

        cutoff = jax.nn.sigmoid(
            self.boundary(pooled)
        )

        fft = jnp.fft.rfft2(
            x,
            axes=(1, 2),
        )

        freq_h = fft.shape[1]
        freq_w = fft.shape[2]

        yy, xx = jnp.meshgrid(
            jnp.arange(freq_h),
            jnp.arange(freq_w),
            indexing="ij",
        )

        radius = jnp.sqrt(
            (yy / jnp.maximum(freq_h - 1, 1)) ** 2
            + (xx / jnp.maximum(freq_w - 1, 1)) ** 2
        )

        radius = radius / (
            jnp.max(radius) + 1e-6
        )

        # Smooth image-dependent low-frequency mask.
        low_mask = jax.nn.sigmoid(
            (cutoff[:, None, None, :] - radius[None, :, :, None])
            * 20.0
        )

        high_mask = 1.0 - low_mask

        low = jnp.fft.irfft2(
            fft * low_mask,
            s=(h, w),
            axes=(1, 2),
        )

        high = jnp.fft.irfft2(
            fft * high_mask,
            s=(h, w),
            axes=(1, 2),
        )

        low = self.low_proj(low)
        high = self.high_proj(high)

        # Low-frequency context gates high-frequency features.
        low_gate = jax.nn.sigmoid(
            jnp.mean(low, axis=-1, keepdims=True)
        )

        high = high * low_gate

        merged = self.merge(
            jnp.concatenate(
                [low, high],
                axis=-1,
            )
        )

        return x + merged * self.scale

class FrequencyRouter(nnx.Module):

    def __init__(
        self,
        in_channels: int,
        num_experts: int = 3,
        rngs: nnx.Rngs = None,
    ):
        self.num_experts = num_experts

        self.conv1 = nnx.Conv(
            in_channels,
            8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            8,
            16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.proj = nnx.Linear(
            16,
            num_experts,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:

        h = nnx.leaky_relu(
            self.conv1(x),
            negative_slope=0.2,
        )

        h = nnx.leaky_relu(
            self.conv2(h),
            negative_slope=0.2,
        )

        pooled = jnp.mean(
            h,
            axis=(1, 2),
        )

        return jax.nn.softmax(
            self.proj(pooled),
            axis=-1,
        )
class RestorationPipeline_E1(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = deg_embed_dim

        self.degradation_encoder = ZeroDegradationEncoder(
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.nafnet = NAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.up_conv = nnx.Conv(
            hidden_dim,
            out_channels * (upsample_scale ** 2),
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        features = self.nafnet(
            x_norm,
            z_d,
        )

        residual = self.pixel_shuffle(
            self.up_conv(features)
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d
class RestorationPipeline_E2(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
    ):
        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.nafnet = NAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.up_conv = nnx.Conv(
            hidden_dim,
            out_channels * (upsample_scale ** 2),
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        features = self.nafnet(
            x_norm,
            z_d,
        )

        residual = self.pixel_shuffle(
            self.up_conv(features)
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d
class RestorationPipeline_E3(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
    ):
        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.dwt = HaarDWT()

        self.ll_intro = nnx.Conv(
            in_channels,
            hidden_dim,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.detail_intro = nnx.Conv(
            in_channels * 3,
            hidden_dim,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.detail_fusion = DetailFusion(
            hidden_dim,
            rngs=rngs,
        )

        self.blocks = nnx.List(
            [
                NAFBlock(
                    hidden_dim,
                    deg_embed_dim,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )

        self.up_conv = nnx.Conv(
            hidden_dim,
            out_channels * ((upsample_scale * 2) ** 2),
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale * 2,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        ll, lh, hl, hh = self.dwt(x_norm)

        ll_features = self.ll_intro(ll)

        detail_features = self.detail_intro(
            jnp.concatenate(
                [lh, hl, hh],
                axis=-1,
            )
        )

        detail_features = self.detail_fusion(
            detail_features,
            detail_features,
            detail_features,
        )

        features = ll_features + detail_features

        for block in self.blocks:
            features = block(
                features,
                z_d,
            )

        residual = self.pixel_shuffle(
            self.up_conv(features)
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d

class RestorationPipeline_E4(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
    ):
        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.nafnet = NAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.frequency_block = AdaptiveFrequencyBlock(
            hidden_dim,
            rngs=rngs,
        )

        self.up_conv = nnx.Conv(
            hidden_dim,
            out_channels * (upsample_scale ** 2),
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        features = self.nafnet(
            x_norm,
            z_d,
        )

        features = self.frequency_block(
            features,
        )

        residual = self.pixel_shuffle(
            self.up_conv(features)
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d

class RestorationPipeline_E5(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
        num_experts: int = 3,
    ):
        self.num_experts = num_experts

        self.degradation_encoder = DegradationEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.nafnet = NAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.router = FrequencyRouter(
            in_channels=in_channels,
            num_experts=num_experts,
            rngs=rngs,
        )

        self.expert_heads = nnx.List(
            [
                nnx.Conv(
                    hidden_dim,
                    out_channels * (upsample_scale ** 2),
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
                for _ in range(num_experts)
            ]
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        features = self.nafnet(
            x_norm,
            z_d,
        )

        routing = self.router(
            x_norm,
        )

        expert_residuals = []

        for expert_head in self.expert_heads:
            residual = self.pixel_shuffle(
                expert_head(features)
            )

            expert_residuals.append(
                residual
            )

        expert_residuals = jnp.stack(
            expert_residuals,
            axis=1,
        )

        residual = jnp.sum(
            expert_residuals
            * routing[:, :, None, None, None],
            axis=1,
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d

class RestorationPipeline_E6(nnx.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,
        rngs: nnx.Rngs,
        num_experts: int = 3,
    ):
        self.num_experts = num_experts

        self.degradation_encoder = BlindDFCTokenEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.nafnet = NAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        self.router = FrequencyRouter(
            in_channels=in_channels,
            num_experts=num_experts,
            rngs=rngs,
        )

        self.expert_heads = nnx.List(
            [
                nnx.Conv(
                    hidden_dim,
                    out_channels * (upsample_scale ** 2),
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
                for _ in range(num_experts)
            ]
        )

        self.pixel_shuffle = PixelShuffle(
            upsample_scale,
        )

    def __call__(
        self,
        x_norm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        z_d = self.degradation_encoder(x_norm)

        features = self.nafnet(
            x_norm,
            z_d,
        )

        routing = self.router(
            x_norm,
        )

        expert_residuals = []

        for expert_head in self.expert_heads:
            residual = self.pixel_shuffle(
                expert_head(features)
            )

            expert_residuals.append(
                residual
            )

        expert_residuals = jnp.stack(
            expert_residuals,
            axis=1,
        )

        residual = jnp.sum(
            expert_residuals
            * routing[:, :, None, None, None],
            axis=1,
        )

        bicubic = jax.image.resize(
            x_norm,
            residual.shape,
            method="bicubic",
        )

        pred = bicubic + residual

        return pred, z_d
