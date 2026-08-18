import argparse
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import qwix
from tqdm import tqdm
import msgpack
from flax.serialization import _msgpack_ext_unpack
import grain


# ══════════════════════════════════════════════════════════════════════════
#  Model definitions (from model_util.py)
# ══════════════════════════════════════════════════════════════════════════

class BlindDFCTokenEncoder(nnx.Module):
    def __init__(
        self,
        in_channels: int,  # Kept for compatibility, but band extractors use in_features=1
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
        self.band_extractors = nnx.List([
                nnx.Conv(
                    in_features=1, 
                    out_features=hidden_dim,
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
                for _ in range(num_bands)
            ])

        self.token_projections = nnx.List([
                nnx.Linear(hidden_dim, token_dim, rngs=rngs)
                for _ in range(num_bands)
            ])
        

        self.token_router = nnx.Linear(num_bands * token_dim, num_bands, rngs=rngs)

        # Final projection
        self.proj = nnx.Linear(token_dim, embed_dim, rngs=rngs)

        # a small MLP that takes a radial profile (e.g., 32 bins)
        # and predicts offsets for mask centres and bandwidths.
        profile_bins = 32
        self.profile_net = nnx.Sequential(
            nnx.Linear(profile_bins, 32, rngs=rngs),
            nnx.relu,
            nnx.Linear(32, 2 * num_bands, rngs=rngs),
        )
        # Initial (fixed) centres and bandwidths (normalised radius)
        self._init_centres = jnp.linspace(0.0, 1.0, num_bands)
        self._init_bandwidth = 1.0 / num_bands

    def _radial_profile(self, energy: jax.Array) -> jax.Array:
        """Compute a 32-bin radial energy profile."""
        h, w = energy.shape[-2], energy.shape[-1]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        radius = jnp.sqrt(
            ((yy - cy) / jnp.maximum(cy, 1.0)) ** 2
            + ((xx - cx) / jnp.maximum(cx, 1.0)) ** 2
        )
        radius = radius / (jnp.max(radius) + 1e-6)

        # 32 intervals require 33 edges.
        bins = jnp.linspace(0.0, 1.0, 33)
        radius_flat = radius.reshape(-1)
        indices = jnp.sum(radius_flat[:, None] >= bins[None, :], axis=1) - 1
        indices = jnp.clip(indices, 0, 31)
        energy_flat = energy.reshape(-1)
        one_hot = jax.nn.one_hot(indices, num_classes=32)
        counts = jnp.sum(one_hot, axis=0)
        sums = jnp.sum(energy_flat[:, None] * one_hot, axis=0)
        return sums / (counts + 1e-6)

    def _adaptive_masks(self, energy: jax.Array) -> jax.Array:
        """Generate stable, differentiable adaptive Gaussian masks."""
        h, w = energy.shape[-2], energy.shape[-1]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        radius = jnp.sqrt(
            ((yy - cy) / jnp.maximum(cy, 1.0)) ** 2
            + ((xx - cx) / jnp.maximum(cx, 1.0)) ** 2
        )
        radius = radius / (jnp.max(radius) + 1e-6)

        profile = self._radial_profile(energy)
        offsets = self.profile_net(profile)
        centre_offsets = offsets[:self.num_bands]
        bandwidth_offsets = offsets[self.num_bands:]

        # Bounded centre movement around the initial bands.
        centres = self._init_centres + 0.15 * jnp.tanh(centre_offsets)
        centres = jnp.clip(centres, 0.0, 1.0)

        # Positive, bounded bandwidth adjustment.
        bandwidths = self._init_bandwidth * jnp.exp(0.25 * jnp.tanh(bandwidth_offsets))
        bandwidths = jnp.clip(bandwidths, 0.02, 0.5)

        # Keep centre/bandwidth assignments together.
        order = jnp.argsort(centres)
        centres = centres[order]
        bandwidths = bandwidths[order]

        masks = []
        for band_index in range(self.num_bands):
            centre = centres[band_index]
            bandwidth = bandwidths[band_index]
            mask = jnp.exp(-0.5 * ((radius - centre) / bandwidth) ** 2)
            masks.append(mask)
        return jnp.stack(masks, axis=0)

    def __call__(self, x: jax.Array) -> jax.Array:
        x_gray = x[..., 0]
        spectrum = jnp.fft.fftshift(jnp.fft.fft2(x_gray, axes=(-2, -1)), axes=(-2, -1))
        energy = jnp.log1p(jnp.abs(spectrum) ** 2)
        energy = energy / (jnp.mean(energy, axis=(-2, -1), keepdims=True) + 1e-6)

        # Use a list comprehension over the batch dim instead of jax.vmap to avoid 
        # flax.errors.TraceContextError when qwix mutates Linear layer states during quantization
        masks = jnp.stack([self._adaptive_masks(energy[b]) for b in range(energy.shape[0])], axis=0)
        tokens = []
        for band_idx in range(self.num_bands):
            band_map = energy * masks[:, band_idx, :, :]
            band_map = band_map[..., None]
            features = self.band_extractors[band_idx](band_map)
            pooled = jnp.mean(features, axis=(1, 2))
            tokens.append(self.token_projections[band_idx](pooled))
        tokens = jnp.stack(tokens, axis=1)
        flat_tokens = tokens.reshape(tokens.shape[0], -1)

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
    def __init__(self, c: int, embed_dim: int, dw_expansion: float = 2.0, ffn_expansion: float = 2.0,
        dilation: int = 1, rngs: nnx.Rngs = None):
        expanded_dw_c = int(c * dw_expansion)
        expanded_ffn_c = int(c * ffn_expansion)

        # Spatial Block Components
        self.norm1 = nnx.LayerNorm(num_features=c, rngs=rngs)
        self.film = FiLM(embed_dim, c, rngs=rngs)
        self.conv1 = nnx.Conv(c, expanded_dw_c, kernel_size=(1, 1), rngs=rngs)

        # Depthwise convolution maps channels 1:1, dilation widens receptive field
        # without adding parameters or downsampling.
        self.conv2 = nnx.Conv(
            expanded_dw_c, expanded_dw_c, kernel_size=(3, 3),
            kernel_dilation=(dilation, dilation),
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



class NAFTrunk(nnx.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_blocks: int,
        embed_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.1,
    ):
        self.intro = nnx.Conv(
            in_channels,
            hidden_dim,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.blocks =nnx.List([
                NAFBlock(
                    hidden_dim,
                    embed_dim,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ])
        
        self.dropouts =nnx.List([
                nnx.Dropout(dropout_rate, rngs=rngs)
                for _ in range(num_blocks)
            ])

    def __call__(
        self,
        x: jax.Array,
        z_d: jax.Array,
    ) -> jax.Array:
        x = self.intro(x)

        for block, dropout in zip(self.blocks, self.dropouts):
            x = block(x, z_d)
            x = dropout(x)

        return x
class MultiScaleNAFTrunk(nnx.Module):
    """
    Lightweight two-scale NAF trunk with per-block dilation cycling.

    Each stage (shallow, bottleneck, output) cycles its blocks through a
    fixed dilation pattern instead of every block using dilation=1. This
    grows the receptive field with depth inside a stage without adding
    downsampling steps or parameters — a cheap substitute for going deeper
    into a full multi-level U-Net while still gaining multi-scale context.

    Input:
        (B, H, W, C) LR image

    Output:
        (B, H, W, hidden_dim) LR features
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_blocks: int,
        embed_dim: int,
        rngs: nnx.Rngs,
        bottleneck_channels: int = 96,
        dropout_rate: float = 0.0,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 8),
    ):
        self.hidden_dim = hidden_dim
        self.bottleneck_channels = bottleneck_channels

        shallow_blocks = max(1, num_blocks // 4)
        bottleneck_blocks = max(1, num_blocks // 2)
        output_blocks = max(1, num_blocks - shallow_blocks - bottleneck_blocks)

        dilations = lambda n: [dilation_cycle[i % len(dilation_cycle)] for i in range(n)]

        self.intro = nnx.Conv(in_channels, hidden_dim, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        self.shallow_blocks = nnx.List([NAFBlock(hidden_dim, embed_dim, dilation=d, rngs=rngs) for d in dilations(shallow_blocks)])
        

        self.downsample = nnx.Conv(hidden_dim, bottleneck_channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs,        )

        self.bottleneck_blocks =nnx.List([NAFBlock(bottleneck_channels, embed_dim, dilation=d, rngs=rngs) for d in dilations(bottleneck_blocks)])
    

        self.up_projection = nnx.Conv(bottleneck_channels, hidden_dim, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.skip_projection = nnx.Conv(hidden_dim, hidden_dim, kernel_size=(1, 1), padding="SAME", rngs=rngs)

        self.output_blocks = nnx.List([NAFBlock(hidden_dim, embed_dim, dilation=d, rngs=rngs) for d in dilations(output_blocks)])

        self.dropouts = nnx.List([nnx.Dropout(dropout_rate, rngs=rngs) for _ in range(shallow_blocks + bottleneck_blocks + output_blocks)])


    def __call__(self, x: jax.Array, z_d: jax.Array) -> jax.Array:
        x = self.intro(x)
        dropout_index = 0
        shallow_skip = x

        for block in self.shallow_blocks:
            x = block(x, z_d)
            x = self.dropouts[dropout_index](x)
            dropout_index += 1

        shallow_skip = self.skip_projection(shallow_skip)

        _, h, w, _ = x.shape
        x = self.downsample(x)

        for block in self.bottleneck_blocks:
            x = block(x, z_d)
            x = self.dropouts[dropout_index](x)
            dropout_index += 1

        x = jax.image.resize(x, shape=(x.shape[0], h, w, self.bottleneck_channels), method="linear")
        x = self.up_projection(x)
        x = x + shallow_skip

        for block in self.output_blocks:
            x = block(x, z_d)
            x = self.dropouts[dropout_index](x)
            dropout_index += 1

        return x

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
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            32,
            64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.proj = nnx.Linear(
            64,
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
def bounded_output(
    x: jax.Array,
    lower: float = 0.0,
    upper: float = 1.0,
) -> jax.Array:
    """
    Forward pass:
        output is clipped to the clean target range.

    Backward pass:
        gradient is passed through unchanged.
    """
    clipped = jnp.clip(
        x,
        lower,
        upper,
    )

    return x + jax.lax.stop_gradient(
        clipped - x
    )

class DWTDownsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs):
        self.dwt = HaarDWT()
        self.squeeze = nnx.Conv(in_channels * 4, out_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs)
        self.mix = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dwt(x)
        x = self.squeeze(x)
        x = x + nnx.leaky_relu(self.mix(x), negative_slope=0.2)
        return x

class HaarDWT(nnx.Module):
    """
    Computes a 2D Haar discrete wavelet transform (DWT).
    Downsamples the spatial resolution by 2x and outputs 4 subbands:
    LL, LH, HL, HH concatenated along the channel axis.
    Input: (B, H, W, C)
    Output: (B, H/2, W/2, 4C)
    """
    def __call__(self, x: jax.Array) -> jax.Array:
        # Strided slicing to get the four corners of each 2x2 patch
        x00 = x[:, 0::2, 0::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x11 = x[:, 1::2, 1::2, :]
        
        # Haar basis combinations (normalized by 0.5)
        LL = (x00 + x10 + x01 + x11) * 0.5
        LH = (x00 + x10 - x01 - x11) * 0.5
        HL = (x00 - x10 + x01 - x11) * 0.5
        HH = (x00 - x10 - x01 + x11) * 0.5
        
        return jnp.concatenate([LL, LH, HL, HH], axis=-1)

class Restoration_Pipeline_P3(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        num_blocks,
        upsample_scale,
        deg_hidden_dim,
        deg_embed_dim,    
        bottleneck_channels,
        rngs: nnx.Rngs,
        num_experts: int = 3,
        dropout_rate: float = 0.05
    
    ):
        self.num_experts = num_experts

        self.degradation_encoder = BlindDFCTokenEncoder(
            in_channels=in_channels,
            hidden_dim=deg_hidden_dim,
            embed_dim=deg_embed_dim,
            rngs=rngs,
        )

        # self.nafnet = NAFTrunk(
        #     in_channels=in_channels,
        #     hidden_dim=hidden_dim,
        #     num_blocks=num_blocks,
        #     embed_dim=deg_embed_dim,
        #     rngs=rngs,
        #     dropout_rate=dropout_rate,
        # )
        self.nafnet = MultiScaleNAFTrunk(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            embed_dim=deg_embed_dim,
            rngs=rngs,
            bottleneck_channels=bottleneck_channels,
            dropout_rate=dropout_rate,
        )

        self.router = FrequencyRouter(
            in_channels=in_channels,
            num_experts=num_experts,
            rngs=rngs,
        )

        self.expert_heads = nnx.List([
                nnx.Conv(
                    hidden_dim,
                    out_channels * (upsample_scale ** 2),
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                )
                for _ in range(num_experts)
            ])

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

        # pred = bicubic + residual

        # return pred, z_d

        raw_pred = bicubic + residual

        pred = bounded_output(
            raw_pred,
        )

        return pred, z_d

def normalize(x, axis=(1, 2)):
    """Normalize input using Z-score per channel, matching training logic."""
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True) + 1e-8
    return (x - mean) / std

def get_model():
    """Initializes model, applies FP8 rule, and loads quantized state."""
    # Must match training configuration
    rngs = nnx.Rngs(42)
    model = Restoration_Pipeline_P3(
        in_channels=1,
        out_channels=1,
        hidden_dim=64,
        num_blocks=16,
        upsample_scale=2,
        deg_hidden_dim=16,
        deg_embed_dim=16,
        bottleneck_channels=96,
        rngs=rngs,
        num_experts=3,
        dropout_rate=0.0
    )
    
    # Re-quantize to initialize FP8 param structures
    rule = qwix.QuantizationRule(
        weight_qtype=jnp.float8_e4m3fn, 
        act_qtype=jnp.float8_e4m3fn,
        op_names=("dot_general",)
    )
    provider = qwix.PtqProvider([rule])
    dummy_input = jnp.ones((1, 256, 256, 1), dtype=jnp.float32)
    quantized_model = qwix.quantize_model(model, provider, dummy_input)
    
    # Load FP8 msgpack
    model_path = Path(__file__).parent / "models" / "nafnet_fp8.msgpack"
    if not model_path.exists():
        raise FileNotFoundError(f"Quantized model not found at {model_path}. Please run the quantization cell in the training notebook.")
    
    data = model_path.read_bytes()

    restored_dict = msgpack.unpackb(data, ext_hook=_msgpack_ext_unpack, strict_map_key=False, raw=False)
    
    # Update the full state (including QuantParam scales)
    nnx.update(quantized_model, restored_dict)
    
    quantized_model.eval()
    
    return quantized_model

class NpySingleDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, noisy_files: list):
        self.noisy_files = list(noisy_files)
        self.noisy_data = [np.load(file).astype(np.float32) for file in self.noisy_files]
        
    def __len__(self):
        return len(self.noisy_data)
        
    def __getitem__(self, index):
        return {
            "noisy_lr": self.noisy_data[index],
            "filename": self.noisy_files[index].name
        }

class GroupedInferenceDataLoader:
    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = batch_size
        
        self.indices_by_shape = {}
        for idx in range(len(source)):
            shape = source.noisy_data[idx].shape[:2]
            if shape not in self.indices_by_shape:
                self.indices_by_shape[shape] = []
            self.indices_by_shape[shape].append(idx)
            
        self.total_batches = 0
        for shape, indices in self.indices_by_shape.items():
            self.total_batches += (len(indices) + batch_size - 1) // batch_size

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        batches = []
        for shape, indices in self.indices_by_shape.items():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i+self.batch_size])
                
        self.current_batch_idx = 0
        self.batches = batches
        return self

    def __next__(self):
        if self.current_batch_idx >= len(self.batches):
            raise StopIteration
            
        batch_indices = self.batches[self.current_batch_idx]
        self.current_batch_idx += 1
        
        batch_noisy = []
        batch_filenames = []
        
        for idx in batch_indices:
            element = self.source[idx]
            noisy = element["noisy_lr"]
            if noisy.ndim == 2:
                noisy = noisy[..., None]
            batch_noisy.append(noisy)
            batch_filenames.append(element["filename"])
            
        return {
            "noisy_lr": np.stack(batch_noisy),
            "filenames": batch_filenames
        }

def create_inference_dataloader(noisy_files, batch_size):
    source = NpySingleDataSource(noisy_files)
    loader = GroupedInferenceDataLoader(source, batch_size)
    return loader

def main():
    parser = argparse.ArgumentParser(description="KLA Hackathon Evaluation Script - Silicon Optometrists")
    parser.add_argument("input_dir", type=str, help="Directory containing noisy .npy files")
    parser.add_argument("output_dir", type=str, help="Directory to save restored .npy files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return
        
    print("Loading FP8 quantized model...")
    model = get_model()
    
    @nnx.jit
    def process_fn(x):
        restored, _ = model(x)
        return restored
        
    loader = create_inference_dataloader(files, args.batch_size)
    
    print(f"Processing {len(files)} files in {len(loader)} batches...")
    for batch in tqdm(loader, desc="Restoring", total=len(loader)):
        noisy = batch["noisy_lr"]
        filenames = batch["filenames"]
        
        x_norm = jnp.array(noisy)
        pred = process_fn(x_norm)
        pred_np = np.asarray(pred)
        
        # Enforce constraints
        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
        pred_np = np.clip(pred_np, 0.0, 1.0)
        
        for i, filename in enumerate(filenames):
            # Check original shape from file data to see if it was 2D
            # If it was 2D originally, remove the channel dimension
            original_shape = np.load(input_dir / filename, mmap_mode='r').shape
            
            img_out = pred_np[i]
            if len(original_shape) == 2:
                img_out = img_out[..., 0]
            np.save(output_dir / filename, img_out)
        
    print(f"Done! Restored images saved to {output_dir}")

if __name__ == "__main__":
    main()
