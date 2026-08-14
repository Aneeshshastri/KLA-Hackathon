import re

with open("model_util.py", "r") as f:
    content = f.read()

# The BlindDFCTokenEncoder ends around line 203.
# We will keep everything up to NAFBlock (inclusive) which ends around line 301.
# Then PixelShuffle starts at 310. We can find the start of BaselineNAFNet.

# Find BaselineNAFNet
baseline_idx = content.find("class BaselineNAFNet")

# Keep everything before BaselineNAFNet
new_content = content[:baseline_idx]

# Add our new classes
new_classes = """
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
        
        self.dropouts = nnx.List(
            [
                nnx.Dropout(dropout_rate, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )

    def __call__(
        self,
        x: jax.Array,
        z_d: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        x = self.intro(x)

        for block, dropout in zip(self.blocks, self.dropouts):
            x = block(x, z_d)
            x = dropout(x, deterministic=deterministic)

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
        rngs: nnx.Rngs,
        num_experts: int = 3,
        dropout_rate: float = 0.1,
    ):
        self.num_experts = num_experts
        self.deterministic = False

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
            dropout_rate=dropout_rate,
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
            deterministic=self.deterministic,
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
"""

new_content += new_classes

with open("model_util.py", "w") as f:
    f.write(new_content)
