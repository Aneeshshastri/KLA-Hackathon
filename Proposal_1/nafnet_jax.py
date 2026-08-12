import flax.nnx as nnx
import jax.numpy as jnp
import jax

class SimpleGate(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x1 * x2

class NAFBlock(nnx.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0., rngs: nnx.Rngs = None):
        dw_channel = c * DW_Expand
        self.conv1 = nnx.Conv(c, dw_channel, kernel_size=(1, 1), padding='VALID', rngs=rngs)
        self.conv2 = nnx.Conv(dw_channel, dw_channel, kernel_size=(3, 3), padding=1, feature_group_count=dw_channel, rngs=rngs)
        self.conv3 = nnx.Conv(dw_channel // 2, c, kernel_size=(1, 1), padding='VALID', rngs=rngs)
        
        self.sca_conv = nnx.Conv(dw_channel // 2, dw_channel // 2, kernel_size=(1, 1), padding='VALID', rngs=rngs)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nnx.Conv(c, ffn_channel, kernel_size=(1, 1), padding='VALID', rngs=rngs)
        self.conv5 = nnx.Conv(ffn_channel // 2, c, kernel_size=(1, 1), padding='VALID', rngs=rngs)

        self.norm1 = nnx.LayerNorm(c, rngs=rngs)
        self.norm2 = nnx.LayerNorm(c, rngs=rngs)

        self.dropout1 = nnx.Dropout(drop_out_rate, deterministic=True) if drop_out_rate > 0. else lambda x: x
        self.dropout2 = nnx.Dropout(drop_out_rate, deterministic=True) if drop_out_rate > 0. else lambda x: x

        self.beta = nnx.Param(jnp.zeros((1, 1, 1, c)))
        self.gamma = nnx.Param(jnp.zeros((1, 1, 1, c)))

    def __call__(self, inp: jax.Array) -> jax.Array:
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        
        # SCA
        sca = jnp.mean(x, axis=(1, 2), keepdims=True)
        sca = self.sca_conv(sca)
        x = x * sca
        
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta.value

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma.value

class StandardNAFNet(nnx.Module):
    def __init__(self, img_channel: int = 3, width: int = 16, middle_blk_num: int = 1, enc_blk_nums: list = [], dec_blk_nums: list = [], rngs: nnx.Rngs = None):
        self.intro = nnx.Conv(img_channel, width, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.ending = nnx.Conv(width, img_channel, kernel_size=(3, 3), padding=1, rngs=rngs)

        chan = width
        encoders, downs, decoders, ups = [], [], [], []
        
        for num in enc_blk_nums:
            encoders.append(nnx.Sequential(*[NAFBlock(chan, rngs=rngs) for _ in range(num)]))
            downs.append(nnx.Conv(chan, 2 * chan, kernel_size=(2, 2), strides=(2, 2), padding='VALID', rngs=rngs))
            chan = chan * 2

        self.encoders = nnx.List(encoders)
        self.downs = nnx.List(downs)

        self.middle_blks = nnx.Sequential(*[NAFBlock(chan, rngs=rngs) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            ups.append(nnx.Conv(chan, chan * 2, kernel_size=(1, 1), padding='VALID', use_bias=False, rngs=rngs))
            chan = chan // 2
            decoders.append(nnx.Sequential(*[NAFBlock(chan, rngs=rngs) for _ in range(num)]))
            
        self.ups = nnx.List(ups)
        self.decoders = nnx.List(decoders)

    def pixel_shuffle(self, x: jax.Array, scale: int = 2) -> jax.Array:
        # x shape: (B, H, W, C)
        b, h, w, c = x.shape
        out_c = c // (scale * scale)
        x = x.reshape(b, h, w, scale, scale, out_c)
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape(b, h * scale, w * scale, out_c)
        return x

    def __call__(self, inp: jax.Array) -> jax.Array:
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = self.pixel_shuffle(x, 2)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x
