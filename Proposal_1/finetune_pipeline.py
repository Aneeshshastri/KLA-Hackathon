import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax import serialization

from nafnet_jax import StandardNAFNet, NAFBlock

class UpsampleTail(nnx.Module):
    def __init__(self, c_in: int, scale: int, rngs: nnx.Rngs):
        self.scale = scale
        self.block1 = NAFBlock(c=c_in, rngs=rngs)
        self.block2 = NAFBlock(c=c_in, rngs=rngs)
        self.up_conv = nnx.Conv(c_in, c_in * (scale ** 2), kernel_size=(3, 3), padding=1, rngs=rngs)
        
    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.up_conv(x)
        
        # PixelShuffle (depth-to-space)
        b, h, w, c = x.shape
        scale = self.scale
        c_out = c // (scale**2)
        x = x.reshape((b, h, w, scale, scale, c_out))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape((b, h * scale, w * scale, c_out))
        return x

class RestorationPipeline_Finetune(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # 1. Initialize the Pre-trained NAFNet
        # The official NAFNet-SIDD-width64 was trained on 3-channel inputs with these specs:
        self.nafnet = StandardNAFNet(
            img_channel=3,
            width=64,
            enc_blk_nums=[2, 2, 4, 8],
            middle_blk_num=12,
            dec_blk_nums=[2, 2, 2, 2],
            rngs=rngs
        )
        
        # 2. Untrained Upsampling Tail
        # We need to map the 3-channel features back to the required output space
        # and do a 2x upsampling via PixelShuffle.
        self.tail_blocks = UpsampleTail(c_in=3, scale=2, rngs=rngs)
        self.final_proj = nnx.Conv(3, 1, kernel_size=(3, 3), padding=1, rngs=rngs)
        
    def __call__(self, x):
        """
        x shape: (B, H, W, 1) - KLA grayscale dataset
        """
        # 1. Adapt 1-channel input to 3-channel (copy along channel dim)
        x_3ch = jnp.repeat(x, 3, axis=-1)
        
        # 2. Forward pass through Pre-trained NAFNet
        features = self.nafnet(x_3ch)
        
        # 3. Forward pass through Untrained Upsampling Tail (2x resolution)
        features_up = self.tail_blocks(features)
        
        # 4. Final Projection to 1-channel output
        out = self.final_proj(features_up)
        
        return out
        
def load_pretrained_nafnet(model: RestorationPipeline_Finetune, msgpack_path="nafnet_pretrained.msgpack"):
    """
    Loads the ported pre-trained weights into the pipeline's NAFNet component.
    """
    with open(msgpack_path, "rb") as f:
        bytes_data = f.read()
        
    state_dict = serialization.msgpack_restore(bytes_data)
    
    _, state = nnx.split(model.nafnet)
    target_dict = state.to_pure_dict() if hasattr(state, 'to_pure_dict') else state.to_dict()
    
    # Restore using target structure so integer keys are properly mapped
    restored_dict = serialization.from_state_dict(target_dict, state_dict)
    
    nnx.update(model.nafnet, restored_dict)
    print("Pre-trained weights successfully loaded into NAFNet component!")
