import os
import torch
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
from flax import serialization
import urllib.request
import re

from nafnet_jax import StandardNAFNet

WEIGHTS_URL = "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width64.pth"
WEIGHTS_FILE = "NAFNet-SIDD-width64.pth"

def download_weights():
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Downloading {WEIGHTS_FILE}...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
    else:
        print(f"Found {WEIGHTS_FILE} locally.")

def transpose_conv(w):
    # PyTorch: (out_c, in_c, kH, kW)
    # JAX: (kH, kW, in_c, out_c)
    return jnp.transpose(w, (2, 3, 1, 0))

def port_weights():
    # 1. Download PyTorch weights
    download_weights()
    
    # 2. Load PyTorch state dict
    state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')
    if 'params' in state_dict:
        state_dict = state_dict['params']

    # 3. Instantiate JAX NAFNet
    print("Instantiating JAX NAFNet...")
    model = StandardNAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
        rngs=nnx.Rngs(0)
    )

    # 4. Extract model state and update
    _, state = nnx.split(model)

    def set_param(jax_path, pt_tensor, is_conv=False):
        curr = state
        for key in jax_path[:-1]:
            curr = curr[key]
        target_name = jax_path[-1]
        pt_np = pt_tensor.detach().numpy()
        if is_conv and target_name == 'kernel':
            pt_np = transpose_conv(pt_np)
        curr[target_name] = jnp.array(pt_np)

    for pt_key, pt_tensor in state_dict.items():
        parts = pt_key.split('.')
        jax_path = []
        is_conv = False
        
        if parts[0] == 'intro' or parts[0] == 'ending':
            target_name = 'kernel' if parts[-1] == 'weight' else parts[-1]
            jax_path = [parts[0], target_name]
            is_conv = (target_name == 'kernel')
            
        elif parts[0] in ['encoders', 'decoders']:
            list_idx = int(parts[1])
            block_idx = int(parts[2])
            target_name = parts[-1]
            
            if target_name == 'weight':
                if 'conv' in parts[3] or 'sca' in parts[3]:
                    target_name = 'kernel'
                    is_conv = True
                elif 'norm' in parts[3]:
                    target_name = 'scale'
            elif target_name in ['beta', 'gamma']:
                pt_tensor = pt_tensor.squeeze()
                
            block_name = 'sca_conv' if parts[3] == 'sca' else parts[3]
            
            if parts[-1] in ['beta', 'gamma']:
                jax_path = [parts[0], list_idx, 'layers', block_idx, target_name]
            else:
                jax_path = [parts[0], list_idx, 'layers', block_idx, block_name, target_name]
                
        elif parts[0] == 'middle_blks':
            block_idx = int(parts[1])
            target_name = parts[-1]
            
            if target_name == 'weight':
                if 'conv' in parts[2] or 'sca' in parts[2]:
                    target_name = 'kernel'
                    is_conv = True
                elif 'norm' in parts[2]:
                    target_name = 'scale'
            elif target_name in ['beta', 'gamma']:
                pt_tensor = pt_tensor.squeeze()
                
            block_name = 'sca_conv' if parts[2] == 'sca' else parts[2]
            
            if parts[-1] in ['beta', 'gamma']:
                jax_path = ['middle_blks', 'layers', block_idx, target_name]
            else:
                jax_path = ['middle_blks', 'layers', block_idx, block_name, target_name]
                
        elif parts[0] in ['downs', 'ups']:
            list_idx = int(parts[1])
            target_name = 'kernel' if parts[-1] == 'weight' else parts[-1]
            jax_path = [parts[0], list_idx, target_name]
            is_conv = (target_name == 'kernel')
            
        else:
            continue
            
        set_param(tuple(jax_path), pt_tensor, is_conv)

    nnx.update(model, state)
    out_file = "nafnet_pretrained.msgpack"
    
    # In flax.nnx, we need to convert State to a regular dict of arrays for serialization
    pure_dict = state.to_pure_dict() if hasattr(state, 'to_pure_dict') else state.to_dict()
    state_dict_for_save = serialization.to_state_dict(pure_dict)
    
    bytes_data = serialization.msgpack_serialize(state_dict_for_save)
    with open(out_file, 'wb') as f:
        f.write(bytes_data)
    print("Done porting to", out_file)

if __name__ == "__main__":
    port_weights()
