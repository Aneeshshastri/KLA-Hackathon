import argparse
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import qwix
from flax.serialization import msgpack_restore
from tqdm import tqdm

from submission_model import Restoration_Pipeline_P3

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
    import msgpack
    from flax.serialization import _msgpack_ext_unpack
    restored_dict = msgpack.unpackb(data, ext_hook=_msgpack_ext_unpack, strict_map_key=False, raw=False)
    
    # Update only Param state
    _, params, _ = nnx.split(quantized_model, nnx.Param, ...)
    nnx.update(params, restored_dict)
    nnx.update(quantized_model, params)
    quantized_model.eval()
    
    return quantized_model

def process_file(file_path, output_dir, process_fn):
    # Load
    noisy = np.load(file_path).astype(np.float32)
    original_shape = noisy.shape
    
    # Add channel dim if (H, W)
    if noisy.ndim == 2:
        noisy = noisy[..., None]
    
    # Add batch dim (1, H, W, 1)
    noisy = noisy[None, ...]
    
    # Normalize (uses standard deviation and mean of the noisy image)
    noisy_jnp = jnp.array(noisy)
    x_norm = normalize(noisy_jnp, axis=(1, 2))
    
    # Inference
    pred = process_fn(x_norm)
    pred_np = np.asarray(pred)
    
    # Remove batch dim
    pred_np = pred_np[0]
    
    # Remove channel dim if original was 2D
    if len(original_shape) == 2:
        pred_np = pred_np[..., 0]
        
    # Strictly enforce constraints
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
    pred_np = np.clip(pred_np, 0.0, 1.0)
    
    # Save
    out_path = output_dir / file_path.name
    np.save(out_path, pred_np)

def main():
    parser = argparse.ArgumentParser(description="KLA Hackathon Evaluation Script - Silicon Optometrists")
    parser.add_argument("input_dir", type=str, help="Directory containing noisy .npy files")
    parser.add_argument("output_dir", type=str, help="Directory to save restored .npy files")
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
        # Model returns (restored, degradation_latent)
        restored, _ = model(x)
        return restored
        
    print(f"Processing {len(files)} files...")
    for f in tqdm(files, desc="Restoring"):
        process_file(f, output_dir, process_fn)
        
    print(f"Done! Restored images saved to {output_dir}")

if __name__ == "__main__":
    main()
