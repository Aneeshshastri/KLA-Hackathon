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

import grain

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
        
        x_norm = normalize(jnp.array(noisy), axis=(1, 2))
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
