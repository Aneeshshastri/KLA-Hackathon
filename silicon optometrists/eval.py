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
from sklearn.model_selection import train_test_split
import grain

import lpips_jax
import dm_pix

from submission_model import Restoration_Pipeline_P3

# ==============================================================================
#                      Metrics from evaluator.py
# ==============================================================================

lpips_alex = lpips_jax.LPIPSEvaluator(net='alexnet', replicate=False)

def LPIPS(gt:jnp.array, im:jnp.array):
    gt = 2*jnp.concatenate([gt] * 3, axis=-1)-1
    im = 2*jnp.concatenate([im] * 3, axis=-1)-1
    distance = lpips_alex(gt, im)
    return distance

def SSIM(gt:jnp.array, im:jnp.array, window_size:int=11):
    return dm_pix.ssim(im, gt)
  
def PSNR(gt:jnp.array, im:jnp.array):
    max_pixel = 1.0
    mse = jnp.mean((gt-im)**2, axis=[1,2])
    psnr = 20 * jnp.log10(max_pixel/(jnp.sqrt(mse) + 1e-8))
    return psnr

# ==============================================================================
#                      DataLoader from dataloader.py
# ==============================================================================

class NpyPairDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, noisy_files: list, gt_files: list):
        self.noisy_files = list(noisy_files)
        self.gt_files = list(gt_files)
        
        self.noisy_data = [np.load(file).astype(np.float32) for file in self.noisy_files]
        self.gt_data = [np.load(file).astype(np.float32) for file in self.gt_files]
        
    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, index):
        return {
            "noisy_lr": self.noisy_data[index],
            "gt": self.gt_data[index]
        }

class GroupedDataLoader:
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
        batch_gt = []
        
        for idx in batch_indices:
            element = self.source[idx]
            noisy = element["noisy_lr"]
            gt = element["gt"]
            
            if noisy.ndim == 2:
                noisy = noisy[..., None]
            if gt.ndim == 2:
                gt = gt[..., None]
                
            batch_noisy.append(noisy)
            batch_gt.append(gt)
            
        return {
            "noisy_lr": np.stack(batch_noisy),
            "gt": np.stack(batch_gt)
        }

def create_src_dataloader(noisy_files, gt_files, batch_size):
    source = NpyPairDataSource(noisy_files, gt_files)
    loader = GroupedDataLoader(source, batch_size)
    return loader

# ==============================================================================
#                      Evaluation Logic
# ==============================================================================

def normalize(x, axis=(1, 2)):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True) + 1e-8
    return (x - mean) / std

def get_model():
    rngs = nnx.Rngs(42)
    model = Restoration_Pipeline_P3(
        in_channels=1, out_channels=1, hidden_dim=64, num_blocks=16,
        upsample_scale=2, deg_hidden_dim=16, deg_embed_dim=16,
        bottleneck_channels=96, rngs=rngs, num_experts=3, dropout_rate=0.0
    )
    
    rule = qwix.QuantizationRule(
        weight_qtype=jnp.float8_e4m3fn, 
        act_qtype=jnp.float8_e4m3fn,
        op_names=('dot_general',)
    )
    provider = qwix.PtqProvider([rule], disable_jit=True)
    
    dummy_input = jnp.ones((1, 256, 256, 1), dtype=jnp.float32)
    quantized_model = qwix.quantize_model(model, provider, dummy_input)
    
    model_path = Path("models/nafnet_fp8.msgpack")
    data = model_path.read_bytes()
    import msgpack
    from flax.serialization import _msgpack_ext_unpack
    restored_dict = msgpack.unpackb(data, ext_hook=_msgpack_ext_unpack, strict_map_key=False, raw=False)
    
    _, params, _ = nnx.split(quantized_model, nnx.Param, ...)
    from flax.serialization import from_state_dict
    params = from_state_dict(params, restored_dict)
    nnx.update(quantized_model, params)
    
    quantized_model.eval()
    return quantized_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate on Validation Split")
    parser.add_argument("train_noisy_dir", type=str, help="Path to full train noisy directory")
    parser.add_argument("train_gt_dir", type=str, help="Path to full train GT directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Validation split seed")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    
    noisy_dir = Path(args.train_noisy_dir)
    gt_dir = Path(args.train_gt_dir)
    
    all_noisy = sorted(list(noisy_dir.glob("*.npy")))
    all_gt = sorted(list(gt_dir.glob("*.npy")))
    
    assert len(all_noisy) > 0 and len(all_noisy) == len(all_gt), "Dataset mismatch or missing files."
    
    # Validation split aligned with train config
    _, val_noisy, _, val_gt = train_test_split(
        all_noisy, all_gt,
        test_size=args.val_split, random_state=args.seed
    )
    
    print(f"Loaded {len(val_noisy)} validation samples.")
    print("Loading model...")
    model = get_model()
    
    @nnx.jit
    def process_fn(x):
        restored, _ = model(x)
        return restored
    
    loader = create_src_dataloader(val_noisy, val_gt, args.batch_size)
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    print(f"Evaluating {len(val_noisy)} validation samples in {len(loader)} batches...")
    for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
        noisy_np = batch["noisy_lr"]
        gt_np = batch["gt"]
        
        x_norm = normalize(jnp.array(noisy_np), axis=(1, 2))
        pred = process_fn(x_norm)
        
        pred = jnp.clip(jnp.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        gt_jnp = jnp.array(gt_np)
        
        batch_psnr = PSNR(gt_jnp, pred)
        batch_ssim = SSIM(gt_jnp, pred)
        batch_lpips = LPIPS(gt_jnp, pred)
        
        all_psnr.extend(np.asarray(batch_psnr).flatten().tolist())
        all_ssim.extend(np.asarray(batch_ssim).flatten().tolist())
        all_lpips.extend(np.asarray(batch_lpips).flatten().tolist())
        
    print("\n--- Validation Results ---")
    print(f"Mean PSNR:  {np.mean(all_psnr):.4f}")
    print(f"Mean SSIM:  {np.mean(all_ssim):.4f}")
    print(f"Mean LPIPS: {np.mean(all_lpips):.4f}")

if __name__ == "__main__":
    main()
