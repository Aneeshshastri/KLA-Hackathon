import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints
import optax
from PIL import Image

from moe_model import IterativeMoE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/moe_router")
    parser.add_argument("--data-dir", type=str, default="../test")
    parser.add_argument("--out-dir", type=str, default="eval_results")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test images to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ── Load Data ──
    data_dir = Path(args.data_dir)
    noisy_dir = data_dir / "NoisyLR"
    gt_dir = data_dir / "GT"
    
    if not noisy_dir.exists() or not gt_dir.exists():
        print(f"Error: Data directories not found in {data_dir}")
        return
        
    noisy_files = sorted(noisy_dir.glob("*.npy"))
    gt_files = sorted(gt_dir.glob("*.npy"))
    
    if len(noisy_files) == 0:
        print("No test data found.")
        return
        
    num_samples = min(args.num_samples, len(noisy_files))
    noisy_files = noisy_files[:num_samples]
    gt_files = gt_files[:num_samples]
    
    print(f"Evaluating on {num_samples} samples...")
    
    # ── Initialize Model ──
    rngs = nnx.Rngs(0)
    model = IterativeMoE(rngs=rngs)
    
    # Load router checkpoint
    # Note: For nnx, we typically extract the state and load it
    state = nnx.state(model)
    print(f"Loading MoE router checkpoint from {args.ckpt_dir}...")
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=args.ckpt_dir, target=state)
    
    if restored_state is state:
        print("Warning: No checkpoint found. Using random weights.")
    else:
        nnx.update(model, restored_state)
        print("Checkpoint loaded successfully!")
        
    # JIT compile the inference step
    @jax.jit
    def infer_step(x):
        # x is (1, 128, 128, 1)
        key = jax.random.key(0)
        # We use deterministic=True for evaluation to use argmax hard routing
        out, decisions = model(x, key, deterministic=True)
        return out, decisions

    expert_names = ["upsample", "deblur", "gaussian", "speckle"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_decisions = []
    
    for i, (n_file, gt_file) in enumerate(tqdm(zip(noisy_files, gt_files), total=num_samples)):
        # Load input (128, 128, 1)
        x = np.load(n_file)
        x = np.expand_dims(x, axis=0)  # (1, 128, 128, 1)
        
        # Load GT (512, 512, 1) or (256, 256, 1) depending on dataset
        gt = np.load(gt_file)
        
        # Run inference
        out, decisions = infer_step(x)
        
        # decisions shape: (1, 4) - batch_size=1, num_steps=4
        dec_seq = decisions[0] # Array of 4 indices
        all_decisions.append(dec_seq)
        
        route_str = " -> ".join([expert_names[int(idx)] for idx in dec_seq])
        
        # Save images for visualization
        out_img = np.array(out[0, ..., 0])  # (256, 256)
        x_img = np.array(x[0, ..., 0])      # (128, 128)
        
        # Scale back to 0-255
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        x_img = np.clip(x_img * 255.0, 0, 255).astype(np.uint8)
        
        # Save
        Image.fromarray(x_img).save(out_dir / f"{i:03d}_input.png")
        Image.fromarray(out_img).save(out_dir / f"{i:03d}_output.png")
        
        # Also save the routing path to a text file
        with open(out_dir / f"{i:03d}_route.txt", "w") as f:
            f.write(route_str)
            
    # Print overall stats
    all_dec = np.concatenate(all_decisions, axis=0)
    counts = np.bincount(all_dec.flatten(), minlength=4)
    print("\nOverall Expert Utilization:")
    for i in range(4):
        print(f"  {expert_names[i]}: {counts[i]} times")
        
    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
