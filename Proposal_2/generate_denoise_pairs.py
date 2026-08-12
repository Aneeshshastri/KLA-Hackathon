"""
Generate isolated MoE expert datasets using the random permutation strategy.

For each expert (B, D, G, S):
  25%: Expert noise only -> Target is clean GT.
  30%: Expert noise + 2 random other noises -> Target is GT + 2 other noises.
  45%: All 4 noises -> Target is GT + 3 other noises.

In all cases, the expert's specific noise is applied LAST to create the Input.
The Target is the image state exactly BEFORE the expert's noise is applied.
Order of "other" noises is fully random.
"""

from pathlib import Path
import argparse
import csv
import sys
import random

import numpy as np

# Reuse helpers from the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noise_reconstruction_generator import (
    downsample_mean,
    gaussian_blur,
    sample_uniform,
)

# ─── Degradation Ops ────────────────────────────────────────────────────────

def op_blur(img, rng):
    sigma = sample_uniform(rng, 0.4, 2.0)
    return gaussian_blur(img, sigma)

def op_downsample(img, rng):
    if img.shape[0] == 256:
        return downsample_mean(img, 128, 128).astype(np.float32)
    return img

def op_gaussian(img, rng):
    return img + rng.normal(0.0, 0.026, size=img.shape).astype(np.float32)

def op_speckle(img, rng):
    return img * (1.0 + rng.normal(0.0, 0.165, size=img.shape).astype(np.float32))

OPS = {
    "blur": op_blur,
    "upsample": op_downsample, # the noise that upsample removes is downsampling
    "gaussian": op_gaussian,
    "speckle": op_speckle,
}

# ─── Generation Logic ───────────────────────────────────────────────────────

def generate_expert_dataset(expert_type, gt_files, output_dir, rng, num_pairs=4000):
    out_noisy = output_dir / "NoisyLR"
    out_gt = output_dir / "GT"
    out_noisy.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    manifest = []

    all_noises = list(OPS.keys())
    other_noises = [n for n in all_noises if n != expert_type]

    for i in range(num_pairs):
        gt_path = rng.choice(gt_files)
        img = np.load(gt_path).astype(np.float32)
        
        # Determine subset of other noises based on probability
        p = rng.random()
        if p < 0.25:
            # 25%: Expert noise only
            chosen_others = []
        elif p < 0.55:
            # 30%: Expert noise + 2 other noises
            chosen_others = rng.choice(other_noises, size=2, replace=False).tolist()
        else:
            # 45%: All noises
            chosen_others = other_noises.copy()
            
        # Shuffle the order of other noises
        rng.shuffle(chosen_others)
        
        # 1. Create Target (apply other noises)
        target = img
        for noise in chosen_others:
            target = OPS[noise](target, rng)
            
        # 2. Create Input (apply expert noise LAST)
        input_img = OPS[expert_type](target, rng)
        
        name = f"{i:06d}.npy"
        np.save(out_noisy / name, input_img)
        np.save(out_gt / name, target)
        
        manifest.append({
            "file": name, 
            "source": gt_path.name, 
            "expert": expert_type,
            "other_noises_applied": "-".join(chosen_others) if chosen_others else "none"
        })

    _write_manifest(output_dir, manifest)
    print(f"[{expert_type}] Wrote {num_pairs} pairs to {output_dir}")

def _write_manifest(output_dir, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with (output_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Generate advanced MoE datasets")
    parser.add_argument("--gt-dir", default="../train/GT")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pairs", type=int, default=4000, help="Pairs per expert (~3GB total)")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.npy"))
    rng = np.random.default_rng(args.seed)

    output_root = Path(args.output_root)
    
    for ext in ["blur", "upsample", "gaussian", "speckle"]:
        generate_expert_dataset(ext, gt_files, output_root / f"{ext}_only", rng, num_pairs=args.num_pairs)
        
if __name__ == "__main__":
    main()
