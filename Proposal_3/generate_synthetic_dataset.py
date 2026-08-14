"""
Generate synthetic dataset by applying all available noises in a random order.
"""

from pathlib import Path
import argparse
import csv
import sys

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
    "upsample": op_downsample, 
    "gaussian": op_gaussian,
    "speckle": op_speckle,
}

# ─── Generation Logic ───────────────────────────────────────────────────────

def generate_dataset(gt_files, output_dir, rng, num_pairs=16000):
    out_noisy = output_dir / "NoisyLR"
    out_gt = output_dir / "GT"
    out_noisy.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    manifest = []

    all_noises = list(OPS.keys())

    for i in range(num_pairs):
        gt_path = rng.choice(gt_files)
        img = np.load(gt_path).astype(np.float32)
        
        # Apply all noises in random order
        chosen_noises = all_noises.copy()
        rng.shuffle(chosen_noises)
        
        # The GT is the original image, and the Input is after all noises are applied.
        target = img
        input_img = img
        for noise in chosen_noises:
            input_img = OPS[noise](input_img, rng)
            
        name = f"{i:06d}.npy"
        np.save(out_noisy / name, input_img)
        np.save(out_gt / name, target)
        
        manifest.append({
            "file": name, 
            "source": gt_path.name, 
            "noises_applied": "-".join(chosen_noises)
        })

    _write_manifest(output_dir, manifest)
    print(f"Wrote {num_pairs} pairs to {output_dir}")

def _write_manifest(output_dir, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with (output_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--gt-dir", default="../train/GT")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pairs", type=int, default=16000, help="Total pairs (~4GB total)")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.npy"))
    rng = np.random.default_rng(args.seed)

    output_root = Path(args.output_root) / "synthetic"
    
    generate_dataset(gt_files, output_root, rng, num_pairs=args.num_pairs)
        
if __name__ == "__main__":
    main()
