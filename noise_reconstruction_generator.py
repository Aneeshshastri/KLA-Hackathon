from pathlib import Path
import argparse
import csv

import numpy as np


SEED = 42
LR_SIZE = 128


def downsample_mean(image, out_h=LR_SIZE, out_w=LR_SIZE):
    h, w = image.shape
    scale_h = h // out_h
    scale_w = w // out_w
    image = image[: out_h * scale_h, : out_w * scale_w]
    return image.reshape(out_h, scale_h, out_w, scale_w).mean(axis=(1, 3))


def gaussian_kernel1d(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)

    radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def convolve_axis_reflect(image, kernel, axis):
    pad = len(kernel) // 2
    if pad == 0:
        return image.astype(np.float32, copy=False)

    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(image, pad_width, mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)

    for i, weight in enumerate(kernel):
        slc = [slice(None)] * image.ndim
        slc[axis] = slice(i, i + image.shape[axis])
        out += weight * padded[tuple(slc)]

    return out


def gaussian_blur(image, sigma):
    kernel = gaussian_kernel1d(sigma)
    blurred = convolve_axis_reflect(image, kernel, axis=0)
    return convolve_axis_reflect(blurred, kernel, axis=1)


def generate_noisy_lr(
    clean_lr,
    rng,
    gaussian_mean,
    gaussian_std,
    speckle_mean,
    speckle_std,
):
    speckle = rng.normal(speckle_mean, speckle_std, size=clean_lr.shape)
    gaussian = rng.normal(gaussian_mean, gaussian_std, size=clean_lr.shape)
    noisy_lr = clean_lr * (1.0 + speckle) + gaussian
    return noisy_lr.astype(np.float32)


def sample_uniform(rng, lo, hi):
    if lo == hi:
        return float(lo)
    return float(rng.uniform(lo, hi))


def generate_dataset(
    gt_dir,
    output_dir,
    variants_per_gt,
    seed,
    use_blur,
    blur_only,
    blur_sigma_min,
    blur_sigma_max,
    gaussian_mean,
    gaussian_std,
    speckle_mean,
    speckle_std,
):
    rng = np.random.default_rng(seed)
    gt_files = sorted(Path(gt_dir).glob("*.npy"))

    output_dir = Path(output_dir)
    out_gt_dir = output_dir / "GT"
    out_noisy_dir = output_dir / "NoisyLR"
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_noisy_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    sample_index = 0

    for gt_path in gt_files:
        gt = np.load(gt_path).astype(np.float32, copy=False)

        for variant in range(variants_per_gt):
            blur_sigma = sample_uniform(rng, blur_sigma_min, blur_sigma_max) if use_blur else 0.0
            degraded_gt = gaussian_blur(gt, blur_sigma) if blur_sigma > 0 else gt
            clean_lr = downsample_mean(degraded_gt).astype(np.float32)

            if blur_only:
                noisy_lr = clean_lr.astype(np.float32, copy=False)
            else:
                noisy_lr = generate_noisy_lr(
                    clean_lr,
                    rng,
                    gaussian_mean,
                    gaussian_std,
                    speckle_mean,
                    speckle_std,
                )

            name = f"{sample_index:06d}.npy"
            np.save(out_gt_dir / name, gt)
            np.save(out_noisy_dir / name, noisy_lr)

            manifest.append(
                {
                    "synthetic_file": name,
                    "source_gt": gt_path.name,
                    "variant": variant,
                    "blur_sigma": blur_sigma,
                    "gaussian_std": 0.0 if blur_only else gaussian_std,
                    "speckle_std": 0.0 if blur_only else speckle_std,
                }
            )
            sample_index += 1

    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "synthetic_file",
                "source_gt",
                "variant",
                "blur_sigma",
                "gaussian_std",
                "speckle_std",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Wrote {sample_index} generated pairs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default="data/train/GT")
    parser.add_argument("--output-dir", default="data/generated_degradation")
    parser.add_argument("--variants-per-gt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--use-blur", action="store_true")
    parser.add_argument("--blur-only", action="store_true")
    parser.add_argument("--blur-sigma-min", type=float, default=0.4)
    parser.add_argument("--blur-sigma-max", type=float, default=2.0)

    parser.add_argument("--gaussian-mean", type=float, default=0.0)
    parser.add_argument("--gaussian-std", type=float, default=0.026)
    parser.add_argument("--speckle-mean", type=float, default=0.0)
    parser.add_argument("--speckle-std", type=float, default=0.165)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        variants_per_gt=args.variants_per_gt,
        seed=args.seed,
        use_blur=args.use_blur,
        blur_only=args.blur_only,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
        gaussian_mean=args.gaussian_mean,
        gaussian_std=args.gaussian_std,
        speckle_mean=args.speckle_mean,
        speckle_std=args.speckle_std,
    )
'''
Blur Only
    python noise_reconstruction_generator.py --output-dir data/generated_blur_only --use-blur --blur-only --blur-sigma-min 0.4 --blur-sigma-max 2.0

Noise Only
    python noise_reconstruction_generator.py --output-dir data/generated_noise_only --gaussian-std 0.026 --speckle-std 0.165

Blur + Noise
    python noise_reconstruction_generator.py --output-dir data/generated_blur_noise --use-blur --blur-sigma-min 0.4 --blur-sigma-max 2.0 --gaussian-std 0.026 --speckle-std 0.165

More OOD / Stronger Blur + Noise
    python noise_reconstruction_generator.py --output-dir data/generated_ood --use-blur --blur-sigma-min 1.5 --blur-sigma-max 3.0 --gaussian-std 0.06 --speckle-std 0.22

Multiple variants per GT
    python noise_reconstruction_generator.py --output-dir data/generated_ood_x3 --variants-per-gt 3 --use-blur --blur-sigma-min 1.5 --blur-sigma-max 3.0 --gaussian-std 0.06 --speckle-std 0.22  
'''
