from pathlib import Path
import argparse
import csv

import numpy as np


SEED = 42
LR_SIZE = 128
EPS = 1e-8
NOISE_BINS = 24
MIN_PIXELS_PER_BIN = 64
DFC_BANDS = 32
BLUR_SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


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


def compute_dfc(clean, degraded, num_bands=DFC_BANDS):
    residual = degraded - clean
    residual_fft = np.fft.fftshift(np.fft.fft2(residual))
    degraded_fft = np.fft.fftshift(np.fft.fft2(degraded))

    residual_power = np.abs(residual_fft) ** 2
    degraded_power = np.abs(degraded_fft) ** 2

    h, w = clean.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0.0, float(radius.max()) + EPS, num_bands + 1)

    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (radius >= lo) & (radius < hi)
        numerator = residual_power[mask].mean()
        denominator = degraded_power[mask].mean() + EPS
        curve.append(numerator / denominator)

    return np.asarray(curve, dtype=np.float32)


def fit_line(x, y):
    if len(x) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
    return float(slope), float(intercept)


def estimate_pair_noise(clean_lr, noisy_lr):
    clean = clean_lr.astype(np.float64, copy=False)
    noisy = noisy_lr.astype(np.float64, copy=False)
    residual = noisy - clean

    clean_flat = clean.ravel()
    residual_flat = residual.ravel()
    bins = np.linspace(float(clean_flat.min()), float(clean_flat.max()) + EPS, NOISE_BINS + 1)

    mean_x = []
    mean_y = []
    var_x = []
    var_y = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (clean_flat >= lo) & (clean_flat < hi)
        if int(mask.sum()) < MIN_PIXELS_PER_BIN:
            continue

        c = clean_flat[mask].mean()
        r = residual_flat[mask]
        mean_x.append(c)
        mean_y.append(r.mean())
        var_x.append(c * c)
        var_y.append(r.var())

    speckle_mean, gaussian_mean = fit_line(mean_x, mean_y)
    speckle_var, gaussian_var = fit_line(var_x, var_y)

    return {
        "gaussian_mean": gaussian_mean,
        "gaussian_std": max(gaussian_var, 0.0) ** 0.5,
        "speckle_mean": speckle_mean,
        "speckle_std": max(speckle_var, 0.0) ** 0.5,
    }


def estimate_pair_blur(gt, noisy_lr):
    base_lr = downsample_mean(gt, noisy_lr.shape[0], noisy_lr.shape[1]).astype(np.float32)
    real_dfc = compute_dfc(base_lr, noisy_lr)

    best_sigma = 0.0
    best_error = float("inf")
    best_clean_lr = base_lr

    for sigma in BLUR_SIGMAS:
        blurred_gt = gaussian_blur(gt, sigma) if sigma > 0 else gt
        candidate_lr = downsample_mean(blurred_gt, noisy_lr.shape[0], noisy_lr.shape[1]).astype(np.float32)
        candidate_dfc = compute_dfc(base_lr, candidate_lr)
        error = float(np.mean((real_dfc - candidate_dfc) ** 2))

        if error < best_error:
            best_sigma = float(sigma)
            best_error = error
            best_clean_lr = candidate_lr

    return best_sigma, best_error, best_clean_lr


def estimate_degradation_stats(gt_dir, noisy_dir):
    gt_files = sorted(Path(gt_dir).glob("*.npy"))
    noisy_dir = Path(noisy_dir)

    rows = []
    for index, gt_path in enumerate(gt_files, 1):
        gt = np.load(gt_path).astype(np.float32, copy=False)
        noisy = np.load(noisy_dir / gt_path.name).astype(np.float32, copy=False)

        blur_sigma, blur_error, clean_lr = estimate_pair_blur(gt, noisy)
        noise = estimate_pair_noise(clean_lr, noisy)
        rows.append(
            {
                "filename": gt_path.name,
                "blur_sigma": blur_sigma,
                "blur_dfc_mse": blur_error,
                **noise,
            }
        )

        if index % 400 == 0:
            print(f"estimated {index}/{len(gt_files)}")

    print_summary(rows)
    return rows


def print_summary(rows):
    blur = np.array([r["blur_sigma"] for r in rows])
    blur_error = np.array([r["blur_dfc_mse"] for r in rows])
    gaussian_std = np.array([r["gaussian_std"] for r in rows])
    speckle_std = np.array([r["speckle_std"] for r in rows])

    print("Estimated degradation from real pairs")
    print(f"pairs: {len(rows)}")
    print(f"blur sigma mean/std: {blur.mean():.6g} / {blur.std():.6g}")
    print(f"blur sigma p05/p50/p95: {np.percentile(blur, 5):.6g} / {np.percentile(blur, 50):.6g} / {np.percentile(blur, 95):.6g}")
    print(f"blur DFC-MSE mean/std: {blur_error.mean():.6g} / {blur_error.std():.6g}")
    print(f"gaussian std mean/std: {gaussian_std.mean():.6g} / {gaussian_std.std():.6g}")
    print(f"speckle std mean/std: {speckle_std.mean():.6g} / {speckle_std.std():.6g}")


def sample_degradation(rows, rng):
    row = rows[int(rng.integers(0, len(rows)))]
    return {
        "blur_sigma": float(row["blur_sigma"]),
        "gaussian_mean": float(row["gaussian_mean"]),
        "gaussian_std": float(row["gaussian_std"]),
        "speckle_mean": float(row["speckle_mean"]),
        "speckle_std": float(row["speckle_std"]),
    }


def generate_noisy_lr(clean_lr, params, rng):
    speckle = rng.normal(params["speckle_mean"], params["speckle_std"], size=clean_lr.shape)
    gaussian = rng.normal(params["gaussian_mean"], params["gaussian_std"], size=clean_lr.shape)
    return (clean_lr * (1.0 + speckle) + gaussian).astype(np.float32)


def generate_dataset(gt_dir, noisy_dir, output_dir, variants_per_gt, seed, blur_only, noise_only):
    rng = np.random.default_rng(seed)
    stats = estimate_degradation_stats(gt_dir, noisy_dir)
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
            params = sample_degradation(stats, rng)
            blur_sigma = 0.0 if noise_only else params["blur_sigma"]
            degraded_gt = gaussian_blur(gt, blur_sigma) if blur_sigma > 0 else gt
            clean_lr = downsample_mean(degraded_gt).astype(np.float32)

            if blur_only:
                noisy_lr = clean_lr
                gaussian_std = 0.0
                speckle_std = 0.0
            else:
                noisy_lr = generate_noisy_lr(clean_lr, params, rng)
                gaussian_std = params["gaussian_std"]
                speckle_std = params["speckle_std"]

            name = f"{sample_index:06d}.npy"
            np.save(out_gt_dir / name, gt)
            np.save(out_noisy_dir / name, noisy_lr)

            manifest.append(
                {
                    "synthetic_file": name,
                    "source_gt": gt_path.name,
                    "variant": variant,
                    "blur_sigma": blur_sigma,
                    "gaussian_std": gaussian_std,
                    "speckle_std": speckle_std,
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
    parser.add_argument("--noisy-dir", default="data/train/NoisyLR")
    parser.add_argument("--output-dir", default="data/generated_degradation")
    parser.add_argument("--variants-per-gt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--blur-only", action="store_true")
    parser.add_argument("--noise-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        gt_dir=args.gt_dir,
        noisy_dir=args.noisy_dir,
        output_dir=args.output_dir,
        variants_per_gt=args.variants_per_gt,
        seed=args.seed,
        blur_only=args.blur_only,
        noise_only=args.noise_only,
    )

'''
FOR NORMAL RECONSTRUCTION:
python noise_reconstruction_generator.py --output-dir data/generated_degradation

FOR BLUR ONLY:
python noise_reconstruction_generator.py --output-dir data/generated_degradation --blur-only
'''
