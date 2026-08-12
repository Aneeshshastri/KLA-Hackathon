"""
Pre-train a single MoE expert on its isolated degradation dataset.

Usage:
    conda run -n ml_env python train_expert.py --expert-type upsample --data-dir data/upsample_only --epochs 40
    conda run -n ml_env python train_expert.py --expert-type deblur   --data-dir data/blur_only     --epochs 40
    conda run -n ml_env python train_expert.py --expert-type gaussian --data-dir data/gaussian_only --epochs 40
    conda run -n ml_env python train_expert.py --expert-type speckle  --data-dir data/speckle_only  --epochs 40
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from dataloader import create_src_dataloader
from train_utils import mixed_loss
from moe_experts import create_expert


# ─── Normalization ────────────────────────────────────────────────────────

EPS = 1e-6

def asinh_normalize(x, axis=None, eps=EPS):
    median = jnp.median(x, axis=axis, keepdims=True)
    q75 = jnp.quantile(x, 0.75, axis=axis, keepdims=True)
    q25 = jnp.quantile(x, 0.25, axis=axis, keepdims=True)
    iqr = q75 - q25
    return jnp.arcsinh((x - median) / (iqr + eps))


# ─── Train / Val Steps ───────────────────────────────────────────────────

@nnx.jit
def train_step(model, optimizer, x_norm, gt):
    def loss_fn(model):
        pred = model(x_norm)
        return mixed_loss(pred, gt)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step(model, x_norm, gt):
    pred = model(x_norm)
    return mixed_loss(pred, gt)


# ─── Checkpoint helpers ──────────────────────────────────────────────────

def build_checkpoint_manager(ckpt_dir, max_to_keep=3):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,
        create=True,
        enable_async_checkpointing=True,
    )
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(str(ckpt_dir), options=options)


def save_checkpoint(manager, step, model, optimizer, epoch, train_loss):
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    metadata = {"epoch": epoch, "step": step, "train_loss": float(train_loss)}
    manager.save(
        step,
        args=ocp.args.Composite(
            model_state=ocp.args.StandardSave(model_state),
            opt_state=ocp.args.StandardSave(opt_state),
            metadata=ocp.args.JsonSave(metadata),
        ),
    )


# ─── Main Training Loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-train a single MoE expert")
    parser.add_argument("--expert-type", required=True, choices=["upsample", "deblur", "gaussian", "speckle"])
    parser.add_argument("--data-dir", required=True, help="Path to isolated dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ckpt-dir", default=None, help="Checkpoint directory (default: ckpt/<expert_type>)")
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = f"ckpt/{args.expert_type}"

    print(f"\n{'='*60}")
    print(f"  Pre-training Expert: {args.expert_type}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Checkpoint: {args.ckpt_dir}")
    print(f"{'='*60}\n")

    # ── Data ──
    data_dir = Path(args.data_dir)
    noisy_dir = data_dir / "NoisyLR"
    gt_dir = data_dir / "GT"

    all_noisy = sorted(noisy_dir.glob("*.npy"))
    all_gt = sorted(gt_dir.glob("*.npy"))
    assert len(all_noisy) == len(all_gt), f"Mismatch: {len(all_noisy)} noisy vs {len(all_gt)} GT"
    print(f"Dataset: {len(all_noisy)} pairs")

    train_noisy, val_noisy, train_gt, val_gt = train_test_split(
        all_noisy, all_gt, test_size=args.val_split, random_state=args.seed,
    )

    train_src, train_loader = create_src_dataloader(
        train_noisy, train_gt,
        batch_size=args.batch_size, worker_count=1,
        seed=args.seed, shuffle=True, augment=True,
    )
    val_src, val_loader = create_src_dataloader(
        val_noisy, val_gt,
        batch_size=args.batch_size, worker_count=1,
        seed=args.seed + 1, shuffle=False, augment=False,
    )

    train_steps = len(train_src) // args.batch_size
    val_steps = len(val_src) // args.batch_size
    print(f"Train: {len(train_src)} → {train_steps} steps/epoch")
    print(f"Val:   {len(val_src)} → {val_steps} steps/epoch")

    # ── Model ──
    rngs = nnx.Rngs(args.seed)
    model = create_expert(args.expert_type, rngs)

    total_steps = train_steps * args.epochs
    warmup = min(2000, total_steps // 5)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup, decay_steps=total_steps - warmup, end_value=1e-6,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
        ),
    )

    ckpt_manager = build_checkpoint_manager(args.ckpt_dir, max_to_keep=3)
    normalizer_fn = asinh_normalize

    print(f"[hardware] backend={jax.default_backend()} devices={jax.devices()}")
    print(f"[schedule] warmup={warmup}  total_steps={total_steps}")

    # ── Training ──
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        epoch_train_losses = []
        pbar = tqdm(range(train_steps), desc=f"Epoch {epoch:03d} [train]", leave=False)
        for step in pbar:
            raw_batch = next(train_loader)
            noisy = raw_batch["noisy_lr"].astype(jnp.float32)
            gt = raw_batch["gt"].astype(jnp.float32)
            x_norm = normalizer_fn(noisy, axis=(1, 2))

            loss = train_step(model, optimizer, x_norm, gt)
            epoch_train_losses.append(float(loss))
            global_step += 1

            if step % 20 == 0:
                pbar.set_postfix(loss=f"{float(loss):.5f}")

        # Validate
        epoch_val_losses = []
        for _ in tqdm(range(val_steps), desc=f"Epoch {epoch:03d} [val]  ", leave=False):
            raw_batch = next(val_loader)
            noisy = raw_batch["noisy_lr"].astype(jnp.float32)
            gt = raw_batch["gt"].astype(jnp.float32)
            x_norm = normalizer_fn(noisy, axis=(1, 2))

            vloss = val_step(model, x_norm, gt)
            epoch_val_losses.append(float(vloss))

        avg_train = np.mean(epoch_train_losses)
        avg_val = np.mean(epoch_val_losses)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] train_loss={avg_train:.5f}  val_loss={avg_val:.5f}  time={dt:.1f}s")

        # Checkpoint
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(ckpt_manager, global_step, model, optimizer, epoch, avg_train)
            print(f"  [checkpoint] saved at step {global_step} (epoch {epoch})")

        if avg_val < best_val_loss:
            best_val_loss = avg_val

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val_loss: {best_val_loss:.5f}")
    print(f"  Checkpoints saved to: {args.ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
