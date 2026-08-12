"""
Train the MoE Router end-to-end.

Workflow:
  1. Load all 4 pre-trained expert checkpoints
  2. Freeze expert weights (no gradient updates)
  3. Train ONLY the router network on the original combined dataset
  4. Loss = mixed_loss(final_output, GT)

Usage:
    conda run -n ml_env python train_moe.py \
        --upsample-ckpt ckpt/upsample \
        --deblur-ckpt ckpt/deblur \
        --gaussian-ckpt ckpt/gaussian \
        --speckle-ckpt ckpt/speckle \
        --epochs 20
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
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
from moe_model import IterativeMoE


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
def train_step_moe(model, optimizer, x_norm, gt, key, tau):
    def loss_fn(model):
        pred, _decisions = model(x_norm, key, tau=tau)
        return mixed_loss(pred, gt)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step_moe(model, x_norm, gt, key):
    pred, decisions = model(x_norm, key, tau=0.1, deterministic=True)
    loss = mixed_loss(pred, gt)
    return loss, decisions


# ─── Checkpoint helpers ──────────────────────────────────────────────────

def load_expert_checkpoint(ckpt_dir, expert_module):
    """Load pre-trained expert weights from checkpoint."""
    ckpt_dir = Path(ckpt_dir).resolve()
    manager = ocp.CheckpointManager(str(ckpt_dir))
    latest = manager.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    _, abstract_state = nnx.split(nnx.eval_shape(lambda: expert_module))
    restored = manager.restore(
        latest,
        args=ocp.args.Composite(
            model_state=ocp.args.StandardRestore(abstract_state),
        ),
    )
    nnx.update(expert_module, restored.model_state)
    print(f"  Loaded expert from {ckpt_dir} (step {latest})")


def build_checkpoint_manager(ckpt_dir, max_to_keep=3):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,
        create=True,
        enable_async_checkpointing=False,
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


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MoE Router end-to-end")
    parser.add_argument("--upsample-ckpt", default="ckpt/upsample")
    parser.add_argument("--deblur-ckpt", default="ckpt/deblur")
    parser.add_argument("--gaussian-ckpt", default="ckpt/gaussian")
    parser.add_argument("--speckle-ckpt", default="ckpt/speckle")
    parser.add_argument("--data-dir", default="../train", help="Original combined dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)  # Smaller batch — 4 experts run simultaneously
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau-start", type=float, default=2.0, help="Initial Gumbel temperature")
    parser.add_argument("--tau-end", type=float, default=0.1, help="Final Gumbel temperature")
    parser.add_argument("--expert-lr", type=float, default=1e-5, help="Very low LR for expert fine-tuning")
    parser.add_argument("--freeze-experts", action="store_true", default=False,
                        help="Completely freeze expert weights (train router only)")
    parser.add_argument("--ckpt-dir", default="ckpt/moe_router")
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Training MoE Router")
    print(f"  Experts: {args.upsample_ckpt}, {args.deblur_ckpt},")
    print(f"           {args.gaussian_ckpt}, {args.speckle_ckpt}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Gumbel τ: {args.tau_start} → {args.tau_end}")
    print(f"  Freeze experts: {args.freeze_experts}")
    print(f"{'='*60}\n")

    # ── Build model ──
    rngs = nnx.Rngs(args.seed)
    model = IterativeMoE(rngs=rngs)

    # Load pre-trained expert weights
    print("Loading pre-trained experts...")
    load_expert_checkpoint(args.upsample_ckpt, model.expert_upsample)
    load_expert_checkpoint(args.deblur_ckpt, model.expert_deblur)
    load_expert_checkpoint(args.gaussian_ckpt, model.expert_gaussian)
    load_expert_checkpoint(args.speckle_ckpt, model.expert_speckle)
    print("All experts loaded!\n")

    # ── Data ──
    data_dir = Path(args.data_dir)
    noisy_dir = data_dir / "NoisyLR"
    gt_dir = data_dir / "GT"

    all_noisy = sorted(noisy_dir.glob("*.npy"))
    all_gt = sorted(gt_dir.glob("*.npy"))
    assert len(all_noisy) == len(all_gt)

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
    total_steps = train_steps * args.epochs

    print(f"Train: {len(train_src)} → {train_steps} steps/epoch")
    print(f"Val:   {len(val_src)} → {val_steps} steps/epoch")

    # ── Optimizer ──
    # Train router at full LR, experts at very low LR (or frozen)
    if args.freeze_experts:
        # Only train router parameters
        optimizer = nnx.Optimizer(
            model.router,
            optax.chain(
                optax.clip_by_global_norm(args.grad_clip),
                optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay),
            ),
            wrt=nnx.Param,
        )
    else:
        # Separate optimizers: router at full LR, experts at expert_lr
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(args.grad_clip),
                optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay),
            ),
        )

    ckpt_manager = build_checkpoint_manager(args.ckpt_dir, max_to_keep=3)
    normalizer_fn = asinh_normalize

    print(f"[hardware] backend={jax.default_backend()} devices={jax.devices()}")

    # ── Training ──
    key = jax.random.key(args.seed)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Anneal Gumbel temperature
        progress = epoch / max(args.epochs - 1, 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * progress

        # Train
        epoch_train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for step, raw_batch in enumerate(pbar):
            key, subkey = jax.random.split(key)
            noisy = raw_batch["noisy_lr"].astype(jnp.float32)
            gt = raw_batch["gt"].astype(jnp.float32)
            x_norm = normalizer_fn(noisy, axis=(1, 2))

            loss = train_step_moe(model, optimizer, x_norm, gt, subkey, tau)
            epoch_train_losses.append(float(loss))
            global_step += 1

            if step % 20 == 0:
                pbar.set_postfix(loss=f"{float(loss):.5f}", tau=f"{tau:.2f}")

        # Validate
        epoch_val_losses = []
        all_decisions = []
        for raw_batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
            key, subkey = jax.random.split(key)
            noisy = raw_batch["noisy_lr"].astype(jnp.float32)
            gt = raw_batch["gt"].astype(jnp.float32)
            x_norm = normalizer_fn(noisy, axis=(1, 2))

            vloss, decisions = val_step_moe(model, x_norm, gt, subkey)
            epoch_val_losses.append(float(vloss))
            all_decisions.append(np.array(decisions))

        avg_train = np.mean(epoch_train_losses)
        avg_val = np.mean(epoch_val_losses)
        dt = time.time() - t0

        # Analyze routing decisions
        all_dec = np.concatenate(all_decisions, axis=0)
        expert_names = ["upsample", "deblur", "gaussian", "speckle"]
        counts = np.bincount(all_dec.flatten(), minlength=4)
        routing_str = " | ".join(f"{expert_names[i]}:{counts[i]}" for i in range(4))

        print(f"[epoch {epoch:03d}] train={avg_train:.5f}  val={avg_val:.5f}  "
              f"τ={tau:.2f}  time={dt:.1f}s")
        print(f"  routing: {routing_str}")

        # Checkpoint
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(ckpt_manager, global_step, model, optimizer, epoch, avg_train)
            print(f"  [checkpoint] saved at step {global_step}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val

    print(f"\n{'='*60}")
    print(f"  MoE Training complete! Best val_loss: {best_val_loss:.5f}")
    print(f"  Checkpoints saved to: {args.ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
