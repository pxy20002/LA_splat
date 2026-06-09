"""AnchorSplat training loop — mini training mode for initial validation.

Usage:
    # Mini training (fast verification)
    python train/train.py --mini

    # Full training
    python train/train.py --pt_paths datasets/scene1.pt datasets/scene2.pt --steps 5000
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.dataset import SceneDataset
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.renderer import assemble_gaussians, render_view, compute_loss, DEFAULT_LOSS_WEIGHTS


# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_IMAGES = "datasets/images/playroom"


def parse_args():
    p = argparse.ArgumentParser(description="AnchorSplat training.")
    inp = p.add_mutually_exclusive_group(required=False)
    inp.add_argument("--image_dir", type=str, default=None,
                     help="Folder of input images (runs MapAnything on-the-fly).")
    inp.add_argument("--pt_path", type=str, default=None,
                     help="Cached .pt prediction file (skips MapAnything).")
    p.add_argument("--steps", type=int, default=5000,
                   help="Total training steps (paper: 5000).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_anchors", type=int, default=256,
                   help="Anchors per scene.")
    p.add_argument("--out_dir", type=str, default="/DATA/disk0/pxy/checkpoints",
                   help="Output root directory for checkpoints.")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save checkpoint every N steps.")
    p.add_argument("--log_every", type=int, default=10,
                   help="Print loss every N steps.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mini", action="store_true",
                   help="Mini training: 100 steps on default scene.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to resume from.")
    p.add_argument("--grad_clip", type=float, default=100.0,
                   help="Gradient clipping max_norm (default 1.0, try 10.0 if always clipping).")
    return p.parse_args()


def _save_dashboard(csv_path, ckpt_dir, name, steps_total, scene_label,
                    l3_scale=1000, l4_scale=10):
    """Plot 4-panel training dashboard from metrics CSV."""
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
    except (OSError, ValueError):
        return
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2:
        return

    steps = data[:, 0].astype(int)
    L = data[:, 1].astype(float)
    L1 = data[:, 2].astype(float)
    L2 = data[:, 3].astype(float)
    L3 = data[:, 4].astype(float)
    L4 = data[:, 5].astype(float)
    ssim_q = data[:, 6].astype(float)
    lpips_v = data[:, 7].astype(float)
    opacity = data[:, 8].astype(float)
    sx = data[:, 9].astype(float)
    sy = data[:, 10].astype(float)
    sz = data[:, 11].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Total Loss
    ax = axes[0, 0]
    ax.plot(steps, L, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Step"); ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss"); ax.grid(True, alpha=0.3)

    # Top-right: Loss components
    ax = axes[0, 1]
    ax.plot(steps, L1, linewidth=0.8, label="L1 (rendering)")
    ax.plot(steps, L2, linewidth=0.8, label="L2 (depth)")
    ax.plot(steps, L3 * l3_scale, linewidth=0.8, label=f"L3 (opacity) ×{l3_scale}")
    ax.plot(steps, L4 * l4_scale, linewidth=0.8, label=f"L4 (scale) ×{l4_scale}")
    ax.set_xlabel("Step")
    ax.set_title("Loss Components (scaled)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-left: SSIM + LPIPS
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(steps, ssim_q, linewidth=0.8, color="tab:green", label="SSIM↑")
    ax2.plot(steps, lpips_v, linewidth=0.8, color="tab:orange", label="LPIPS↓")
    ax.set_xlabel("Step"); ax.set_ylabel("SSIM ↑", color="tab:green")
    ax2.set_ylabel("LPIPS ↓", color="tab:orange")
    ax.set_title("SSIM ↑ & LPIPS ↓")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right: α + scale per axis
    ax = axes[1, 1]
    ax.plot(steps, opacity, linewidth=0.8, color="tab:purple", label="α")
    ax.plot(steps, sx, linewidth=0.8, alpha=0.6, label="sX")
    ax.plot(steps, sy, linewidth=0.8, alpha=0.6, label="sY")
    ax.plot(steps, sz, linewidth=0.8, alpha=0.6, label="sZ")
    ax.set_xlabel("Step"); ax.set_title("Opacity & Scale per Axis")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f"AnchorSplat — {steps_total} steps  |  {scene_label}", fontsize=12)
    fig.tight_layout()
    dash_path = ckpt_dir / f"{name}.png"
    fig.savefig(dash_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Reproducibility ───────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed: {args.seed}")

    # ── Mini mode overrides ────────────────────────────────────────────
    if args.mini:
        args.steps = 100
        args.save_every = 20
        args.log_every = 5
        print("=== MINI TRAINING MODE === (100 steps)")

    # ── Timestamped checkpoint dir ─────────────────────────────────────
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    ckpt_dir = Path(args.out_dir) / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ── Dataset ─────────────────────────────────────────────────────────
    if args.image_dir is None and args.pt_path is None:
        args.image_dir = DEFAULT_IMAGES
        print(f"Using default: --image_dir {DEFAULT_IMAGES}")

    dataset = SceneDataset(
        image_dir=args.image_dir,
        pt_path=args.pt_path,
        num_anchors=args.num_anchors,
        device=device,
    )
    batch = dataset.get_batch()
    H, W = batch["H"], batch["W"]
    V = dataset.n_views
    print(f"\nScene: {args.image_dir or args.pt_path}")
    print(f"  Views: {V} (all used every step)  |  "
          f"Resolution: {H}x{W}  |  Anchors: {args.num_anchors}")

    # ── Models ──────────────────────────────────────────────────────────
    unet = LightweightUNet(in_channels=10, out_channels=64).train().to(device)
    decoder = GaussianDecoder(in_dim=64).train().to(device)

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )

    start_step = 0
    all_losses = []

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        all_losses = ckpt.get("losses", [])
        print(f"  Resumed at step {start_step}")

    unet_params = sum(p.numel() for p in unet.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTrainable params: U-Net={unet_params:,} + Decoder={dec_params:,} "
          f"= {unet_params + dec_params:,}")
    w = DEFAULT_LOSS_WEIGHTS
    print(f"Loss weights: λI={w['lambda_I']:.0f}  |  "
          f"γSSIM={w['gamma_ssim']}  γLPIPS={w['gamma_lpips']}  |  "
          f"λD={w['lambda_D']:.0f}  |  λα={w['lambda_alpha']}  |  λs={w['lambda_s']:.0f}")

    # ── CSV logger ─────────────────────────────────────────────────────
    csv_path = ckpt_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "L", "L1", "L2", "L3", "L4",
        "SSIM_q", "LPIPS", "opacity_mean", "sX", "sY", "sZ", "eq_r",
        "grad_raw", "grad_clipped", "time_s",
    ])

    # ── Save initial weights ──────────────────────────────────────────
    torch.save({"unet": unet.state_dict(), "decoder": decoder.state_dict()},
               ckpt_dir / "init_weights.pt")

    # ── Training loop ───────────────────────────────────────────────────
    losses = all_losses.copy()
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {args.steps} steps, {V} views/step (all)")
    print(f"{'='*60}")

    for step in range(start_step, args.steps):
        batch = dataset.get_batch(device=device)
        u_input = batch["u_input"]        # [V, 10, H, W]
        anchors = batch["anchors"]         # [N, 3]

        # ── Forward ──────────────────────────────────────────────────
        features = unet(u_input)           # [V, 64, H, W]
        anchor_feats, vis = project_and_aggregate(
            anchors, features, batch["depths"], batch["poses"], batch["Ks"],
        )                                  # [N, 64]
        gs_out = decoder(anchor_feats, anchors)  # Eq.8: {F_j, A_j}
        gaussians = assemble_gaussians(gs_out, anchors)

        # ── Render + Loss per view ───────────────────────────────────
        total_loss = 0.0

        nan_views = 0
        # Per-view accumulators (raw metrics before weighting)
        ssim_q_sum = 0.0   # real SSIM (higher=better)
        lpips_sum = 0.0
        l1_raw_sum = 0.0
        depth_raw_sum = 0.0

        for i in range(V):
            rendered, rendered_depth, _ = render_view(
                gaussians, batch["poses"][i], batch["Ks"][i], H, W,
                mvs_depth=batch["depths"][i],
            )
            gt_img = batch["gt_imgs"][i]
            gt_depth = batch["depths"][i].squeeze(0).unsqueeze(-1)

            ld = compute_loss(
                rendered, rendered_depth, gt_img, gt_depth,
                gaussians["opacities"], gaussians["scales"],
            )
            if torch.isfinite(ld["total"]):
                total_loss = total_loss + ld["total"]
                ssim_q_sum += (1.0 - ld["ssim"])  # ssim_loss→quality
                lpips_sum += ld["lpips"]
                l1_raw_sum += ld["l1"]
                depth_raw_sum += ld["depth"]
            else:
                nan_views += 1
                print(f"  ⚠ Step {step}: view {i} NaN, skipped.")

        if nan_views >= V:
            # All views NaN — skip this step entirely
            total_loss = torch.tensor(float("nan"), device=device)

        valid_views = V - nan_views
        vv = max(valid_views, 1)
        total_loss = total_loss / vv
        loss_val = total_loss.item()

        # Weighted loss components
        ssim_avg = ssim_q_sum / vv   # real SSIM quality (0-1, higher=better)
        lpips_avg = lpips_sum / vv   # raw LPIPS (lower=better)
        L1_w = (l1_raw_sum / vv + 0.2 * (1.0 - ssim_avg) + 0.2 * lpips_avg) * 200.0
        L2_w = (depth_raw_sum / vv) * 100.0
        L3_w = (1.0 - gaussians["opacities"].mean()).item() * 0.1
        L4_w = gaussians["scales"].prod(dim=1).mean().item() * 10000.0

        # ── NaN guard: skip this step if loss exploded ──────────────
        if not torch.isfinite(total_loss):
            print(f"\n  ⚠ Step {step}: NaN loss detected, skipping update")
            losses.append(float("nan"))
            csv_writer.writerow([
                step, "nan", "nan", "nan", "nan", "nan", "nan", "nan",
                "nan", "nan", "nan", "nan", "nan", "0",
            ])
            # Reload last good checkpoint if available
            last_ckpt = sorted(ckpt_dir.glob("ckpt_*.pt"))
            if last_ckpt:
                ckpt = torch.load(str(last_ckpt[-1]), map_location=device, weights_only=False)
                unet.load_state_dict(ckpt["unet"])
                decoder.load_state_dict(ckpt["decoder"])
                optimizer.load_state_dict(ckpt["optimizer"])
                print(f"  ↻ Restored from {last_ckpt[-1].name}")
            continue

        # ── Backward ─────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()

        # Check for NaN gradients (CUDA kernel instability)
        has_nan_grad = False
        for p in optimizer.param_groups[0]["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nan_grad = True
                break
        if has_nan_grad:
            print(f"  ⚠ Step {step}: NaN gradient detected, reverting weights")
            last_ckpt = sorted(ckpt_dir.glob("ckpt_*.pt"))
            if last_ckpt:
                ckpt = torch.load(str(last_ckpt[-1]), map_location=device, weights_only=False)
                unet.load_state_dict(ckpt["unet"])
                decoder.load_state_dict(ckpt["decoder"])
                optimizer.load_state_dict(ckpt["optimizer"])
                print(f"  ↻ Restored from {last_ckpt[-1].name}")

        # Gradient norm BEFORE clipping (for monitoring)
        raw_norm = 0.0
        for p in optimizer.param_groups[0]["params"]:
            if p.grad is not None:
                raw_norm += p.grad.data.norm(2).item() ** 2
        raw_norm = raw_norm ** 0.5

        # Clip
        torch.nn.utils.clip_grad_norm_(
            list(unet.parameters()) + list(decoder.parameters()),
            max_norm=args.grad_clip,
        )

        # Gradient norm AFTER clipping
        grad_norm = 0.0
        for p in optimizer.param_groups[0]["params"]:
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        loss_val = total_loss.item()
        losses.append(loss_val)

        # ── Diagnostics ──────────────────────────────────────────────
        opacity_mean = gaussians["opacities"].mean().item()
        scales = gaussians["scales"]  # [4N, 3]
        sx, sy, sz = scales[:, 0].mean().item(), scales[:, 1].mean().item(), scales[:, 2].mean().item()
        mean_vol = scales.prod(dim=1).mean().item()
        eq_r = mean_vol ** (1 / 3) if mean_vol > 0 else 0.0  # equivalent radius

        # ── CSV log ──────────────────────────────────────────────────
        csv_writer.writerow([
            step, f"{loss_val:.4f}",
            f"{L1_w:.2f}", f"{L2_w:.2f}", f"{L3_w:.4f}", f"{L4_w:.2f}",
            f"{ssim_avg:.4f}", f"{lpips_avg:.4f}",
            f"{opacity_mean:.4f}", f"{sx:.4f}", f"{sy:.4f}", f"{sz:.4f}", f"{eq_r:.4f}",
            f"{raw_norm:.2f}", f"{grad_norm:.2f}", f"{time.time() - t_start:.1f}",
        ])

        # ── Console log ─────────────────────────────────────────────
        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"  {step:5d}/{args.steps:5d} | "
                  f"L={loss_val:7.2f} | "
                  f"{L1_w:7.2f} {L2_w:7.2f} {L3_w:7.4f} {L4_w:7.2f} | "
                  f"↑{ssim_avg:.4f} ↓{lpips_avg:.4f} | "
                  f"α={opacity_mean:.3f} s({sx:.3f},{sy:.3f},{sz:.3f}) r={eq_r:.3f} | "
                  f"{raw_norm:.1f}/{grad_norm:.1f} | "
                  f"{elapsed:.0f}s")

        # ── Save checkpoint + dashboard ──────────────────────────────
        if (step + 1) % args.save_every == 0:
            ckpt = {
                "step": step,
                "unet": unet.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "losses": losses,
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"ckpt_{step + 1:05d}.pt"
            torch.save(ckpt, ckpt_path)
            _save_dashboard(csv_path, ckpt_dir, f"training_dashboard_{step+1:05d}",
                            args.steps, args.image_dir or args.pt_path)
            print(f"  ── checkpoint saved → {ckpt_path}")

    # ── Final checkpoint ───────────────────────────────────────────────
    ckpt = {
        "step": args.steps - 1,
        "unet": unet.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "ckpt_final.pt")
    print(f"\nFinal checkpoint → {ckpt_dir / 'ckpt_final.pt'}")

    csv_file.close()
    _save_dashboard(csv_path, ckpt_dir, "training_dashboard_final",
                    args.steps, args.image_dir or args.pt_path)
    print(f"Dashboard → {ckpt_dir}/")


    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s  ({elapsed / args.steps:.2f}s/step)")
    print(f"Metrics CSV → {csv_path}")


if __name__ == "__main__":
    main()
