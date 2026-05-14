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
from anchorsplat.renderer import assemble_gaussians, render_view, compute_loss


# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_PT = "datasets/mapanything_predictions_truck_v4.pt"


def parse_args():
    p = argparse.ArgumentParser(description="AnchorSplat training.")
    p.add_argument("--pt_paths", type=str, nargs="+", default=[DEFAULT_PT],
                   help="One or more .pt prediction files.")
    p.add_argument("--steps", type=int, default=5000,
                   help="Total training steps (paper: 5000 for Stage 1).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_anchors", type=int, default=512)
    p.add_argument("--num_views", type=int, default=2,
                   help="Views per training step (paper: 4).")
    p.add_argument("--out_dir", type=str, default="checkpoints",
                   help="Output root directory for checkpoints.")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save checkpoint every N steps.")
    p.add_argument("--log_every", type=int, default=10,
                   help="Print loss every N steps.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mini", action="store_true",
                   help="Mini training mode: 100 steps on truck_v4.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to resume from.")
    p.add_argument("--grad_clip", type=float, default=100.0,
                   help="Gradient clipping max_norm (default 1.0, try 10.0 if always clipping).")
    return p.parse_args()


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
        args.num_views = 2
        print("=== MINI TRAINING MODE === (100 steps)")

    # ── Timestamped checkpoint dir ─────────────────────────────────────
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    ckpt_dir = Path(args.out_dir) / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset = SceneDataset(
        [str(Path(d).resolve()) for d in args.pt_paths],
        num_anchors=args.num_anchors,
        device=device,
    )
    scene = dataset.get_scene(0)
    H, W = scene["H"], scene["W"]
    print(f"\nScene: {args.pt_paths[0]}")
    print(f"  Views: {scene['n_views']}  |  Resolution: {H}x{W}  |  Anchors: {args.num_anchors}")

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

    # ── CSV logger ─────────────────────────────────────────────────────
    csv_path = ckpt_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "loss", "l1", "ssim", "lpips", "depth_loss",
        "opacity_reg", "scale_reg", "opacity_mean", "scale_mean",
        "rendered_mean", "rendered_std", "grad_raw", "grad_clipped", "time_s",
    ])

    # ── Save initial weights ──────────────────────────────────────────
    torch.save({"unet": unet.state_dict(), "decoder": decoder.state_dict()},
               ckpt_dir / "init_weights.pt")

    # ── Training loop ───────────────────────────────────────────────────
    losses = all_losses.copy()
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {args.steps} steps, {args.num_views} views/step")
    print(f"{'='*60}")

    for step in range(start_step, args.steps):
        batch = dataset.sample_views(scene, args.num_views, device=device)
        u_input = batch["u_input"]        # [V, 10, H, W]
        anchors = batch["anchors"]         # [N, 3]

        # ── Forward ──────────────────────────────────────────────────
        features = unet(u_input)           # [V, 64, H, W]
        anchor_feats, vis = project_and_aggregate(
            anchors, features, batch["depths"], batch["poses"], batch["Ks"],
        )                                  # [N, 64]
        gs_out = decoder(anchor_feats)
        gaussians = assemble_gaussians(gs_out, anchors)

        # ── Render + Loss per view ───────────────────────────────────
        total_loss = 0.0
        loss_dict = {}

        for i in range(args.num_views):
            rendered, rendered_depth, _ = render_view(
                gaussians, batch["poses"][i], batch["Ks"][i], H, W,
            )
            gt_img = batch["gt_imgs"][i]
            gt_depth = batch["depths"][i].squeeze(0).unsqueeze(-1)

            ld = compute_loss(
                rendered, rendered_depth, gt_img, gt_depth,
                gaussians["opacities"], gaussians["scales"],
            )
            total_loss = total_loss + ld["total"]
            if i == 0:
                loss_dict = {k: v.item() for k, v in ld.items()}

        total_loss = total_loss / args.num_views
        loss_val = total_loss.item()

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
        scale_mean = gaussians["scales"].mean().item()
        rendered_mean = rendered.mean().item()
        rendered_std = rendered.std().item()

        # ── CSV log ──────────────────────────────────────────────────
        csv_writer.writerow([
            step, f"{loss_val:.4f}",
            f"{loss_dict.get('l1', 0):.4f}",
            f"{loss_dict.get('ssim', 0):.4f}",
            f"{loss_dict.get('lpips', 0):.4f}",
            f"{loss_dict.get('depth', 0):.4f}",
            f"{loss_dict.get('opacity_reg', 0):.4f}",
            f"{loss_dict.get('scale_reg', 0):.4f}",
            f"{opacity_mean:.4f}", f"{scale_mean:.4f}",
            f"{rendered_mean:.4f}", f"{rendered_std:.4f}",
            f"{raw_norm:.2f}", f"{grad_norm:.2f}", f"{time.time() - t_start:.1f}",
        ])

        # ── Console log ─────────────────────────────────────────────
        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"  Step {step:5d}/{args.steps}  |  "
                   f"loss={loss_val:.2f}  |  "
                   f"L1={loss_dict.get('l1', 0):.4f}  |  "
                   f"SSIM={loss_dict.get('ssim', 0):.4f}  |  "
                   f"α={opacity_mean:.3f}  |  "
                   f"s={scale_mean:.3f}  |  "
                   f"grad={raw_norm:.1f}/{grad_norm:.1f}  |  "
                   f"t={elapsed:.1f}s")

        # ── Save checkpoint ──────────────────────────────────────────
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

    # ── Loss curve ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Smoothed
    if len(losses) > 50:
        kernel = np.ones(20) / 20
        smoothed = np.convolve(losses, kernel, mode="valid")
        axes[1].plot(range(len(losses)), losses, alpha=0.3, color="tab:blue")
        axes[1].plot(range(19, len(losses)), smoothed, color="tab:blue")
    else:
        axes[1].plot(losses)
    axes[1].set_xlabel("Step")
    axes[1].set_title("Loss (with smoothing)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"AnchorSplat — {args.steps} steps  |  {args.pt_paths[0]}")
    fig.tight_layout()
    loss_path = ckpt_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    print(f"Loss curve → {loss_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s  ({elapsed / args.steps:.2f}s/step)")
    print(f"Metrics CSV → {csv_path}")


if __name__ == "__main__":
    main()
