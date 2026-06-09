"""Evaluate a trained checkpoint: render views and compare with GT."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.dataset import SceneDataset
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.renderer import assemble_gaussians, render_view


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained AnchorSplat checkpoint.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to checkpoint .pt file.")
    inp = p.add_mutually_exclusive_group(required=False)
    inp.add_argument("--image_dir", type=str, default=None,
                     help="Folder of input images.")
    inp.add_argument("--pt_path", type=str, default=None,
                     help="Cached .pt prediction file.")
    p.add_argument("--num_anchors", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_path", type=str, default=None,
                   help="Save comparison image to this path.")
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Which views to render (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"  Trained for {ckpt['step'] + 1} steps")
    print(f"  Final loss: {ckpt['losses'][-1]:.2f}")

    # ── Load dataset ─────────────────────────────────────────────────
    if args.image_dir is None and args.pt_path is None:
        args.pt_path = "datasets/mapanything_predictions_truck_v4.pt"
    dataset = SceneDataset(
        image_dir=args.image_dir, pt_path=args.pt_path,
        num_anchors=args.num_anchors, device=device,
    )
    batch = dataset.get_batch()
    H, W = batch["H"], batch["W"]

    # ── Build models, load weights ──────────────────────────────────
    unet = LightweightUNet(in_channels=10, out_channels=64).eval().to(device)
    decoder = GaussianDecoder(in_dim=64).eval().to(device)
    unet.load_state_dict(ckpt["unet"])
    decoder.load_state_dict(ckpt["decoder"])

    # ── Full pipeline ───────────────────────────────────────────────
    u_input = batch["u_input"]
    anchors = batch["anchors"]
    depths_t = batch["depths"]
    poses_t = batch["poses"]
    Ks_t = batch["Ks"]

    with torch.no_grad():
        features = unet(u_input)  # [V, 64, H, W]

    anchor_feats, vis = project_and_aggregate(
        anchors, features, depths_t, poses_t, Ks_t,
    )

    with torch.no_grad():
        gs_out = decoder(anchor_feats, anchors)
    gaussians = assemble_gaussians(gs_out, anchors)

    # ── Render selected views ───────────────────────────────────────
    view_indices = args.views if args.views is not None else list(range(dataset.n_views))
    n_v = len(view_indices)

    fig, axes = plt.subplots(n_v, 5, figsize=(21, 4 * n_v),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.08]})
    if n_v == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_v):
        axes[row, 4].axis("off")  # hide colorbar placeholder

    # First pass: collect all depth values for unified colorbar
    all_d_vals, all_gt_vals = [], []
    render_results = []
    for vi in view_indices:
        with torch.no_grad():
            rendered, depth, _ = render_view(
                gaussians, poses_t[vi], Ks_t[vi], H, W,
            )
        d_np = depth.cpu().squeeze(-1).numpy()
        gt_d = batch["depths"].squeeze(1)[vi].cpu().numpy()
        all_d_vals.append(d_np)
        all_gt_vals.append(gt_d)
        render_results.append((vi, rendered, d_np, gt_d))

    vmin = min(np.min(d) for d in all_d_vals + all_gt_vals)
    vmax = max(np.max(d) for d in all_d_vals + all_gt_vals)

    for row, (vi, rendered, d_np, gt_d) in enumerate(render_results):
        # 1) GT RGB
        gt = batch["gt_imgs"][vi].cpu().numpy()
        gt_disp = (np.clip(gt, 0, 1) * 255).astype(np.uint8) if gt.max() <= 1.0 else np.clip(gt, 0, 255).astype(np.uint8)
        axes[row, 0].imshow(gt_disp)
        axes[row, 0].set_title(f"View {vi} — GT")
        axes[row, 0].axis("off")

        # 2) Rendered RGB
        r_np = rendered.cpu().numpy()
        r_disp = np.clip(r_np, 0, 1) if r_np.max() <= 1.1 else np.clip(r_np / 255., 0, 1)
        axes[row, 1].imshow(r_disp)
        axes[row, 1].set_title("Rendered")
        axes[row, 1].axis("off")

        # 3) GT Depth (MVS)
        im2 = axes[row, 2].imshow(gt_d, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[row, 2].set_title("GT Depth")
        axes[row, 2].axis("off")

        # 4) Rendered Depth
        im1 = axes[row, 3].imshow(d_np, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[row, 3].set_title("Rendered Depth")
        axes[row, 3].axis("off")
        # Colorbar in dedicated 5th column
        cbar = fig.colorbar(im1, cax=axes[row, 4])
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])

    fig.suptitle(f"AnchorSplat — step {ckpt['step'] + 1}  |  "
                 f"{n_v} views  |  {gaussians['means'].shape[0]} Gaussians  |  "
                 f"Depth [{vmin:.2f}, {vmax:.2f}]",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
