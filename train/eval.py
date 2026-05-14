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
    p.add_argument("--pt_path", type=str,
                   default="datasets/mapanything_predictions_truck_v4.pt")
    p.add_argument("--num_anchors", type=int, default=512)
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
    dataset = SceneDataset(
        [str(Path(args.pt_path).resolve())],
        num_anchors=args.num_anchors, device=device,
    )
    scene = dataset.get_scene(0)
    H, W = scene["H"], scene["W"]

    # ── Build models, load weights ──────────────────────────────────
    unet = LightweightUNet(in_channels=10, out_channels=64).eval().to(device)
    decoder = GaussianDecoder(in_dim=64).eval().to(device)
    unet.load_state_dict(ckpt["unet"])
    decoder.load_state_dict(ckpt["decoder"])

    # ── Full pipeline ───────────────────────────────────────────────
    u_input = scene["unet_inputs"].to(device)  # [V, 10, H, W]
    anchors = scene["anchors"].to(device)

    with torch.no_grad():
        features = unet(u_input)  # [V, 64, H, W]

    depths_t = torch.stack(scene["depths"], dim=0).squeeze(-1).unsqueeze(1).to(device)
    poses_t = torch.stack(scene["poses"], dim=0).to(device)
    Ks_t = torch.stack(scene["Ks"], dim=0).to(device)

    anchor_feats, vis = project_and_aggregate(
        anchors, features, depths_t, poses_t, Ks_t,
    )

    with torch.no_grad():
        gs_out = decoder(anchor_feats)
    gaussians = assemble_gaussians(gs_out, anchors)

    # ── Render selected views ───────────────────────────────────────
    view_indices = args.views if args.views is not None else list(range(scene["n_views"]))
    n_v = len(view_indices)

    fig, axes = plt.subplots(n_v, 3, figsize=(15, 4 * n_v))
    if n_v == 1:
        axes = axes.reshape(1, -1)

    for row, vi in enumerate(view_indices):
        with torch.no_grad():
            rendered, depth, _ = render_view(
                gaussians, poses_t[vi], Ks_t[vi], H, W,
            )

        # GT
        gt = scene["imgs"][vi].numpy()
        gt_disp = (np.clip(gt, 0, 1) * 255).astype(np.uint8) if gt.max() <= 1.0 else np.clip(gt, 0, 255).astype(np.uint8)
        axes[row, 0].imshow(gt_disp)
        axes[row, 0].set_title(f"View {vi} — GT")
        axes[row, 0].axis("off")

        # Rendered
        r_np = rendered.cpu().numpy()
        r_disp = np.clip(r_np, 0, 1) if r_np.max() <= 1.1 else np.clip(r_np / 255., 0, 1)
        axes[row, 1].imshow(r_disp)
        axes[row, 1].set_title("Rendered")
        axes[row, 1].axis("off")

        # Depth
        d_np = depth.cpu().squeeze(-1).numpy()
        im = axes[row, 2].imshow(d_np, cmap="jet")
        axes[row, 2].set_title(f"Depth  [{d_np.min():.2f}, {d_np.max():.2f}]")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

    # Loss curve (small inset or separate)
    fig.suptitle(f"AnchorSplat -- step {ckpt['step'] + 1}  |  "
                 f"{n_v} views  |  {gaussians['means'].shape[0]} Gaussians",
                 fontsize=12)
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
