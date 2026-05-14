"""End-to-end sanity check: real predictions → full pipeline → render + save comparison."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.renderer import assemble_gaussians, render_view


def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline end-to-end test.")
    p.add_argument("--predictions_path", type=str,
                   default="outputs/mapanything_predictions_truck_v4.pt")
    p.add_argument("--num_anchors", type=int, default=512)
    p.add_argument("--view_to_render", type=int, default=0,
                   help="Which view to render and compare against GT.")
    p.add_argument("--save_path", type=str, default=None,
                   help="Save comparison image to this path.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) Load
    print(f"\nLoading: {args.predictions_path}")
    data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    n_views = len(predictions)
    print(f"  Views: {n_views}")

    # 2) Anchors
    print(f"\n--- Anchor Predictor ({args.num_anchors} anchors) ---")
    result = AnchorPredictor.from_predictions(predictions, num_anchors=args.num_anchors)
    anchors = result["anchors"]
    print(f"  Anchors: {anchors.shape}")

    # 3) U-Net features + projection
    print("\n--- U-Net + Feature Projection ---")
    unet = LightweightUNet(in_channels=10, out_channels=64).eval().to(device)
    decoder = GaussianDecoder(in_dim=64).eval().to(device)

    fmaps, dmaps, poses, Ks, imgs = [], [], [], [], []
    for pred in predictions:
        rgb = pred["img_no_norm"].squeeze(0)
        depth = pred["depth_along_ray"].squeeze(0)
        K = pred["intrinsics"].squeeze(0)
        pose = pred["camera_poses"].squeeze(0)
        H, W = rgb.shape[0], rgb.shape[1]

        rays = compute_plucker_rays(K, pose, H, W)
        rgb_ch = rgb.permute(2, 0, 1).to(device)
        d_ch = depth.squeeze(-1).unsqueeze(0).to(device)
        inp = torch.cat([rgb_ch, d_ch, rays.to(device)], dim=0).unsqueeze(0)

        with torch.no_grad():
            f = unet(inp).squeeze(0).cpu()
        fmaps.append(f)
        dmaps.append(depth.squeeze(-1).unsqueeze(0))
        poses.append(pose)
        Ks.append(K)
        imgs.append(rgb)

    fmaps_t = torch.stack(fmaps, dim=0)
    dmaps_t = torch.stack(dmaps, dim=0)
    poses_t = torch.stack(poses, dim=0)
    Ks_t = torch.stack(Ks, dim=0)

    anchor_feats, vis = project_and_aggregate(anchors, fmaps_t, dmaps_t, poses_t, Ks_t)
    visible = (vis.sum(dim=0) > 0).sum().item()
    print(f"  Anchor features: {list(anchor_feats.shape)}")
    print(f"  Visible anchors: {visible}/{args.num_anchors}")

    # 4) Gaussian Decoder
    print("\n--- Gaussian Decoder ---")
    with torch.no_grad():
        gs_out = decoder(anchor_feats.to(device))
    gaussians = assemble_gaussians(gs_out, anchors.to(device))
    print(f"  Gaussians: {gaussians['means'].shape[0]}")

    # 5) Render one view
    ri = args.view_to_render
    print(f"\n--- Render View {ri} ---")
    gs_gpu = {k: v.to(device) for k, v in gaussians.items()}
    rendered, depth, alpha = render_view(
        gs_gpu, poses_t[ri].to(device), Ks_t[ri].to(device), H, W,
    )
    rendered = rendered.cpu()
    depth = depth.cpu()
    alpha = alpha.cpu()
    print(f"  Rendered: {list(rendered.shape)}")
    print(f"  Depth:    {list(depth.shape)}")

    # 6) Visualize: GT | Rendered | Depth | Alpha
    gt = imgs[ri]
    gt_np = gt.numpy()
    if gt_np.max() > 10:
        gt_disp = np.clip(gt_np, 0, 255).astype(np.uint8)
    elif gt_np.max() <= 1.0:
        gt_disp = np.clip(gt_np * 255, 0, 255).astype(np.uint8) if gt_np.max() >= 0 else np.clip(gt_np, 0, 1)
    else:
        gt_disp = np.clip(gt_np, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt_disp)
    axes[0].set_title("GT Image")
    axes[0].axis("off")

    rendered_np = rendered.numpy()
    if rendered_np.max() > 10:
        rendered_disp = np.clip(rendered_np, 0, 255).astype(np.uint8)
    elif rendered_np.max() >= 0:
        rendered_disp = np.clip(rendered_np, 0, 1)
    else:
        rendered_disp = (rendered_np - rendered_np.min()) / max(rendered_np.max() - rendered_np.min(), 1e-8)
    axes[1].imshow(rendered_disp)
    axes[1].set_title("Rendered")
    axes[1].axis("off")

    d = depth.squeeze(-1).numpy()
    im_d = axes[2].imshow(d, cmap="jet")
    axes[2].set_title(f"Rendered Depth\nmin={d.min():.3f} max={d.max():.3f}")
    axes[2].axis("off")
    plt.colorbar(im_d, ax=axes[2], fraction=0.046)

    a = alpha.squeeze(-1).numpy()
    axes[3].imshow(a, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Alpha\nmax={a.max():.3f}")
    axes[3].axis("off")

    fig.suptitle(f"AnchorSplat E2E — View {ri}  |  {visible}/{args.num_anchors} anchors visible  |  "
                 f"{gaussians['means'].shape[0]} Gaussians", fontsize=12)
    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
