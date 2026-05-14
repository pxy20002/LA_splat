"""Test script for Plücker ray embeddings — computes rays and visualizes."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.ray_embeddings import compute_plucker_rays


def _normalize_for_display(tensor_3ch: torch.Tensor) -> np.ndarray:
    """Map a [3, H, W] tensor to [H, W, 3] float32 image (per-channel min-max normalization)."""
    arr = tensor_3ch.detach().cpu().numpy()  # [3, H, W]
    img = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
    for c in range(3):
        ch = arr[c]
        lo, hi = ch.min(), ch.max()
        if hi - lo > 1e-8:
            img[:, :, c] = (ch - lo) / (hi - lo)
        else:
            img[:, :, c] = 0.5
    return img


def visualize_one_view(pred, view_idx, save_path=None, show=True, cmap="jet"):
    """Generate a 2×2 figure for a single view. Returns the 6-ch ray tensor.

    Args:
        pred: view prediction dict.
        view_idx: view index (for title).
        save_path: if set, save figure to this path (after show if show=True).
        show: if True, display the interactive window. Set False for batch saving.
        cmap: matplotlib colormap for depth display (default: 'jet').
    """
    img = pred["img_no_norm"].squeeze(0)        # [H, W, 3]
    depth = pred["depth_along_ray"].squeeze(0)   # [H, W, 1]
    K = pred["intrinsics"].squeeze(0)            # [3, 3]
    pose = pred["camera_poses"].squeeze(0)       # [4, 4]

    H, W = img.shape[0], img.shape[1]

    rays = compute_plucker_rays(K, pose, H, W)   # [6, H, W]
    direction = rays[:3]                          # [3, H, W]
    moment = rays[3:6]                            # [3, H, W]

    # Build 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    rgb = img.detach().cpu().numpy()
    rgb_disp = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.max() > 10 else np.clip(rgb, 0, 1)
    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title("RGB Image")
    axes[0, 0].axis("off")

    d = depth.squeeze(-1).detach().cpu().numpy()
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / max(d_max - d_min, 1e-8)
    im_d = axes[0, 1].imshow(d_norm, cmap=cmap)
    axes[0, 1].set_title(f"Depth (along ray, {cmap})\nmin={d_min:.3f}  max={d_max:.3f}")
    axes[0, 1].axis("off")
    plt.colorbar(im_d, ax=axes[0, 1], fraction=0.046)

    dir_img = _normalize_for_display(direction)
    axes[1, 0].imshow(dir_img)
    axes[1, 0].set_title("Plücker Direction (R=X, G=Y, B=Z)")
    axes[1, 0].axis("off")

    mom_img = _normalize_for_display(moment)
    axes[1, 1].imshow(mom_img)
    axes[1, 1].set_title("Plücker Moment  (camera_center × direction)")
    axes[1, 1].axis("off")

    fig.suptitle(
        f"Plücker Ray Embeddings — View {view_idx}  |  "
        f"Dir range: [{direction.min():.3f}, {direction.max():.3f}]  |  "
        f"Moment range: [{moment.min():.3f}, {moment.max():.3f}]",
        fontsize=10,
    )
    plt.tight_layout()

    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rays


def build_unet_input(rgb: torch.Tensor, depth: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    """Stack RGB(3) + Depth(1) + Plücker(6) → [10, H, W]."""
    # rgb:     [H,W,3] or [1,H,W,3]
    # depth:   [H,W,1] or [1,H,W,1]
    # rays:    [6,H,W]
    if rgb.dim() == 4:
        rgb = rgb.squeeze(0)
    if depth.dim() == 4:
        depth = depth.squeeze(0)

    rgb_ch = rgb.permute(2, 0, 1)           # [3, H, W]
    d_ch = depth.squeeze(-1).unsqueeze(0)    # [1, H, W]
    u_input = torch.cat([rgb_ch, d_ch, rays.cpu()], dim=0)  # [10, H, W]
    return u_input


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize Plücker ray embeddings from MVS predictions."
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="outputs/mapanything_predictions_truck_v4.pt",
        help="Path to saved .pt predictions file.",
    )
    parser.add_argument(
        "--view_index",
        type=int,
        default=None,
        help="Visualize a single view (interactive window).",
    )
    parser.add_argument(
        "--all_views",
        action="store_true",
        help="Generate PNG visualizations for all views.",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="Limit number of views with --all_views or --slideshow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ray_vis",
        help="Directory for saved PNGs (used with --all_views or --slideshow).",
    )
    parser.add_argument(
        "--slideshow",
        action="store_true",
        help="Interactive mode: show each view, save PNG on close, then show next view.",
    )
    parser.add_argument(
        "--save_unet_inputs",
        action="store_true",
        help="Save [10, H, W] U-Net inputs as a .pt file for module 2b.",
    )
    parser.add_argument(
        "--unet_inputs_path",
        type=str,
        default="outputs/unet_inputs.pt",
        help="Output path for U-Net inputs.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        choices=["jet", "turbo", "viridis", "plasma", "inferno", "magma"],
        help="Colormap for depth display (default: jet, matches AnchorSplat paper).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading: {args.predictions_path}")
    data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    print(f"Loaded {len(predictions)} views")

    # --- Single view mode ---
    if args.view_index is not None:
        pred = predictions[args.view_index]
        H, W = pred["img_no_norm"].shape[1], pred["img_no_norm"].shape[2]
        print(f"View {args.view_index}: H={H}, W={W}")
        rays = visualize_one_view(pred, args.view_index, cmap=args.cmap)
        return

    # --- Slideshow mode: show each view, save on close ---
    if args.slideshow:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        views = range(len(predictions))
        if args.max_views is not None:
            views = views[: args.max_views]

        print(f"Slideshow mode: {len(views)} views. Close each window to advance.")
        for i in views:
            png_path = out_dir / f"view_{i:04d}.png"
            print(f"  View {i}/{len(views) - 1}  →  {png_path}")
            visualize_one_view(predictions[i], i, save_path=str(png_path), show=True, cmap=args.cmap)

        print(f"Done. {len(views)} PNGs saved to {out_dir}/")
        return

    # --- All views batch mode (no display) ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = range(len(predictions))
    if args.max_views is not None:
        views = views[: args.max_views]

    print(f"Batch generating {len(views)} views → {out_dir}/")
    for i in views:
        png_path = out_dir / f"view_{i:04d}.png"
        visualize_one_view(predictions[i], i, save_path=str(png_path), show=False, cmap=args.cmap)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(views)} done")

    print(f"Done. {len(views)} PNGs saved to {out_dir}/")

    # --- Optional: build and save U-Net inputs ---
    if args.save_unet_inputs:
        unet_inputs = []
        for pred in predictions:
            rgb = pred["img_no_norm"].cpu()
            depth = pred["depth_along_ray"].cpu()
            K = pred["intrinsics"].squeeze(0).cpu()
            pose = pred["camera_poses"].squeeze(0).cpu()
            H, W = rgb.shape[1], rgb.shape[2]
            rays = compute_plucker_rays(K, pose, H, W)  # [6, H, W]
            inp = build_unet_input(rgb, depth, rays)     # [10, H, W]
            unet_inputs.append(inp)

        unet_path = Path(args.unet_inputs_path)
        unet_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"inputs": unet_inputs, "meta": data.get("meta", {})}, unet_path)
        print(f"Saved U-Net inputs ({len(unet_inputs)} views) to {unet_path}")


if __name__ == "__main__":
    main()


# 三种模式：                                                             
                                
# ┌────────────────┬──────────────────────────────────────┐
# │      命令       │                 行为                 │
# ├────────────────┼──────────────────────────────────────┤
# │ --view_index 0 │ 弹出单张窗口，不保存                 │                                                                              
# ├────────────────┼──────────────────────────────────────┤
# │ --all_views    │ 后台批量保存 PNG，不弹窗             │                                                                              
# ├────────────────┼──────────────────────────────────────┤                                                                            
# │ --slideshow    │ 逐个弹窗，关闭后自动保存并跳到下一张 │                                                                              
# └────────────────┴──────────────────────────────────────┘          

# 默认 colormap 改为 jet（深蓝→青→绿→黄→红，匹配论文风格）
# 深度做 min-max 归一化拉伸对比度，近处深蓝、远处红，区分度明显提升
# --cmap 可选值：jet（论文同款）、turbo（jet 的升级版）、viridis、plasma、inferno、magma