"""Test script for LightweightUNet — forward pass, parameter count, feature visualization."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.unet import LightweightUNet
from anchorsplat.ray_embeddings import compute_plucker_rays


def build_unet_inputs_from_predictions(predictions: list) -> list:
    """Build [10, H, W] tensors from a list of prediction dicts."""
    inputs = []
    for pred in predictions:
        rgb = pred["img_no_norm"].cpu()
        depth = pred["depth_along_ray"].cpu()
        K = pred["intrinsics"].squeeze(0).cpu()
        pose = pred["camera_poses"].squeeze(0).cpu()
        H, W = rgb.shape[1], rgb.shape[2]

        rays = compute_plucker_rays(K, pose, H, W)  # [6, H, W]

        rgb_ch = rgb.squeeze(0).permute(2, 0, 1)    # [3, H, W]
        d_ch = depth.squeeze(0).squeeze(-1).unsqueeze(0)  # [1, H, W]
        inp = torch.cat([rgb_ch, d_ch, rays], dim=0)      # [10, H, W]
        inputs.append(inp)
    return inputs


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Test LightweightUNet forward pass and visualize features.")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="outputs/mapanything_predictions_truck_v4.pt",
        help="Path to saved .pt predictions file (used to build U-Net inputs).",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to pre-saved unet_inputs.pt (skips prediction loading).",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="U-Net output channels (feature dimension).",
    )
    parser.add_argument(
        "--num_display_channels",
        type=int,
        default=8,
        help="How many output channels to show as heatmaps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Build or load U-Net inputs
    if args.input_path is not None:
        print(f"Loading U-Net inputs from: {args.input_path}")
        data = torch.load(args.input_path, map_location="cpu", weights_only=False)
        inputs = data["inputs"]
        print(f"Loaded {len(inputs)} tensors")
    else:
        print(f"Building U-Net inputs from: {args.predictions_path}")
        data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)
        predictions = data["predictions"]
        inputs = build_unet_inputs_from_predictions(predictions)
        print(f"Built {len(inputs)} tensors from {len(predictions)} views")

    # 2) Create model
    H, W = inputs[0].shape[1], inputs[0].shape[2]
    model = LightweightUNet(in_channels=10, out_channels=args.out_channels)
    model.eval()
    n_params = count_parameters(model)
    print(f"\nU-Net params: {n_params:,}  |  input: [10, {H}, {W}]  |  output: [{args.out_channels}, {H}, {W}]")

    # 3) Forward pass (batch of 2 views to verify batch handling)
    batch = torch.stack(inputs[:2], dim=0)  # [2, 10, H, W]
    print(f"Batch shape: {list(batch.shape)}")

    with torch.no_grad():
        features = model(batch)  # [2, C_out, H, W]
    print(f"Output shape: {list(features.shape)}")
    print(f"Output stats: mean={features.mean():.4f}  std={features.std():.4f}  "
          f"min={features.min():.4f}  max={features.max():.4f}")

    # 4) Visualize: show first N output channels of first view
    feat = features[0]  # [C_out, H, W]
    n_disp = min(args.num_display_channels, args.out_channels)

    cols = 4
    rows = (n_disp + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for c in range(n_disp):
        ch = feat[c].cpu().numpy()
        im = axes[c].imshow(ch, cmap="RdBu_r", interpolation="nearest")
        axes[c].set_title(f"ch {c}")
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c], fraction=0.046)

    for c in range(n_disp, len(axes)):
        axes[c].axis("off")

    fig.suptitle(
        f"U-Net Output Features — first {n_disp} of {args.out_channels} channels\n"
        f"Params: {n_params:,}  |  Shape: [{args.out_channels}, {H}, {W}]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
