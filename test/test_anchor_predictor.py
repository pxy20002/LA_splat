"""Test script for AnchorPredictor — loads MVS predictions, generates anchors, visualizes."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.anchor_predictor import AnchorPredictor


def build_point_cloud(points: torch.Tensor, color: list = None):
    """Convert a [N, 3] tensor to an open3d PointCloud."""
    if points.ndim == 3:
        points = points.reshape(-1, 3)
    pts = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None:
        colors = np.tile(color, (pts.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def parse_args():
    parser = argparse.ArgumentParser(description="Test AnchorPredictor with saved MVS predictions.")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="outputs/mapanything_predictions_v4.pt",
        help="Path to saved .pt predictions file.",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=1024,
        help="Number of anchors to sample.",
    )
    parser.add_argument(
        "--clip_percentile_low",
        type=float,
        default=0.02,
        help="Low percentile for 3D clipping.",
    )
    parser.add_argument(
        "--clip_percentile_high",
        type=float,
        default=0.98,
        help="High percentile for 3D clipping.",
    )
    parser.add_argument(
        "--max_display_points",
        type=int,
        default=50000,
        help="Max points to show in visualization (downsampled for speed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load predictions
    print(f"Loading predictions from: {args.predictions_path}")
    data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    print(f"Loaded {len(predictions)} views")

    for i, pred in enumerate(predictions):
        pts3d = pred["pts3d"]
        has_mask = "mask" in pred
        if has_mask:
            valid_count = pred["mask"].sum().item()
            total_count = pred["mask"].numel()
            pct = 100 * valid_count / total_count
        else:
            valid_count = "N/A"
            total_count = pts3d.shape[1] * pts3d.shape[2]
            pct = "N/A (no mask)"
        print(f"  View {i}: pts3d={list(pts3d.shape)}, valid={valid_count}/{total_count} "
              f"({pct})")

    # 2) Generate anchors
    print(f"\nGenerating {args.num_anchors} anchors "
          f"(clip percentile: [{args.clip_percentile_low}, {args.clip_percentile_high}])...")

    result = AnchorPredictor.from_predictions(
        predictions,
        num_anchors=args.num_anchors,
        clip_percentile=(args.clip_percentile_low, args.clip_percentile_high),
    )

    anchors = result["anchors"]  # [num_anchors, 3]
    clip_info = result["clip_info"]

    # 3) Print statistics
    print(f"\n--- Clipping Info ---")
    print(f"  X bounds: [{clip_info['x'][0]:.3f}, {clip_info['x'][1]:.3f}]")
    print(f"  Y bounds: [{clip_info['y'][0]:.3f}, {clip_info['y'][1]:.3f}]")
    print(f"  Z bounds: [{clip_info['z'][0]:.3f}, {clip_info['z'][1]:.3f}]")
    print(f"  Kept points: {clip_info['kept']:,} / {clip_info['total']:,} "
          f"({100 * clip_info['kept'] / max(1, clip_info['total']):.1f}%)")

    total_valid = sum(v.shape[0] for v in result["pts3d_per_view"])
    print(f"\n--- Anchor Statistics ---")
    print(f"  Total valid points (all views): {total_valid:,}")
    print(f"  Anchors sampled: {anchors.shape[0]}")
    print(f"  Anchor coordinate range:")
    for dim, name in enumerate(["X", "Y", "Z"]):
        print(f"    {name}: [{anchors[:, dim].min():.3f}, {anchors[:, dim].max():.3f}]")

    # 4) Build visualization
    combined = torch.cat(result["pts3d_per_view"], dim=0)
    # Downsample display points for speed
    if combined.shape[0] > args.max_display_points:
        idx = torch.randperm(combined.shape[0])[: args.max_display_points]
        combined = combined[idx]

    pcd_points = build_point_cloud(combined, color=[0.5, 0.5, 0.5])   # gray
    pcd_anchors = build_point_cloud(anchors, color=[1.0, 0.0, 0.0])    # red

    o3d.visualization.draw_geometries(
        [pcd_points, pcd_anchors],
        window_name="AnchorPredictor — gray=points, red=anchors",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    main()
