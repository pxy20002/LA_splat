"""Test FeatureProjector: synthetic projection check, visibility filtering, integration."""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.feature_projector import project_and_aggregate
from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays
from anchorsplat.unet import LightweightUNet


def test_synthetic_projection():
    """Layer 1: known 3D point → known 2D location → verify feature sampling."""
    print("=" * 60)
    print("Test 1 — Synthetic projection correctness")
    print("=" * 60)

    H, W = 64, 64
    # A point at (0, 0, 5) in world = 5m directly in front of origin camera
    anchor = torch.tensor([[0.0, 0.0, 5.0]])  # [1, 3]

    # Camera at origin, looking +Z
    pose = torch.eye(4)
    # Intrinsics: cx=32, cy=32, focal so that center pixel maps to (0,0,1)
    K = torch.tensor([[100.0, 0.0, 32.0],
                      [0.0, 100.0, 32.0],
                      [0.0, 0.0, 1.0]])

    # Feature map: put unique value at center pixel (32, 32)
    feat = torch.zeros(1, 3, H, W)
    feat[0, :, 32, 32] = torch.tensor([1.0, 2.0, 3.0])

    depths = torch.ones(1, 1, H, W) * 5.0  # depth map: 5m everywhere

    result, vis = project_and_aggregate(
        anchor, feat, depths,
        camera_poses=pose.unsqueeze(0),
        intrinsics=K.unsqueeze(0),
        depth_thresh_rel=0.02,
    )

    ok = torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4) and vis[0, 0].item()
    print(f"  Anchor at camera center → feature sample: {result[0]}")
    print(f"  Visibility:      {vis[0, 0].item()} (expected True)")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    assert ok, "Projection is incorrect!"
    print()


def test_visibility_filtering():
    """Layer 2: anchors behind camera / out of bounds / occluded are properly excluded."""
    print("=" * 60)
    print("Test 2 — Visibility filtering")
    print("=" * 60)

    H, W = 64, 64
    K = torch.tensor([[100.0, 0.0, 32.0],
                      [0.0, 100.0, 32.0],
                      [0.0, 0.0, 1.0]])
    pose = torch.eye(4)

    feat = torch.randn(1, 3, H, W)
    depths = torch.ones(1, 1, H, W) * 5.0

    # Three anchors:
    #   a: in front, in bounds, depth consistent → VISIBLE
    #   b: behind camera (z<0)                  → INVISIBLE
    #   c: outside image bounds                 → INVISIBLE
    anchors = torch.tensor([
        [0.0, 0.0, 5.0],     # visible
        [0.0, 0.0, -1.0],    # behind camera
        [0.0, 100.0, 5.0],   # out of view (projects above image)
    ])

    _, vis = project_and_aggregate(
        anchors, feat, depths,
        camera_poses=pose.unsqueeze(0),
        intrinsics=K.unsqueeze(0),
        depth_thresh_rel=0.02,
    )

    expected = [True, False, False]
    passed = all(vis[0, i].item() == expected[i] for i in range(3))
    for i in range(3):
        print(f"  Anchor {i} visible: {vis[0, i].item()} (expected {expected[i]})")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    assert passed, "Visibility filtering is wrong!"
    print()


def test_real_data(predictions_path: str):
    """Layer 3: end-to-end integration with real predictions."""
    print("=" * 60)
    print("Test 3 — Real data integration")
    print("=" * 60)

    data = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    n_views = len(predictions)
    print(f"  Loaded {n_views} views")

    # Generate anchors
    result = AnchorPredictor.from_predictions(predictions, num_anchors=512)
    anchors = result["anchors"]  # [N, 3]
    print(f"  Anchors: {anchors.shape}")

    # Build U-Net inputs per view
    feature_maps = []
    depth_maps = []
    all_poses = []
    all_K = []

    unet = LightweightUNet(in_channels=10, out_channels=64)
    unet.eval()

    for pred in predictions[:max(4, n_views)]:
        rgb = pred["img_no_norm"].squeeze(0).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        depth = pred["depth_along_ray"].squeeze(0).squeeze(-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        K = pred["intrinsics"].squeeze(0)  # [3, 3]
        pose = pred["camera_poses"].squeeze(0)  # [4, 4]
        H, W = rgb.shape[2], rgb.shape[3]

        rays = compute_plucker_rays(K, pose, H, W)  # [6, H, W]
        u_input = torch.cat([rgb.squeeze(0), depth.squeeze(0).squeeze(0).unsqueeze(0), rays], dim=0)  # [10, H, W]
        u_input = u_input.unsqueeze(0)  # [1, 10, H, W]

        with torch.no_grad():
            feat = unet(u_input)  # [1, C, H, W]
        feature_maps.append(feat.squeeze(0))
        depth_maps.append(depth.squeeze(0))
        all_poses.append(pose)
        all_K.append(K)

    fmaps = torch.stack(feature_maps, dim=0)  # [V, C, H, W]
    dmaps = torch.stack(depth_maps, dim=0)    # [V, 1, H, W]
    poses = torch.stack(all_poses, dim=0)      # [V, 4, 4]
    Ks = torch.stack(all_K, dim=0)            # [V, 3, 3]

    V, C = fmaps.shape[0], fmaps.shape[1]
    print(f"  Feature maps: [{V}, {C}, {H}, {W}]")

    # Project & aggregate
    anchor_feats, visibility = project_and_aggregate(
        anchors, fmaps, dmaps, poses, Ks, depth_thresh_rel=0.02,
    )
    print(f"  Anchor features: {list(anchor_feats.shape)}")
    print(f"  Visibility: [{V}, {anchors.shape[0]}]")

    # Statistics
    views_per_anchor = visibility.sum(dim=0)
    views_per_anchor_f = views_per_anchor.float()
    print(f"  Views per anchor: min={views_per_anchor.min().item()}, "
          f"mean={views_per_anchor_f.mean().item():.1f}, "
          f"max={views_per_anchor.max().item()}")
    print(f"  Anchors seen by 0 views: {(views_per_anchor == 0).sum().item()}")
    nonzero_feat = (anchor_feats.norm(dim=1) > 1e-6).sum().item()
    print(f"  Non-zero feature anchors: {nonzero_feat}/{anchors.shape[0]}")

    # Sanity
    assert anchor_feats.shape == (anchors.shape[0], C), "Output shape mismatch"
    assert views_per_anchor.max() <= V, "Too many visible views"
    print("  Result: PASS")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Test feature projector (module 3).")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="outputs/mapanything_predictions_truck_v4.pt",
        help="Path to .pt predictions for integration test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_synthetic_projection()
    test_visibility_filtering()
    test_real_data(args.predictions_path)


if __name__ == "__main__":
    main()
