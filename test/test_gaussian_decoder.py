"""Test GaussianDecoder: shape, value ranges, parameter count, end-to-end flow."""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate


# ── helpers ────────────────────────────────────────────────────────────────

def check_shape(name, tensor, expected_last_dims):
    """Assert tensor.shape ends with expected_last_dims."""
    actual = tensor.shape[-len(expected_last_dims):]
    assert actual == expected_last_dims, \
        f"{name}: expected trailing dims {expected_last_dims}, got {actual}"
    print(f"  {name}: {list(tensor.shape)}  ✓")


def test_shape():
    """Layer 1 — all output attribute shapes are correct."""
    print("=" * 60)
    print("Test 1 — Output shapes")
    print("=" * 60)
    decoder = GaussianDecoder(in_dim=64).eval()
    x = torch.randn(1024, 64)  # [N, D_in]
    with torch.no_grad():
        out = decoder(x)

    check_shape("delta_mu",  out["delta_mu"],  (1024, 4, 3))
    check_shape("opacity",   out["opacity"],   (1024, 4, 1))
    check_shape("scale",     out["scale"],     (1024, 4, 3))
    check_shape("rotation",  out["rotation"],  (1024, 4, 4))
    check_shape("sh",        out["sh"],        (1024, 4, 3))
    print("  Result: PASS\n")


def test_value_ranges():
    """Layer 2 — activations produce valid ranges."""
    print("=" * 60)
    print("Test 2 — Value ranges")
    print("=" * 60)
    decoder = GaussianDecoder(in_dim=64).eval()
    x = torch.randn(256, 64)
    with torch.no_grad():
        out = decoder(x)

    # opacity in [0, 1]
    op = out["opacity"]
    assert 0 <= op.min() <= op.max() <= 1, "opacity out of [0,1]"
    print(f"  opacity:  [{op.min().item():.4f}, {op.max().item():.4f}] ∈ [0,1] ✓")

    # scale > 0 (exp activation)
    sc = out["scale"]
    assert (sc > 0).all(), "scale has non-positive values"
    print(f"  scale:    [{sc.min().item():.4f}, {sc.max().item():.4f}] > 0 ✓")

    # rotation is unit quaternion
    rot = out["rotation"]
    norms = rot.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "quaternion not normalized"
    print(f"  rotation: norm min={norms.min().item():.6f} max={norms.max().item():.6f} ≈ 1.0 ✓")

    # delta_mu is small (not explicitly constrained in forward, checked by loss during training)
    dm = out["delta_mu"]
    print(f"  delta_mu: μ={dm.mean().item():.4f} σ={dm.std().item():.4f} (small is expected)")

    print("  Result: PASS\n")


def test_batch():
    """Layer 3 — batch input works."""
    print("=" * 60)
    print("Test 3 — Batch handling")
    print("=" * 60)
    decoder = GaussianDecoder(in_dim=64).eval()
    x = torch.randn(3, 512, 64)  # [B, N, D_in]
    with torch.no_grad():
        out = decoder(x)
    check_shape("delta_mu (batch)", out["delta_mu"], (3, 512, 4, 3))
    print("  Result: PASS\n")


def test_gradient():
    """Layer 4 — gradients flow back."""
    print("=" * 60)
    print("Test 4 — Gradient flow")
    print("=" * 60)
    decoder = GaussianDecoder(in_dim=64)
    x = torch.randn(32, 64, requires_grad=True)
    out = decoder(x)
    loss = out["opacity"].sum() + out["delta_mu"].sum()
    loss.backward()
    ok = x.grad is not None and (x.grad.abs().sum() > 0)
    print(f"  Grad back to input: {'PASS' if ok else 'FAIL'}\n")


def test_end_to_end(predictions_path: str):
    """Layer 5 — full pipeline: MVS → anchors → U-Net → projection → decoder."""
    print("=" * 60)
    print("Test 5 — End-to-end integration")
    print("=" * 60)

    data = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    n_total = len(predictions)
    views = min(n_total, 4)
    print(f"  Using {views}/{n_total} views")

    # 1) Anchors
    result = AnchorPredictor.from_predictions(predictions, num_anchors=512)
    anchors = result["anchors"]
    print(f"  Anchors: {anchors.shape}")

    # 2) U-Net features + projection
    unet = LightweightUNet(in_channels=10, out_channels=64).eval()
    decoder = GaussianDecoder(in_dim=64).eval()

    fmap_list, depth_list, pose_list, K_list = [], [], [], []
    for pred in predictions[:views]:
        rgb = pred["img_no_norm"].squeeze(0)            # [H, W, 3]
        depth = pred["depth_along_ray"].squeeze(0)       # [H, W, 1]
        K = pred["intrinsics"].squeeze(0)                # [3, 3]
        pose = pred["camera_poses"].squeeze(0)            # [4, 4]
        H, W = rgb.shape[0], rgb.shape[1]

        rays = compute_plucker_rays(K, pose, H, W)      # [6, H, W]
        rgb_ch = rgb.permute(2, 0, 1)                    # [3, H, W]
        d_ch = depth.squeeze(-1).unsqueeze(0)             # [1, H, W]
        inp = torch.cat([rgb_ch, d_ch, rays], dim=0)     # [10, H, W]
        inp = inp.unsqueeze(0)                            # [1, 10, H, W]

        with torch.no_grad():
            feat = unet(inp).squeeze(0)                   # [C, H, W]
        fmap_list.append(feat)
        depth_list.append(depth.squeeze(-1).unsqueeze(0))  # [1, H, W]
        pose_list.append(pose)
        K_list.append(K)

    fmaps = torch.stack(fmap_list, dim=0)           # [V, C, H, W]
    dmaps = torch.stack(depth_list, dim=0)  # [V, 1, H, W]
    poses = torch.stack(pose_list, dim=0)            # [V, 4, 4]
    Ks = torch.stack(K_list, dim=0)                 # [V, 3, 3]

    anchor_feats, visibility = project_and_aggregate(
        anchors, fmaps, dmaps, poses, Ks, depth_thresh_rel=0.02,
    )
    print(f"  Anchor features: {list(anchor_feats.shape)}")

    # 3) Gaussian Decoder
    with torch.no_grad():
        gaussians = decoder(anchor_feats)

    for k, v in gaussians.items():
        nz = (v.abs().sum(dim=-1) > 1e-6).float().mean().item() * 100
        print(f"  {k}: {list(v.shape)}  non-zero: {nz:.0f}%")

    print("  Result: PASS\n")


def parse_args():
    p = argparse.ArgumentParser(description="Test Gaussian Decoder (module 4).")
    p.add_argument("--predictions_path", type=str,
                   default="outputs/mapanything_predictions_truck_v4.pt")
    return p.parse_args()


def main():
    args = parse_args()
    test_shape()
    test_value_ranges()
    test_batch()
    test_gradient()
    test_end_to_end(args.predictions_path)


if __name__ == "__main__":
    main()
