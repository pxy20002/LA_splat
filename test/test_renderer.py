"""Test module 5: renderer + loss — shape tests (local) and rendering (server with GPU)."""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.renderer import assemble_gaussians, render_view, compute_loss
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate


def test_assemble():
    """Layer 1 — Gaussian assembly from decoder output."""
    print("=" * 60)
    print("Test 1 — Gaussian assembly")
    print("=" * 60)
    N = 128
    decoder = GaussianDecoder(in_dim=64).eval()
    x = torch.randn(N, 64)
    with torch.no_grad():
        out = decoder(x)
    anchors = torch.randn(N, 3)
    g = assemble_gaussians(out, anchors)

    assert g["means"].shape == (N * 4, 3)
    assert g["quats"].shape == (N * 4, 4)
    assert g["scales"].shape == (N * 4, 3)
    assert g["opacities"].shape == (N * 4,)
    assert g["colors"].shape == (N * 4, 3)
    # Means should be near anchors (offsets are small initially)
    anchor_centers = anchors.unsqueeze(1).expand(N, 4, 3).reshape(N * 4, 3)
    dist = (g["means"] - anchor_centers).norm(dim=1).mean()
    print(f"  Mean offset from anchors: {dist:.4f} (expected small)")
    print("  Result: PASS\n")


def test_loss():
    """Layer 2 — Loss computation produces reasonable values."""
    print("=" * 60)
    print("Test 2 — Loss computation")
    print("=" * 60)
    H, W = 256, 256
    r = torch.rand(H, W, 3)
    rd = torch.rand(H, W, 1)
    gt = r + 0.01 * torch.randn(H, W, 3)
    gtd = rd + 0.01 * torch.randn(H, W, 1)
    losses = compute_loss(r, rd, gt, gtd, torch.rand(100), torch.rand(100, 3))
    print(f"  L_total = {losses['total'].item():.2f}")
    print(f"  L1  = {losses['l1'].item():.4f}")
    print(f"  SSIM= {losses['ssim'].item():.4f}")
    print(f"  L_D = {losses['depth'].item():.4f}")
    assert losses["total"] > 0, "Loss should be positive"
    print("  Result: PASS\n")


def test_end_to_end(predictions_path: str):
    """Layer 3 — Full pipeline, excluding rendering (runs locally)."""
    print("=" * 60)
    print("Test 3 — Pipeline through Gaussian assembly")
    print("=" * 60)
    data = torch.load(predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    views = min(len(predictions), 4)
    print(f"  Views: {views}")

    result = AnchorPredictor.from_predictions(predictions, num_anchors=512)
    anchors = result["anchors"]

    unet = LightweightUNet(in_channels=10, out_channels=64).eval()
    decoder = GaussianDecoder(in_dim=64).eval()

    fmaps, depths, poses, Ks = [], [], [], []
    for pred in predictions[:views]:
        rgb = pred["img_no_norm"].squeeze(0)
        d = pred["depth_along_ray"].squeeze(0)
        K = pred["intrinsics"].squeeze(0)
        pose = pred["camera_poses"].squeeze(0)
        H, W = rgb.shape[0], rgb.shape[1]
        rays = compute_plucker_rays(K, pose, H, W)
        rgb_ch = rgb.permute(2, 0, 1)
        d_ch = d.squeeze(-1).unsqueeze(0)
        inp = torch.cat([rgb_ch, d_ch, rays], dim=0).unsqueeze(0)
        with torch.no_grad():
            f = unet(inp).squeeze(0)
        fmaps.append(f)
        depths.append(d.squeeze(-1).unsqueeze(0))
        poses.append(pose)
        Ks.append(K)

    fmaps_t = torch.stack(fmaps, dim=0)
    dmaps_t = torch.stack(depths, dim=0)
    poses_t = torch.stack(poses, dim=0)
    Ks_t = torch.stack(Ks, dim=0)

    anchor_feats, vis = project_and_aggregate(anchors, fmaps_t, dmaps_t, poses_t, Ks_t)
    with torch.no_grad():
        gs_out = decoder(anchor_feats)

    gaussians = assemble_gaussians(gs_out, anchors)
    print(f"  Gaussians: {gaussians['means'].shape[0]} total")
    print(f"  Visible anchors: {(vis.sum(dim=0) > 0).sum().item()}/{anchors.shape[0]}")
    print("  Result: PASS\n")


def test_render():
    """Layer 4 — Render synthetic Gaussians (requires CUDA)."""
    print("=" * 60)
    print("Test 4 — Rendering (CUDA required)")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    device = "cuda"
    N, H, W = 200, 256, 256
    means = torch.randn(N, 3, device=device) * 0.3
    means[:, 2] = means[:, 2].abs() + 1.5
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(torch.randn(N, 3, device=device) * 0.3) * 0.05
    opacities = torch.sigmoid(torch.randn(N, device=device)) * 0.5 + 0.25
    colors = torch.sigmoid(torch.randn(N, 3, device=device))
    K = torch.tensor([[300., 0, 128], [0, 300, 128], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device)

    gaussians = {"means": means, "quats": quats, "scales": scales,
                 "opacities": opacities, "colors": colors}

    rendered, rendered_depth, alpha = render_view(gaussians, viewmat, K, H, W)
    losses = compute_loss(rendered, rendered_depth, rendered, rendered_depth,
                          opacities, scales)
    print(f"  Rendered: {list(rendered.shape)}")
    print(f"  Depth:    {list(rendered_depth.shape)}")
    print(f"  Loss:     {losses['total'].item():.2f}")

    # Gradient test
    g = means.detach().clone().requires_grad_(True)
    g_gauss = {**gaussians, "means": g}
    r, rd, _ = render_view(g_gauss, viewmat, K, H, W)
    l = compute_loss(r, rd, r, rd, opacities, scales)["total"]
    l.backward()
    print(f"  Grad means: {g.grad.norm():.4f}")
    print("  Result: PASS\n")


def parse_args():
    p = argparse.ArgumentParser(description="Test renderer (module 5).")
    p.add_argument("--predictions_path", type=str,
                   default="outputs/mapanything_predictions_truck_v4.pt")
    return p.parse_args()


def main():
    args = parse_args()
    test_assemble()
    test_loss()
    test_end_to_end(args.predictions_path)
    test_render()


if __name__ == "__main__":
    main()
