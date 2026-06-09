"""Gradient verification: check that loss gradients flow through ALL trainable modules."""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays
from anchorsplat.unet import LightweightUNet
from anchorsplat.feature_projector import project_and_aggregate
from anchorsplat.gaussian_decoder import GaussianDecoder
from anchorsplat.renderer import assemble_gaussians, render_view, compute_loss


def check_grad(name, module):
    """Report gradient statistics for a module's trainable parameters."""
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    if not grads:
        print(f"  {name}: no trainable params (frozen)")
        return

    nonzeros = sum((g is not None and g.abs().sum() > 0).item() for g in grads)
    total = len(grads)
    max_grad = max(g.abs().max().item() if g is not None else 0 for g in grads)

    status = "PASS" if nonzeros == total else "FAIL"
    info = f"has_grad={nonzeros}/{total}"
    if nonzeros < total:
        zero_params = [i for i, g in enumerate(grads) if g is None or g.abs().sum() == 0]
        info += f" zero_params={zero_params[:5]}"
    print(f"  {name}: max_grad={max_grad:.6f}  {info}  [{status}]")
    return nonzeros == total


def parse_args():
    p = argparse.ArgumentParser(description="Gradient flow verification for full AnchorSplat pipeline.")
    p.add_argument("--predictions_path", type=str,
                   default="outputs/mapanything_predictions_truck_v4.pt")
    p.add_argument("--num_anchors", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1) Load predictions ──────────────────────────────────────────
    print(f"\nLoading: {args.predictions_path}")
    data = torch.load(args.predictions_path, map_location="cpu", weights_only=False)
    predictions = data["predictions"]
    n_views = len(predictions)
    # Use 2 views for speed
    use_views = min(n_views, 2)
    print(f"  Using {use_views}/{n_views} views")

    # ── 2) Anchors ────────────────────────────────────────────────────
    print(f"\n--- Anchor Predictor ({args.num_anchors} anchors) ---")
    result = AnchorPredictor.from_predictions(predictions, num_anchors=args.num_anchors)
    anchors = result["anchors"].to(device)
    print(f"  Anchors: {anchors.shape}")

    # ── 3) Build U-Net inputs ─────────────────────────────────────────
    print("\n--- Building U-Net inputs ---")
    fmaps, dmaps, poses, Ks, imgs = [], [], [], [], []
    for pred in predictions[:use_views]:
        rgb = pred["img_no_norm"].squeeze(0)
        depth = pred["depth_along_ray"].squeeze(0)
        K = pred["intrinsics"].squeeze(0)
        pose = pred["camera_poses"].squeeze(0)
        H, W = rgb.shape[0], rgb.shape[1]

        rays = compute_plucker_rays(K, pose, H, W)
        rgb_ch = rgb.permute(2, 0, 1)
        d_ch = depth.squeeze(-1).unsqueeze(0)
        inp = torch.cat([rgb_ch, d_ch, rays], dim=0).unsqueeze(0)

        imgs.append(rgb)
        fmaps.append(inp)          # [1, 10, H, W] raw input
        dmaps.append(depth.squeeze(-1).unsqueeze(0))
        poses.append(pose)
        Ks.append(K)

    inputs_t = torch.cat(fmaps, dim=0).to(device)        # [V, 10, H, W]
    dmaps_t = torch.stack(dmaps, dim=0)                   # [V, 1, H, W]
    poses_t = torch.stack(poses, dim=0).to(device)        # [V, 4, 4]
    Ks_t = torch.stack(Ks, dim=0).to(device)              # [V, 3, 3]
    print(f"  Input batch: {list(inputs_t.shape)}")

    # ── 4) Forward pass through all trainable modules ─────────────────
    print("\n--- Forward pass ---")

    # 4a) U-Net (trainable) — keep on device, no .cpu()
    unet = LightweightUNet(in_channels=10, out_channels=64).train().to(device)
    features = unet(inputs_t)  # [V, 64, H, W] on device
    features.retain_grad()     # allow checking .grad on non-leaf tensor
    print(f"  U-Net output: {list(features.shape)}")

    # 4b) Feature Projection — run on same device
    anchor_feats, vis = project_and_aggregate(
        anchors, features, dmaps_t.to(device), poses_t, Ks_t
    )
    print(f"  Anchor features: {list(anchor_feats.shape)}")

    # 4c) Gaussian Decoder (trainable)
    decoder = GaussianDecoder(in_dim=64).train().to(device)
    gs_out = decoder(anchor_feats, anchors)
    print(f"  Decoder output: delta_mu={list(gs_out['delta_mu'].shape)}")

    # 4d) Assemble + Render
    gaussians = assemble_gaussians(gs_out, anchors)
    print(f"  Gaussians: {gaussians['means'].shape[0]}")

    # Render one view as supervision target
    ri = 0
    gs_gpu = {k: v.to(device).clone().detach().requires_grad_(True)
              if k == "means" else v.to(device)
              for k, v in gaussians.items()}
    # Re-compute gaussians with differentiable means
    gs_gpu_all = assemble_gaussians(gs_out, anchors)
    rendered, rendered_depth, alpha = render_view(
        gs_gpu_all, poses_t[ri], Ks_t[ri], H, W,
    )
    print(f"  Rendered: {list(rendered.shape)}")

    # ── 5) Compute loss ──────────────────────────────────────────────
    print("\n--- Loss ---")
    gt_img = imgs[ri].to(device)
    gt_depth = dmaps[ri].to(device).squeeze(0).unsqueeze(-1)
    # Resize if needed
    if rendered.shape[:2] != gt_img.shape[:2]:
        rendered = torch.nn.functional.interpolate(
            rendered.permute(2, 0, 1).unsqueeze(0),
            size=(gt_img.shape[0], gt_img.shape[1]), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    losses = compute_loss(
        rendered, rendered_depth,
        gt_img.to(device), gt_depth.to(device),
        gaussians["opacities"].to(device), gaussians["scales"].to(device),
    )
    print(f"  Total loss: {losses['total'].item():.4f}")

    # ── 6) Backward ───────────────────────────────────────────────────
    print("\n--- Backward ---")
    losses["total"].backward()

    # ── 7) Check gradients ────────────────────────────────────────────
    print("\n--- Gradient Check ---")
    all_ok = True
    all_ok &= check_grad("U-Net", unet)
    all_ok &= check_grad("Gaussian Decoder", decoder)
    # features (U-Net output) should have grad through projection too
    feat_has_grad = features.grad is not None and features.grad.abs().sum() > 0
    print(f"  U-Net output (features): has_grad={feat_has_grad}  "
          f"max_grad={features.grad.abs().max().item():.6f}" if feat_has_grad else f"  U-Net output (features): has_grad=False")
    all_ok &= feat_has_grad

    print(f"\n{'='*60}")
    print(f"  OVERALL: {'ALL GRADIENTS FLOW ✓' if all_ok else 'GRADIENT BROKEN ✗'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
