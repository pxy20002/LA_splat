"""Minimal gsplat render test — isolate rendering from our pipeline."""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import gsplat


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  gsplat: {gsplat.__version__}")
    print(f"PyTorch: {torch.__version__}  |  CUDA arch: {torch.cuda.get_device_capability()}")

    N, H, W = 100, 256, 256

    # All Gaussians clearly in front of camera
    means = torch.randn(N, 3, device=device) * 0.1
    means[:, 2] = means[:, 2].abs() + 1.0  # Z in [1.0, 1.1]
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(torch.randn(N, 3, device=device) * 0.3) * 0.03
    opacities = torch.sigmoid(torch.randn(N, device=device)) * 0.5 + 0.25
    colors = torch.sigmoid(torch.randn(N, 3, device=device))
    K = torch.tensor([[300., 0, 128.], [0, 300, 128], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device)
    tile_size = 16
    tw, th = (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size

    print("\n--- Step 1: Projection ---")
    radii, m2d, depths, conics, comps = gsplat.fully_fused_projection(
        means=means, covars=None, quats=quats, scales=scales,
        viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
        width=W, height=H, packed=False,
    )
    print(f"  means2d: {m2d.shape}  conics: {conics.shape}  depths: {depths.shape}")
    print(f"  radii range: [{radii.min():.2f}, {radii.max():.2f}]")

    print("\n--- Step 2: Tile intersection ---")
    to, fl, nt = gsplat.isect_tiles(m2d, radii, depths, tile_size, tw, th, packed=False, n_images=1)
    io = gsplat.isect_offset_encode(to.long(), 1, tw, th).int()
    print(f"  tile_offsets: {to.shape}  flatten_ids: {fl.shape}")

    print("\n--- Step 3: RGB rasterization ---")
    r, alpha = gsplat.rasterize_to_pixels(
        m2d, conics,
        colors.unsqueeze(0), opacities.unsqueeze(0),
        W, H, tile_size,
        io, fl.int(),
        torch.ones(1, 3, device=device), packed=False,
    )
    print(f"  Rendered: {list(r.shape)}  range=[{r.min():.3f}, {r.max():.3f}]")

    print("\n--- Step 4: Depth rasterization ---")
    depth_colors = depths.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3).contiguous()
    rd, _ = gsplat.rasterize_to_pixels(
        m2d, conics,
        depth_colors.float(),
        opacities.unsqueeze(0),
        W, H, tile_size,
        io, fl.int(),
        torch.zeros(1, 3, device=device), packed=False,
    )
    print(f"  Depth: {list(rd.shape)}  range=[{rd.min():.3f}, {rd.max():.3f}]")

    print("\n--- Step 5: Gradient check ---")
    g = means.detach().clone().requires_grad_(True)
    q2 = quats.detach().clone()
    s2 = scales.detach().clone()
    radii2, m2d2, depths2, conics2, _ = gsplat.fully_fused_projection(
        means=g, covars=None, quats=q2, scales=s2,
        viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0), width=W, height=H, packed=False,
    )
    to2, fl2, _ = gsplat.isect_tiles(m2d2, radii2, depths2, tile_size, tw, th, packed=False, n_images=1)
    io2 = gsplat.isect_offset_encode(to2.long(), 1, tw, th).int()
    r2, _ = gsplat.rasterize_to_pixels(
        m2d2, conics2, colors.unsqueeze(0), opacities.unsqueeze(0),
        W, H, tile_size, io2, fl2.int(),
        torch.ones(1, 3, device=device), packed=False,
    )
    r2.mean().backward()
    print(f"  grad norm: {g.grad.norm():.4f}")

    print("\ngsplat OK! All 5 steps passed.")


if __name__ == "__main__":
    main()
