"""Module 5: Differentiable 3DGS rendering via diff-gaussian-rasterization + composite loss."""

import math
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


# OpenCV (+Z forward, +Y down) → OpenGL (-Z forward, +Y up) conversion
_OPENCV_TO_OPENGL = torch.tensor([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0,  1.0, 0.0],   # Z: keep positive, 3DGS rasterizer expects this
    [0.0,  0.0,  0.0, 1.0],
])


def _quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] quaternion from xyzw (gsplat) to wxyz (diff-gaussian-rasterization)."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _projection_matrix_from_K(K: torch.Tensor, H: int, W: int,
                               znear: float = 0.01, zfar: float = 100.0):
    """
    Build an OpenGL-style projection matrix from pinhole intrinsics.

    Args:
        K:  [3, 3] intrinsics.
        H, W: image resolution.
        znear, zfar: near / far clipping planes.

    Returns:
        proj: [4, 4] projection matrix.
        tanfovx, tanfovy: half-FOV tangents.
    """
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()

    tanfovx = 0.5 * W / fx
    tanfovy = 0.5 * H / fy

    top = tanfovy * znear
    bottom = -top
    right = tanfovx * znear
    left = -right

    proj = torch.zeros(4, 4, device=K.device, dtype=K.dtype)
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0

    return proj, tanfovx, tanfovy


def assemble_gaussians(decoder_output: dict, anchors: torch.Tensor) -> dict:
    """
    Flatten per-anchor Gaussian attributes (N × 4) into per-Gaussian lists (4N).

    Returns dict with means, quats, scales, opacities, colors — each [4N, *].
    """
    N = anchors.shape[0]
    G = 4

    means = anchors.unsqueeze(1).expand(N, G, 3) + decoder_output["delta_mu"]
    means = means.reshape(N * G, 3)

    opacities = decoder_output["opacity"].reshape(N * G, 1)  # [4N, 1]
    scales = decoder_output["scale"].reshape(N * G, 3)        # [4N, 3]
    # Convert xyzw → wxyz
    quats = _quat_xyzw_to_wxyz(decoder_output["rotation"].reshape(N * G, 4))
    colors = decoder_output["sh"].reshape(N * G, 3)           # [4N, 3]

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


def render_view(
    gaussians: dict,
    camera_pose: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    background: torch.Tensor = None,
    scale_modifier: float = 1.0,
) -> tuple:
    """
    Render a single view via diff-gaussian-rasterization.

    Args:
        gaussians:    dict with means, quats, scales, opacities, colors.
        camera_pose:  [4, 4] cam2world (OpenCV convention: +X right, +Y down, +Z forward).
        K:            [3, 3] intrinsics.
        H, W:         output resolution.
        background:   [3] or None (white if None).
        scale_modifier: global scale factor.

    Returns:
        rendered:       [H, W, 3] RGB.
        rendered_depth: [H, W, 1] depth.
        alpha:          [H, W, 1] opacity.
    """
    device = gaussians["means"].device

    if background is None:
        background = torch.ones(3, device=device)
    elif background.dim() == 1:
        background = background.to(device)
    else:
        background = background.reshape(3).to(device)

    # Convert OpenCV cam2world → OpenGL world2cam
    c2w_gl = camera_pose.to(device) @ _OPENCV_TO_OPENGL.to(device)
    w2c = torch.linalg.inv(c2w_gl)  # [4, 4]

    cam_center = camera_pose[:3, 3].to(device)  # world-frame camera center

    proj, tanfovx, tanfovy = _projection_matrix_from_K(K.to(device), H, W)
    full_proj = proj @ w2c  # [4, 4]

    means = torch.nan_to_num(gaussians["means"].to(device), nan=0.0, posinf=1.0, neginf=0.0)
    quats = torch.nan_to_num(gaussians["quats"].to(device), nan=0.0, posinf=1.0, neginf=0.0)
    scales_g = torch.nan_to_num(gaussians["scales"].to(device), nan=0.0, posinf=1.0, neginf=0.0).clamp(min=1e-8, max=100.0)
    opacities_g = torch.nan_to_num(gaussians["opacities"].to(device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).squeeze(-1)  # [N]
    colors_g = torch.nan_to_num(gaussians["colors"].to(device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    zeros2d = torch.zeros(means.shape[0], 2, device=device)

    # --- RGB pass ---
    rgb_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tanfovx, tanfovy=tanfovy, bg=background,
        scale_modifier=scale_modifier,
        viewmatrix=w2c.transpose(0, 1), projmatrix=full_proj.transpose(0, 1),
        sh_degree=0, campos=cam_center, prefiltered=False, debug=False,
    )
    rendered, _ = GaussianRasterizer(rgb_settings)(
        means3D=means, means2D=zeros2d, shs=None,
        colors_precomp=colors_g, opacities=opacities_g,
        scales=scales_g, rotations=quats, cov3D_precomp=None,
    )
    rendered = rendered.permute(1, 2, 0)  # [H, W, 3]
    rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)

    # --- Alpha pass: white Gaussians on black → R channel = accumulated alpha ---
    white = torch.ones_like(colors_g)
    black = torch.zeros(3, device=device)
    alpha_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tanfovx, tanfovy=tanfovy, bg=black,
        scale_modifier=scale_modifier,
        viewmatrix=w2c.transpose(0, 1), projmatrix=full_proj.transpose(0, 1),
        sh_degree=0, campos=cam_center, prefiltered=False, debug=False,
    )
    alpha_raw, _ = GaussianRasterizer(alpha_settings)(
        means3D=means, means2D=zeros2d, shs=None,
        colors_precomp=white, opacities=opacities_g,
        scales=scales_g, rotations=quats, cov3D_precomp=None,
    )
    alpha = alpha_raw[0:1].permute(1, 2, 0)  # [H, W, 1]

    # --- Depth pass: depth as color on black → divide by alpha for clean depth ---
    depth_val = means[:, 2:3].expand(-1, 3)
    depth_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tanfovx, tanfovy=tanfovy, bg=black,
        scale_modifier=scale_modifier,
        viewmatrix=w2c.transpose(0, 1), projmatrix=full_proj.transpose(0, 1),
        sh_degree=0, campos=cam_center, prefiltered=False, debug=False,
    )
    depth_raw, _ = GaussianRasterizer(depth_settings)(
        means3D=means, means2D=zeros2d, shs=None,
        colors_precomp=depth_val, opacities=opacities_g,
        scales=scales_g, rotations=quats, cov3D_precomp=None,
    )
    depth = depth_raw[0:1].permute(1, 2, 0)  # [H, W, 1]
    depth = depth / alpha.clamp(min=1e-8)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    return rendered, depth, alpha


def compute_loss(
    rendered: torch.Tensor,
    rendered_depth: torch.Tensor,
    gt_image: torch.Tensor,
    gt_depth: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    weights: dict = None,
) -> dict:
    """
    Compute the composite AnchorSplat loss.

    L = λ_I·l_I  +  λ_D·l_D  +  λ_α·l_α  +  λ_s·l_s

    where l_I = L1 + γ_SSIM·(1-SSIM) + γ_LPIPS·LPIPS
    """
    if weights is None:
        weights = {
            "lambda_I": 200.0,
            "gamma_ssim": 0.2,
            "gamma_lpips": 0.2,
            "lambda_D": 100.0,
            "lambda_alpha": 0.1,
            "lambda_s": 1e4,
        }

    rendered = rendered.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
    gt = gt_image.permute(2, 0, 1).unsqueeze(0)          # [1, 3, H, W]

    l1 = F.l1_loss(rendered, gt)
    ssim_val = _ssim(rendered, gt)
    ssim_loss = 1.0 - ssim_val

    loss_I = l1 + weights["gamma_ssim"] * ssim_loss

    try:
        from lpips import LPIPS
        lpips_fn = LPIPS(net="vgg").to(rendered.device)
        lpips_val = lpips_fn(rendered.clamp(0, 1), gt.clamp(0, 1)).mean()
        loss_I = loss_I + weights["gamma_lpips"] * lpips_val
    except ImportError:
        lpips_val = torch.tensor(0.0, device=rendered.device)

    loss_D = F.l1_loss(rendered_depth, gt_depth)

    loss_alpha = (1.0 - opacities).mean()
    loss_scale = scales.prod(dim=1).mean()

    total = (
        weights["lambda_I"] * loss_I
        + weights["lambda_D"] * loss_D
        + weights["lambda_alpha"] * loss_alpha
        + weights["lambda_s"] * loss_scale
    )

    return {
        "total": total,
        "l1": l1.detach(),
        "ssim": ssim_loss.detach(),
        "lpips": lpips_val.detach() if isinstance(lpips_val, torch.Tensor) else lpips_val,
        "depth": loss_D.detach(),
        "opacity_reg": loss_alpha.detach(),
        "scale_reg": loss_scale.detach(),
    }


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM between two [1, 3, H, W] image batches."""
    C = img1.shape[1]
    sigma = 1.5
    gauss = torch.arange(window_size, device=img1.device, dtype=torch.float32)
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window_1d.unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
