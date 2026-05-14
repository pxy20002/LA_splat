import torch
import torch.nn.functional as F


def project_and_aggregate(
    anchors: torch.Tensor,
    feature_maps: torch.Tensor,
    depth_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_thresh_rel: float = 0.02,
):
    """
    Project 3D anchors to all views simultaneously (vectorized), sample 2D
    features, and aggregate with visibility-aware average pooling.

    Args:
        anchors:         [N, 3]    world-frame 3D anchor positions.
        feature_maps:    [V, C, H, W]  2D feature maps from U-Net.
        depth_maps:      [V, 1, H, W]  predicted depth (along ray) per view.
        camera_poses:    [V, 4, 4]  cam2world matrices.
        intrinsics:      [V, 3, 3]  pinhole intrinsics.
        depth_thresh_rel: float     relative depth-consistency threshold.

    Returns:
        anchor_features: [N, C]     aggregated anchor features.
        visibility:      [V, N]     bool mask of which views see each anchor.
    """
    V, C, H, W = feature_maps.shape
    N = anchors.shape[0]
    device = anchors.device
    dtype = feature_maps.dtype

    # ── 1) World → Camera (batch) ─────────────────────────────────────
    w2c = torch.linalg.inv(camera_poses.to(dtype))          # [V, 4, 4]
    R = w2c[:, :3, :3]                                       # [V, 3, 3]
    t = w2c[:, :3, 3]                                        # [V, 3]
    pts_cam = anchors.unsqueeze(0) @ R.transpose(-1, -2) + t.unsqueeze(1)  # [V, N, 3]

    # ── 2) Camera → Pixel (batch) ─────────────────────────────────────
    pixels = pts_cam @ intrinsics.to(dtype).transpose(-1, -2)  # [V, N, 3]
    z_cam = pixels[..., 2].clamp(min=1e-8)                     # [V, N]
    uv = pixels[..., :2] / z_cam.unsqueeze(-1)                 # [V, N, 2]

    # ── 3) Visibility mask ────────────────────────────────────────────
    vis = (z_cam > 0)                                          # in front
    vis &= (uv[..., 0] >= 0) & (uv[..., 0] < W - 1)           # inside W
    vis &= (uv[..., 1] >= 0) & (uv[..., 1] < H - 1)           # inside H

    # ── 4) Depth consistency ──────────────────────────────────────────
    uv_norm = uv.clone()
    uv_norm[..., 0] = 2.0 * uv_norm[..., 0] / (W - 1) - 1.0
    uv_norm[..., 1] = 2.0 * uv_norm[..., 1] / (H - 1) - 1.0
    uv_norm_grid = uv_norm.unsqueeze(2)                        # [V, N, 1, 2]

    d_sampled = F.grid_sample(
        depth_maps.to(dtype), uv_norm_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )                                                          # [V, 1, N, 1]
    d_sampled = d_sampled.squeeze(1).squeeze(-1)               # [V, N]

    ray_dist = pts_cam.norm(dim=-1)                            # [V, N]
    # Only check depth where vis is already True
    depth_ok = (d_sampled - ray_dist).abs() <= depth_thresh_rel * ray_dist.abs()
    vis = vis & depth_ok

    # ── 5) Feature sampling ───────────────────────────────────────────
    # grid_sample works on all V views at once
    sampled = F.grid_sample(
        feature_maps.to(dtype), uv_norm_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )                                                          # [V, C, N, 1]
    sampled = sampled.squeeze(-1).permute(0, 2, 1)             # [V, N, C]

    # Zero out invisible anchor samples
    sampled = sampled * vis.unsqueeze(-1).to(dtype)            # [V, N, C]

    # ── 6) Average pooling ────────────────────────────────────────────
    count = vis.sum(dim=0).clamp(min=1).to(dtype)              # [N]
    anchor_features = sampled.sum(dim=0) / count.unsqueeze(-1) # [N, C]

    return anchor_features, vis
