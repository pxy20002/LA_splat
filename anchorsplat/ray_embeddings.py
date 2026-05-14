import torch


def compute_plucker_rays(
    intrinsics: torch.Tensor,
    camera_poses: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Compute 6-channel Plücker ray embeddings for every pixel of a view.

    Args:
        intrinsics:  [3, 3] pinhole camera intrinsics matrix K.
        camera_poses: [4, 4] cam2world matrix (OpenCV convention: +X right, +Y down, +Z forward).
        height, width: image resolution.

    Returns:
        rays: [6, H, W] tensor.
              Channels 0..2 = Plücker direction  (unit vector in world frame).
              Channels 3..5 = Plücker moment     (camera_center × direction).
    """
    device = intrinsics.device
    dtype = intrinsics.dtype

    # 1) Pixel grid in homogeneous coordinates  [3, H*W]
    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # [H, W] each

    ones = torch.ones_like(uu)
    pixels = torch.stack([uu, vv, ones], dim=0)    # [3, H, W]
    pixels_flat = pixels.reshape(3, -1)             # [3, H*W]

    # 2) Camera-frame ray directions:  d_cam = K^{-1} @ pixel
    K_inv = torch.linalg.inv(intrinsics)            # [3, 3]
    d_cam = K_inv @ pixels_flat                     # [3, H*W]
    d_cam = d_cam / d_cam.norm(dim=0, keepdim=True).clamp(min=1e-8)  # unit vectors

    # 3) World-frame direction:  d_world = R @ d_cam
    R = camera_poses[:3, :3]                        # [3, 3]
    d_world = R @ d_cam                             # [3, H*W]
    d_world = d_world / d_world.norm(dim=0, keepdim=True).clamp(min=1e-8)

    # 4) Camera center in world frame (translation of cam2world)
    C = camera_poses[:3, 3]                         # [3]

    # 5) Plücker moment = C × d_world
    moment = torch.cross(C.unsqueeze(1).expand_as(d_world), d_world, dim=0)  # [3, H*W]

    # 6) Reshape and concatenate
    d_world = d_world.reshape(3, height, width)
    moment = moment.reshape(3, height, width)

    rays = torch.cat([d_world, moment], dim=0)      # [6, H, W]

    return rays
