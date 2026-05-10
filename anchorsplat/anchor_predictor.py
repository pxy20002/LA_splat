import torch

#: Size threshold for computing quantiles on a random subset instead of the full tensor.
_QUANTILE_SUBSET_SIZE = 500_000
#: Maximum points to feed into FPS; random downsampling is applied above this.
_FPS_MAX_INPUT = 200_000


def _quantile_on_subset(t: torch.Tensor, q: float, max_n: int = _QUANTILE_SUBSET_SIZE) -> torch.Tensor:
    """Compute quantile on a random subset if ``t`` has more than ``max_n`` elements."""
    if t.numel() <= max_n:
        return torch.quantile(t, q)
    idx = torch.randperm(t.numel(), device=t.device)[:max_n]
    return torch.quantile(t[idx], q)


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Pure PyTorch FPS (Farthest Point Sampling).

    Args:
        points: [N, 3] point cloud in world frame.
        num_samples: number of points to sample.

    Returns:
        indices: [num_samples] indices into ``points`` of the sampled anchors.
    """
    N = points.shape[0]
    if N <= num_samples:
        return torch.arange(N, device=points.device)

    distances = torch.full((N,), torch.inf, device=points.device, dtype=points.dtype)
    indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)

    farthest = torch.randint(0, N, (1,), device=points.device).item()

    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest]
        dist_to_centroid = ((points - centroid) ** 2).sum(dim=1)
        distances = torch.minimum(distances, dist_to_centroid)
        farthest = torch.argmax(distances).item()

    return indices


def clip_point_cloud(points: torch.Tensor, percentile: tuple = (0.02, 0.98)):
    """
    Clip point cloud to remove MVS outliers.

    Points outside the percentile range on any axis are discarded.
    For large point clouds quantiles are estimated on a random subset.

    Args:
        points: [N, 3] point cloud.
        percentile: (low, high) quantile thresholds per axis.

    Returns:
        clipped: [M, 3] clipped point cloud.
        mask: [N] bool mask of retained points.
        info: dict with per-axis (low, high) bounds.
    """
    low_p, high_p = percentile

    x_low = _quantile_on_subset(points[:, 0], low_p)
    x_high = _quantile_on_subset(points[:, 0], high_p)
    y_low = _quantile_on_subset(points[:, 1], low_p)
    y_high = _quantile_on_subset(points[:, 1], high_p)
    z_low = _quantile_on_subset(points[:, 2], low_p)
    z_high = _quantile_on_subset(points[:, 2], high_p)

    mask = (
        (points[:, 0] >= x_low) & (points[:, 0] <= x_high)
        & (points[:, 1] >= y_low) & (points[:, 1] <= y_high)
        & (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    )

    info = {
        "x": (x_low.item(), x_high.item()),
        "y": (y_low.item(), y_high.item()),
        "z": (z_low.item(), z_high.item()),
        "kept": mask.sum().item(),
        "total": points.shape[0],
    }

    return points[mask], mask, info


class AnchorPredictor:
    """
    Generates sparse 3D anchors from multi-view MVS predictions.

    Pipeline:
      1. Extract valid 3D points (optionally filtered by MVS mask).
      2. Concatenate points across all views.
      3. 3D percentile-based clipping to remove outliers.
      4. Farthest Point Sampling (FPS) to produce a fixed number of anchors.

    Args:
        num_anchors: number of output anchors (default 1024).
        clip_percentile: (low, high) quantile thresholds for 3D clipping.
    """

    def __init__(self, num_anchors: int = 1024, clip_percentile: tuple = (0.02, 0.98)):
        self.num_anchors = num_anchors
        self.clip_percentile = clip_percentile

    def sample_anchors(self, point_cloud: torch.Tensor) -> dict:
        """
        Clip → (optional downsample) → FPS a single combined point cloud into anchors.

        When the point cloud exceeds ``_FPS_MAX_INPUT``, it is randomly downsampled
        before FPS to keep runtime and memory under control.

        Args:
            point_cloud: [N, 3] combined points from all views (world frame).

        Returns:
            dict with keys:
              - anchors: [num_anchors, 3]
              - clip_info: dict with per-axis bounds and keep ratio.
              - fps_indices: [num_anchors] indices into the (possibly downsampled) point cloud.
        """
        if point_cloud.shape[0] == 0:
            return {
                "anchors": torch.empty((0, 3), device=point_cloud.device),
                "clip_info": {"x": (0, 0), "y": (0, 0), "z": (0, 0), "kept": 0, "total": 0},
                "fps_indices": torch.empty(0, dtype=torch.long),
            }

        clipped, _, clip_info = clip_point_cloud(point_cloud, self.clip_percentile)

        if clipped.shape[0] > _FPS_MAX_INPUT:
            idx = torch.randperm(clipped.shape[0], device=clipped.device)[:_FPS_MAX_INPUT]
            clipped = clipped[idx]

        fps_idx = farthest_point_sampling(clipped, self.num_anchors)
        anchors = clipped[fps_idx]

        return {
            "anchors": anchors,
            "clip_info": clip_info,
            "fps_indices": fps_idx,
        }

    @staticmethod
    def from_predictions(
        predictions: list,
        num_anchors: int = 1024,
        clip_percentile: tuple = (0.02, 0.98),
    ) -> dict:
        """
        Offline mode: generate anchors from pre-saved MVS inference results.

        Supports both MapAnything (with optional ``mask`` key) and VGGT (no mask).

        Args:
            predictions: list of per-view dicts from MVS inference.
            num_anchors: number of output anchors.
            clip_percentile: quantile thresholds for 3D clipping.

        Returns:
            dict with keys:
              - anchors: [num_anchors, 3]
              - clip_info: dict with per-axis bounds and keep ratio.
              - view_data: list of per-view tensors needed downstream.
              - pts3d_per_view: list of [M_i, 3] valid point tensors per view.
              - used_mask: whether the MVS mask was applied.
        """
        predictor = AnchorPredictor(num_anchors=num_anchors, clip_percentile=clip_percentile)

        # Detect whether all predictions carry a mask key.
        has_mask = all("mask" in pred for pred in predictions)

        all_points = []
        view_data = []
        pts3d_per_view = []

        for pred in predictions:
            pts3d = pred["pts3d"]  # [1, H, W, 3]
            H, W = pts3d.shape[1], pts3d.shape[2]

            pts_flat = pts3d.reshape(-1, 3)  # [H*W, 3]

            if has_mask:
                mask = pred["mask"]  # [1, H, W, 1]
                mask_flat = mask.reshape(-1)
                valid = pts_flat[mask_flat]
                if valid.shape[0] == 0:
                    # Mask discarded everything for this view — fall back to all points.
                    valid = pts_flat
            else:
                valid = pts_flat

            all_points.append(valid)
            pts3d_per_view.append(valid)

            view_data.append({
                "ray_directions": pred["ray_directions"],    # [1, H, W, 3]
                "depth_along_ray": pred["depth_along_ray"],  # [1, H, W, 1]
                "camera_poses": pred.get("camera_poses"),    # [1, 4, 4] or None for VGGT
                "intrinsics": pred.get("intrinsics"),        # [1, 3, 3] or None for VGGT
                "img_no_norm": pred.get("img_no_norm"),      # [1, H, W, 3]
                "mask": pred.get("mask"),                    # [1, H, W, 1] or None
                "valid_pts3d": valid,
            })

        combined = torch.cat(all_points, dim=0)

        result = predictor.sample_anchors(combined)
        result["view_data"] = view_data
        result["pts3d_per_view"] = pts3d_per_view
        result["used_mask"] = has_mask

        return result
