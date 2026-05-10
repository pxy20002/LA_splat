import argparse
from pathlib import Path

import numpy as np
import torch


try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError("open3d is required. Install with: pip install open3d") from exc


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _normalize_pts_shape(pts3d: torch.Tensor) -> torch.Tensor:
    if pts3d.ndim == 3 and pts3d.shape[-1] == 3:
        return pts3d.unsqueeze(0)
    if pts3d.ndim == 4 and pts3d.shape[-1] == 3:
        return pts3d
    raise ValueError(f"Unexpected pts3d shape: {tuple(pts3d.shape)}")


def _normalize_scalar_map_shape(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.ndim != 3:
        raise ValueError(f"Unexpected {name} shape: {tuple(value.shape)}")
    if value.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch for {name}, expected {batch_size}, got {value.shape[0]}"
        )
    return value


def load_dense_from_predictions(pred_path: Path, conf_threshold: float, min_depth: float, max_depth: float):
    payload = torch.load(pred_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "predictions" in payload:
        predictions = payload["predictions"]
    elif isinstance(payload, list):
        predictions = payload
    else:
        raise TypeError("Unsupported prediction file format.")

    all_points = []
    all_scores = []
    for pred in predictions:
        pts3d = _normalize_pts_shape(_as_tensor(pred["pts3d"]).float())
        batch_size = pts3d.shape[0]

        conf = pred.get("conf", None)
        if conf is not None:
            conf = _normalize_scalar_map_shape(_as_tensor(conf).float(), batch_size, "conf")

        depth = pred.get("depth_along_ray", None)
        if depth is not None:
            depth = _normalize_scalar_map_shape(
                _as_tensor(depth).float(), batch_size, "depth_along_ray"
            )

        for batch_idx in range(batch_size):
            points = pts3d[batch_idx]
            valid = torch.isfinite(points).all(dim=-1)

            if depth is not None:
                curr_depth = depth[batch_idx]
                valid = valid & torch.isfinite(curr_depth) & (curr_depth > min_depth)
                if max_depth > 0:
                    valid = valid & (curr_depth < max_depth)

            if conf is not None:
                curr_conf = conf[batch_idx]
                valid = valid & torch.isfinite(curr_conf) & (curr_conf >= conf_threshold)
                scores = curr_conf
            else:
                scores = torch.ones_like(valid, dtype=torch.float32)

            points_flat = points.reshape(-1, 3)[valid.reshape(-1)]
            scores_flat = scores.reshape(-1)[valid.reshape(-1)].clamp_min(1e-8)

            all_points.append(points_flat)
            all_scores.append(scores_flat)

    if not all_points:
        raise RuntimeError("No valid dense points loaded.")

    points = torch.cat(all_points, dim=0).cpu().numpy()
    scores = torch.cat(all_scores, dim=0).cpu().numpy()
    return points, scores


def load_dense_from_npz(npz_path: Path):
    data = np.load(npz_path)
    if "points" not in data:
        raise KeyError("dense_fused.npz must contain key 'points'.")
    points = data["points"]
    if "scores" in data:
        scores = data["scores"]
    else:
        scores = np.ones((points.shape[0],), dtype=np.float32)
    return points, scores


def colorize_by_score(scores: np.ndarray):
    if scores.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max - s_min < 1e-8:
        t = np.zeros_like(scores, dtype=np.float32)
    else:
        t = (scores - s_min) / (s_max - s_min)
    colors = np.stack([t, np.zeros_like(t), 1.0 - t], axis=-1).astype(np.float32)
    return colors


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dense 3D point cloud.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input file path. Supports vggt_predictions.pt or dense_fused.npz.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Only used for .pt input.",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=1e-6,
        help="Only used for .pt input.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=-1.0,
        help="Only used for .pt input. <=0 means disabled.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=2500000,
        help="Randomly keep at most this many points for visualization speed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if input_path.suffix == ".pt":
        points, scores = load_dense_from_predictions(
            pred_path=input_path,
            conf_threshold=args.conf_threshold,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
    elif input_path.suffix == ".npz":
        points, scores = load_dense_from_npz(input_path)
    else:
        raise ValueError("Unsupported input format. Use .pt or .npz.")

    if args.max_points > 0 and points.shape[0] > args.max_points:
        idx = np.random.choice(points.shape[0], size=args.max_points, replace=False)
        points = points[idx]
        scores = scores[idx]

    colors = colorize_by_score(scores)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print(f"Visualizing points: {points.shape[0]}")
    o3d.visualization.draw_geometries([pcd], window_name="Dense Point Cloud")


if __name__ == "__main__":
    main()
