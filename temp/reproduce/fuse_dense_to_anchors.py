import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def load_predictions(pred_path: Path):
    payload = torch.load(pred_path, map_location="cpu", weights_only=False)

    if isinstance(payload, list):
        return payload, {}

    if isinstance(payload, dict):
        if "predictions" in payload:
            return payload["predictions"], payload.get("meta", {})
        return [payload], {}

    raise TypeError("Prediction file format is unsupported.")


def fuse_dense_points(
    predictions,
    conf_threshold: float,
    min_depth: float,
    max_depth: float,
    topk_per_view: int,
):
    all_points = []
    all_weights = []
    per_view_stats = []

    for view_idx, pred in enumerate(predictions):
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
                weights = curr_conf
            else:
                weights = torch.ones_like(valid, dtype=torch.float32)

            flat_valid = valid.reshape(-1)
            points_flat = points.reshape(-1, 3)[flat_valid]
            weights_flat = weights.reshape(-1)[flat_valid].clamp_min(1e-8)

            num_before_topk = points_flat.shape[0]

            if topk_per_view > 0 and points_flat.shape[0] > topk_per_view:
                _, topk_idx = torch.topk(weights_flat, k=topk_per_view, largest=True)
                points_flat = points_flat[topk_idx]
                weights_flat = weights_flat[topk_idx]

            all_points.append(points_flat)
            all_weights.append(weights_flat)
            per_view_stats.append(
                {
                    "view_idx": view_idx,
                    "batch_idx": batch_idx,
                    "kept_points": int(points_flat.shape[0]),
                    "before_topk": int(num_before_topk),
                }
            )

    if not all_points:
        raise RuntimeError("No points collected from predictions.")

    dense_points = torch.cat(all_points, dim=0)
    dense_weights = torch.cat(all_weights, dim=0)

    return dense_points, dense_weights, per_view_stats


def voxel_downsample_to_anchors(
    points: torch.Tensor,
    weights: torch.Tensor,
    voxel_size: float,
    min_points_per_voxel: int,
):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    if points.numel() == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, 3), dtype=torch.int64),
        )

    min_xyz = points.min(dim=0).values
    voxel_coords = torch.floor((points - min_xyz) / voxel_size).to(torch.int64)

    unique_voxels, inverse, counts = torch.unique(
        voxel_coords, dim=0, return_inverse=True, return_counts=True
    )

    num_voxels = unique_voxels.shape[0]
    weighted_sum = torch.zeros((num_voxels, 3), dtype=points.dtype)
    weight_sum = torch.zeros((num_voxels,), dtype=weights.dtype)

    weighted_sum.index_add_(0, inverse, points * weights.unsqueeze(-1))
    weight_sum.index_add_(0, inverse, weights)

    anchors = weighted_sum / weight_sum.unsqueeze(-1).clamp_min(1e-8)
    anchor_scores = weight_sum / counts.to(weight_sum.dtype).clamp_min(1.0)

    keep = counts >= min_points_per_voxel
    return anchors[keep], anchor_scores[keep], counts[keep], unique_voxels[keep]


def enforce_fixed_anchor_count(
    anchors: torch.Tensor,
    anchor_scores: torch.Tensor,
    counts: torch.Tensor,
    voxel_coords: torch.Tensor,
    dense_points: torch.Tensor,
    dense_weights: torch.Tensor,
    num_anchors: int,
):
    if num_anchors <= 0:
        return anchors, anchor_scores, counts, voxel_coords

    if anchors.shape[0] >= num_anchors:
        _, top_idx = torch.topk(anchor_scores, k=num_anchors, largest=True)
        return (
            anchors[top_idx],
            anchor_scores[top_idx],
            counts[top_idx],
            voxel_coords[top_idx],
        )

    num_need = num_anchors - anchors.shape[0]
    if dense_points.shape[0] == 0:
        return anchors, anchor_scores, counts, voxel_coords

    if dense_points.shape[0] >= num_need:
        _, dense_top_idx = torch.topk(dense_weights, k=num_need, largest=True)
        add_points = dense_points[dense_top_idx]
        add_scores = dense_weights[dense_top_idx]
    else:
        _, dense_sorted_idx = torch.topk(
            dense_weights, k=dense_points.shape[0], largest=True
        )
        add_points = dense_points[dense_sorted_idx]
        add_scores = dense_weights[dense_sorted_idx]
        repeat_n = num_need - add_points.shape[0]
        if add_points.shape[0] > 0:
            add_points = torch.cat([add_points, add_points[:1].repeat(repeat_n, 1)], dim=0)
            add_scores = torch.cat([add_scores, add_scores[:1].repeat(repeat_n)], dim=0)

    add_counts = torch.ones((add_points.shape[0],), dtype=counts.dtype)
    add_voxel_coords = torch.full(
        (add_points.shape[0], 3), -1, dtype=voxel_coords.dtype
    )

    anchors = torch.cat([anchors, add_points], dim=0)
    anchor_scores = torch.cat([anchor_scores, add_scores], dim=0)
    counts = torch.cat([counts, add_counts], dim=0)
    voxel_coords = torch.cat([voxel_coords, add_voxel_coords], dim=0)

    if anchors.shape[0] > num_anchors:
        _, top_idx = torch.topk(anchor_scores, k=num_anchors, largest=True)
        anchors = anchors[top_idx]
        anchor_scores = anchor_scores[top_idx]
        counts = counts[top_idx]
        voxel_coords = voxel_coords[top_idx]

    return anchors, anchor_scores, counts, voxel_coords


def save_ascii_ply(path: Path, points: np.ndarray, scores: np.ndarray, counts: np.ndarray):
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float confidence\n")
        file.write("property int observations\n")
        file.write("end_header\n")
        for point, score, count in zip(points, scores, counts):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {float(score):.6f} {int(count)}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse dense multi-view points and downsample to sparse 3D anchors."
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to stage-1 prediction .pt file (from reproduce/run_vggt.py --save_predictions).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reproduce_outputs/anchors",
        help="Output directory.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Minimum confidence for keeping dense points.",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=1e-6,
        help="Minimum valid depth_along_ray.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=-1.0,
        help="Maximum valid depth_along_ray. <=0 means disabled.",
    )
    parser.add_argument(
        "--topk_per_view",
        type=int,
        default=0,
        help="Keep top-K points by confidence per view; <=0 means disabled.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for anchor downsampling (same unit as predicted scene scale).",
    )
    parser.add_argument(
        "--min_points_per_voxel",
        type=int,
        default=1,
        help="Keep voxel only if it contains at least this many points.",
    )
    parser.add_argument(
        "--max_anchors",
        type=int,
        default=0,
        help="If >0, keep top anchors by confidence.",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=0,
        help="If >0, force output to exactly this many anchors (e.g., 2048).",
    )
    parser.add_argument(
        "--save_dense_npz",
        action="store_true",
        help="Also save fused dense points and confidences.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pred_path = Path(args.pred_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, meta = load_predictions(pred_path)
    print(f"Loaded {len(predictions)} views from: {pred_path}")

    dense_points, dense_weights, per_view_stats = fuse_dense_points(
        predictions=predictions,
        conf_threshold=args.conf_threshold,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        topk_per_view=args.topk_per_view,
    )
    print(f"Fused dense points: {dense_points.shape[0]}")

    anchors, anchor_scores, counts, voxel_coords = voxel_downsample_to_anchors(
        points=dense_points,
        weights=dense_weights,
        voxel_size=args.voxel_size,
        min_points_per_voxel=args.min_points_per_voxel,
    )

    if args.max_anchors > 0 and anchors.shape[0] > args.max_anchors:
        _, top_idx = torch.topk(anchor_scores, k=args.max_anchors, largest=True)
        anchors = anchors[top_idx]
        anchor_scores = anchor_scores[top_idx]
        counts = counts[top_idx]
        voxel_coords = voxel_coords[top_idx]

    anchors, anchor_scores, counts, voxel_coords = enforce_fixed_anchor_count(
        anchors=anchors,
        anchor_scores=anchor_scores,
        counts=counts,
        voxel_coords=voxel_coords,
        dense_points=dense_points,
        dense_weights=dense_weights,
        num_anchors=args.num_anchors,
    )

    anchors_np = anchors.cpu().numpy()
    anchor_scores_np = anchor_scores.cpu().numpy()
    counts_np = counts.cpu().numpy()
    voxel_coords_np = voxel_coords.cpu().numpy()

    np.savez_compressed(
        output_dir / "anchors.npz",
        anchors=anchors_np,
        scores=anchor_scores_np,
        counts=counts_np,
        voxel_coords=voxel_coords_np,
        voxel_size=np.array([args.voxel_size], dtype=np.float32),
    )
    save_ascii_ply(output_dir / "anchors.ply", anchors_np, anchor_scores_np, counts_np)

    if args.save_dense_npz:
        np.savez_compressed(
            output_dir / "dense_fused.npz",
            points=dense_points.cpu().numpy(),
            scores=dense_weights.cpu().numpy(),
        )

    stats = {
        "input_prediction_path": str(pred_path),
        "num_views": len(predictions),
        "dense_points": int(dense_points.shape[0]),
        "anchors": int(anchors.shape[0]),
        "conf_threshold": args.conf_threshold,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "topk_per_view": args.topk_per_view,
        "voxel_size": args.voxel_size,
        "min_points_per_voxel": args.min_points_per_voxel,
        "max_anchors": args.max_anchors,
        "num_anchors": args.num_anchors,
        "meta": meta,
        "per_view_stats": per_view_stats,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print(f"Saved anchors npz: {output_dir / 'anchors.npz'}")
    print(f"Saved anchors ply: {output_dir / 'anchors.ply'}")
    print(f"Saved stats json: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
