import argparse
from pathlib import Path

import numpy as np


try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError("open3d is required. Install with: pip install open3d") from exc


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
    parser = argparse.ArgumentParser(description="Visualize sparse anchor points.")
    parser.add_argument(
        "--anchors_path",
        type=str,
        default="reproduce_outputs/anchors/anchors.npz",
        help="Path to anchors.npz",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=-1.0,
        help="Keep anchors with score >= min_score. <0 means disabled.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Keep top max_points by score for visualization. <=0 means disabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    anchors_path = Path(args.anchors_path)

    data = np.load(anchors_path)
    if "anchors" not in data:
        raise KeyError("anchors.npz must contain key 'anchors'.")

    anchors = data["anchors"]
    scores = data["scores"] if "scores" in data else np.ones((anchors.shape[0],), dtype=np.float32)

    if args.min_score >= 0:
        keep = scores >= args.min_score
        anchors = anchors[keep]
        scores = scores[keep]

    if args.max_points > 0 and anchors.shape[0] > args.max_points:
        idx = np.argsort(scores)[-args.max_points :]
        anchors = anchors[idx]
        scores = scores[idx]

    colors = colorize_by_score(scores)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchors.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print(f"Visualizing anchors: {anchors.shape[0]}")
    o3d.visualization.draw_geometries([pcd], window_name="Sparse Anchors")


if __name__ == "__main__":
    main()
