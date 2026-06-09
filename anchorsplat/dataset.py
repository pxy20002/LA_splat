"""Scene dataset: loads predictions from images (MapAnything) or cached .pt, precomputes anchors."""

from pathlib import Path
from typing import List, Optional, Union

import torch

from anchorsplat.anchor_predictor import AnchorPredictor
from anchorsplat.ray_embeddings import compute_plucker_rays


def _build_unet_inputs_for_views(predictions: list) -> list:
    """Build [10, H, W] U-Net input tensors for a list of prediction dicts."""
    inputs = []
    for pred in predictions:
        rgb = pred["img_no_norm"]              # [1, H, W, 3]
        depth = pred["depth_along_ray"]        # [1, H, W, 1]
        K = pred["intrinsics"].squeeze(0)       # [3, 3]
        pose = pred["camera_poses"].squeeze(0)  # [4, 4]

        H, W = rgb.shape[1], rgb.shape[2]
        rays = compute_plucker_rays(K, pose, H, W)  # [6, H, W]

        rgb_ch = rgb.squeeze(0).permute(2, 0, 1)     # [3, H, W]
        d_ch = depth.squeeze(0).squeeze(-1).unsqueeze(0)  # [1, H, W]
        inp = torch.cat([rgb_ch, d_ch, rays], dim=0) # [10, H, W]
        inputs.append(inp)
    return inputs


def _predictions_from_images(image_dir: str, device: str) -> list:
    """Run MapAnything on a folder of images. Cache result as .pt for later reuse."""
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    print(f"Running MapAnything on: {image_dir}")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    views = load_images(image_dir)
    n_images = len(views)
    print(f"  Loaded {n_images} images")

    # Check for cache with image count in filename
    cache_path = Path(image_dir) / f"mapanything_v{n_images}.pt"
    if cache_path.exists():
        print(f"Loading cached: {cache_path}")
        return _predictions_from_pt(str(cache_path))

    for v in views:
        v["img"] = v["img"].to(device)

    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=(device == "cuda"),
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    print(f"  MapAnything done: {len(predictions)} views")

    # Save cache
    torch.save({"predictions": predictions}, cache_path)
    print(f"  Cached → {cache_path}")
    return predictions


def _predictions_from_pt(pt_path: str) -> list:
    """Load cached predictions from a .pt file."""
    print(f"Loading cached: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data["predictions"]


class SceneDataset:
    """
    Wraps a scene (from images or cached .pt) as a fixed training dataset.

    All views are used every step — anchors and training data come from
    the same source, avoiding the anchor/view inconsistency issue.
    """

    def __init__(
        self,
        image_dir: str = None,
        pt_path: str = None,
        num_anchors: int = 1024,
        device: str = "cuda",
    ):
        self.device = device
        self.num_anchors = num_anchors

        if image_dir is not None:
            predictions = _predictions_from_images(image_dir, device)
        elif pt_path is not None:
            predictions = _predictions_from_pt(pt_path)
        else:
            raise ValueError("Must provide --image_dir or --pt_path")

        n_views = len(predictions)
        print(f"  Views: {n_views}")

        # Precompute anchors (CPU, frozen) — from ALL views
        print(f"  Precomputing anchors ({num_anchors})...")
        result = AnchorPredictor.from_predictions(
            predictions, num_anchors=num_anchors
        )
        anchors = result["anchors"]  # [N, 3]

        # Precompute U-Net inputs
        print(f"  Building U-Net inputs...")
        unet_in = _build_unet_inputs_for_views(predictions)
        unet_inputs_t = torch.stack(unet_in, dim=0)  # [V, 10, H, W]

        # Extract per-view data
        imgs, depths, poses, Ks = [], [], [], []
        for pred in predictions:
            imgs.append(pred["img_no_norm"].squeeze(0))              # [H, W, 3]
            depths.append(pred["depth_along_ray"].squeeze(0))       # [H, W, 1]
            poses.append(pred["camera_poses"].squeeze(0))             # [4, 4]
            Ks.append(pred["intrinsics"].squeeze(0))                 # [3, 3]

        H, W = imgs[0].shape[0], imgs[0].shape[1]
        self.n_views = n_views

        self._scene = {
            "anchors": anchors,
            "unet_inputs": unet_inputs_t,    # [V, 10, H, W]
            "imgs": imgs,                     # list of [H, W, 3]
            "depths": depths,                # list of [H, W, 1]
            "poses": poses,                   # list of [4, 4]
            "Ks": Ks,                        # list of [3, 3]
            "n_views": n_views,
            "H": H, "W": W,
        }

    def get_batch(self, device: str = None) -> dict:
        """Return the full training batch (all views, every step)."""
        if device is None:
            device = self.device
        s = self._scene
        V, H, W = s["n_views"], s["H"], s["W"]

        depths = torch.stack(s["depths"], dim=0).squeeze(-1).unsqueeze(1).to(device)
        poses = torch.stack(s["poses"], dim=0).to(device)
        Ks = torch.stack(s["Ks"], dim=0).to(device)
        imgs = torch.stack(s["imgs"], dim=0).to(device)

        return {
            "u_input": s["unet_inputs"].to(device),  # [V, 10, H, W]
            "depths": depths,                         # [V, 1, H, W]
            "poses": poses,                           # [V, 4, 4]
            "Ks": Ks,                                # [V, 3, 3]
            "gt_imgs": imgs,                          # [V, H, W, 3]
            "anchors": s["anchors"].to(device),       # [N, 3]
            "H": H, "W": W,
        }
