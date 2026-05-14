"""Scene dataset: loads .pt predictions, precomputes anchors, builds training batches."""

from pathlib import Path
from typing import List, Optional

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


class SceneDataset:
    """
    Wraps one or more .pt prediction files as a training dataset.

    Precomputes anchors (frozen) and provides pre-built U-Net input tensors
    so the training loop doesn't need to repeat geometry computations.
    """

    def __init__(
        self,
        pt_paths: List[str],
        num_anchors: int = 1024,
        device: str = "cuda",
    ):
        self.device = device
        self.num_anchors = num_anchors

        # Per-scene storage
        self.scenes = []  # list of dicts

        for pt_path in pt_paths:
            print(f"Loading: {pt_path}")
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            predictions = data["predictions"]
            n_views = len(predictions)

            # Precompute anchors (CPU, frozen)
            print(f"  Precomputing anchors ({num_anchors}) for {n_views} views...")
            result = AnchorPredictor.from_predictions(
                predictions, num_anchors=num_anchors
            )
            anchors = result["anchors"]  # [N, 3]

            # Precompute U-Net inputs
            print(f"  Building U-Net inputs...")
            unet_inputs = _build_unet_inputs_for_views(predictions)
            # Stack into precomputed tensor [V, 10, H, W]
            unet_inputs_t = torch.stack(unet_inputs, dim=0)

            # Extract per-view data
            imgs = []
            depths = []
            poses = []
            Ks = []
            for pred in predictions:
                imgs.append(pred["img_no_norm"].squeeze(0))              # [H, W, 3]
                depths.append(pred["depth_along_ray"].squeeze(0))       # [H, W, 1]
                poses.append(pred["camera_poses"].squeeze(0))             # [4, 4]
                Ks.append(pred["intrinsics"].squeeze(0))                 # [3, 3]

            H, W = imgs[0].shape[0], imgs[0].shape[1]

            self.scenes.append({
                "anchors": anchors,
                "unet_inputs": unet_inputs_t,    # [V, 10, H, W]
                "imgs": imgs,                     # list of [H, W, 3]
                "depths": depths,                # list of [H, W, 1]
                "poses": poses,                   # list of [4, 4]
                "Ks": Ks,                        # list of [3, 3]
                "n_views": n_views,
                "H": H,
                "W": W,
            })

    def get_scene(self, scene_idx: int = 0) -> dict:
        """Return a scene dict with all views available for sampling."""
        return self.scenes[scene_idx]

    def sample_views(self, scene_dict: dict, num_views: int, device: str = None):
        """
        Randomly sample `num_views` from a scene and return a training batch.

        Returns a dict with keys:
          u_input:  [V, 10, H, W] on device
          depths:   [V, 1, H, W]  on device
          poses:    [V, 4, 4]     on device
          Ks:       [V, 3, 3]     on device
          gt_imgs:  [V, H, W, 3]  on device
          anchors:  [N, 3]        on device
          H, W:     int
        """
        if device is None:
            device = self.device

        n = scene_dict["n_views"]
        indices = torch.randperm(n)[:num_views].tolist()

        unet_in = scene_dict["unet_inputs"][indices].to(device)  # [V, 10, H, W]

        depths = torch.stack([scene_dict["depths"][i] for i in indices], dim=0)
        depths = depths.squeeze(-1).unsqueeze(1).to(device)  # [V, 1, H, W]

        poses = torch.stack([scene_dict["poses"][i] for i in indices], dim=0).to(device)
        Ks = torch.stack([scene_dict["Ks"][i] for i in indices], dim=0).to(device)
        imgs = torch.stack([scene_dict["imgs"][i] for i in indices], dim=0).to(device)

        return {
            "u_input": unet_in,
            "depths": depths,
            "poses": poses,
            "Ks": Ks,
            "gt_imgs": imgs,
            "anchors": scene_dict["anchors"].to(device),
            "H": scene_dict["H"],
            "W": scene_dict["W"],
            "indices": indices,
        }

    def __len__(self):
        return len(self.scenes)
