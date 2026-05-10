import argparse
from pathlib import Path

import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def _to_cpu_detached(predictions):
    cpu_predictions = []
    for pred in predictions:
        cpu_pred = {}
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                cpu_pred[key] = value.detach().cpu()
            else:
                cpu_pred[key] = value
        cpu_predictions.append(cpu_pred)
    return cpu_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run MapAnything inference and save outputs.")
    parser.add_argument(
        "--image_source",
        type=str,
        default="../anchorsplat/images_s/",
        help="Image folder path, or a comma-separated image path list.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=518,
        help="Input resolution set for load_images.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch size for load_images.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model raw predictions (CPU tensors) for stage-2 anchor fusion.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../anchorsplat/outputs/mapanything_predictions.pt",
        help="Output .pt path for saved predictions.",
    )
    parser.add_argument(
        "--apply_mask",
        type=bool,
        default=True,
        help="Apply non-ambiguous mask (default True). Set to False to keep all points.",
    )
    parser.add_argument(
        "--mask_edges",
        type=bool,
        default=True,
        help="Apply edge mask on top of non-ambiguous mask (default True).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Auto device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2) Init MapAnything model with pretrained weights
    print("Loading MapAnything model from HuggingFace...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.eval()

    # 3) Load images with DINOv2 normalization (required by MapAnything)
    image_source = args.image_source
    print(f"Loading images from: {image_source}")
    views = load_images(
        folder_or_list=image_source,
        resolution_set=args.resolution,
        norm_type="dinov2",
        patch_size=args.patch_size,
    )
    print(f"Loaded {len(views)} views")

    for view in views:
        view["img"] = view["img"].to(device)

    # 4) Inference (use infer() for built-in post-processing)
    print("Running MapAnything inference...")
    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=(device == "cuda"),
            amp_dtype="bf16",
            apply_mask=args.apply_mask,
            mask_edges=args.mask_edges,
        )

    print("Inference complete! Output keys available per view:")

    # 5) Print output shapes
    for i, pred in enumerate(predictions):
        print(f"\n--- View {i} ---")
        print(f"  3D Points world  (pts3d):          {pred['pts3d'].shape}")
        print(f"  Depth along ray  (depth_along_ray): {pred['depth_along_ray'].shape}")
        print(f"  Depth Z          (depth_z):         {pred['depth_z'].shape}")
        print(f"  Ray directions   (ray_directions):  {pred['ray_directions'].shape}")
        print(f"  Intrinsics       (intrinsics):      {pred['intrinsics'].shape}")
        print(f"  Camera poses     (camera_poses):    {pred['camera_poses'].shape}")
        print(f"  Cam translation  (cam_trans):       {pred['cam_trans'].shape}")
        print(f"  Cam quaternions  (cam_quats):       {pred['cam_quats'].shape}")
        print(f"  Confidence       (conf):            {pred['conf'].shape}")
        if "mask" in pred:
            print(f"  Mask             (mask):            {pred['mask'].shape}")
        else:
            print(f"  Mask             (mask):            N/A (mask disabled)")

    if args.save_predictions:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "predictions": _to_cpu_detached(predictions),
            "meta": {
                "model": "mapanything",
                "image_source": image_source,
                "resolution": args.resolution,
                "patch_size": args.patch_size,
            },
        }
        torch.save(payload, out_path)
        print(f"\nSaved predictions to: {out_path}")


if __name__ == "__main__":
    main()
