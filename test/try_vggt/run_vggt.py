import argparse
from pathlib import Path

import torch

from mapanything.models import init_model_from_config
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
    parser = argparse.ArgumentParser(description="Run VGGT inference and optionally save raw outputs.")
    parser.add_argument(
        "--image_source",
        type=str,
        default="./images_s/",
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
        default="reproduce_outputs/vggt_predictions.pt",
        help="Output .pt path for saved predictions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Auto device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2) Init VGGT model from configs/model/vggt.yaml
    print("Loading VGGT model...")
    model = init_model_from_config("vggt", device=device)
    model.eval()

    # 3) Load images
    image_source = args.image_source
    print(f"Loading images from: {image_source}")
    views = load_images(
        folder_or_list=image_source,
        resolution_set=args.resolution,
        norm_type="identity",
        patch_size=args.patch_size,
    )
    print(f"Loaded {len(views)} views")

    for view in views:
        view["img"] = view["img"].to(device)

    # 4) Inference
    print("Running VGGT inference...")
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                predictions = model(views)
        else:
            predictions = model(views)

    print("Inference complete! Output keys available per view:")

    # 5) Print output shapes
    for i, pred in enumerate(predictions):
        print(f"\n--- View {i} ---")
        print(f"3D Points (pts3d): {pred['pts3d'].shape}")
        print(f"Depth (depth_along_ray): {pred['depth_along_ray'].shape}")
        print(f"Ray Directions (ray_directions): {pred['ray_directions'].shape}")
        print(f"Camera Translation (cam_trans): {pred['cam_trans'].shape}")
        print(f"Camera Quaternions (cam_quats): {pred['cam_quats'].shape}")
        print(f"Confidence (conf): {pred['conf'].shape}")

    if args.save_predictions:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "predictions": _to_cpu_detached(predictions),
            "meta": {
                "model": "vggt",
                "image_source": image_source,
                "resolution": args.resolution,
                "patch_size": args.patch_size,
            },
        }
        torch.save(payload, out_path)
        print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()