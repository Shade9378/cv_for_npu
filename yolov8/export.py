"""
export.py
---------
YOLOv8 ONNX export script.

Usage:
    python export.py --weights /path/to/best.pt --output /path/to/export/dir
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 ONNX export script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the exported ONNX model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = str(Path(args.weights).resolve())
    output_dir   = str(Path(args.output).resolve())

    print(f"\nExporting model")
    print(f"  Weights : {weights_path}")
    print(f"  Output  : {output_dir}\n")

    model = YOLO(weights_path)
    path_to_onnx_file = model.export(
        format="onnx",
        project=output_dir,
    )

    print(f"\nModel successfully exported to:")
    print(f"  Path: {path_to_onnx_file}")


if __name__ == "__main__":
    main()