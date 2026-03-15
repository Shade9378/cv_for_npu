"""
train.py
--------
YOLOv8 training script with configurable parameters.
Defaults to yolov8n.pt if no weights are provided.

Usage:
    python train.py --data /path/to/data.yaml [options]
"""

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--weights",
        type=str,
        default="yolov8/yolov8n.pt",
        help="Path to model weights (.pt file). Defaults to yolov8n.pt",
    )


    # Data
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml",
    )
    data.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square)",
    )

    # Training
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    train.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size",
    )
    train.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Initial learning rate",
    )
    train.add_argument(
        "--lrf",
        type=float,
        default=None,
        help="Final learning rate as a fraction of lr0 (optional)",
    )



    # Output
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--project",
        type=str,
        default="yolov8/runs",
        help="Directory to save training runs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nTraining configuration")
    print(f"  Weights   : {args.weights}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  LR0       : {args.lr0}")
    print(f"  LRF       : {args.lrf if args.lrf is not None else 'not set'}")
    print(f"  Project   : {args.project}\n")

    model = YOLO(args.weights)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        pretrained=True,
        amp=False,
        project=args.project,
    )

    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()