"""
Pipeline Script
---------------
Steps:
  1. ArUco dataset generation (C++)
  2. Train / val / test split + data.yaml generation
  3. Model training (loops until user is satisfied)
  4. Optimizations (optional, commented out)
  5. ONNX export
  6. Conversion + post-optimization
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ARUCO_GENERATOR_BIN = "dataset_construction/batch_maker"
DATA_SPLIT_SCRIPT   = "yolov8/data_split.py"
MODEL_SCRIPT        = "yolov8/train.py"
# OPTIMIZATION_SCRIPT = "path/to/optimize.py"
ONNX_EXPORT_SCRIPT  = "yolov8/export.py"
# CONVERSION_SCRIPT   = "path/to/convert.py"

def run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP : {step_name}")
    print(f"  CMD  : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] '{step_name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"[OK] '{step_name}' completed successfully.")


def get_latest_weights(project: str, fallback: str) -> str:
    """Find best.pt from the most recent training run, fallback to input weights if not found."""
    candidates = glob.glob(f"{project}/train*/weights/best.pt")
    if not candidates:
        print(f"[WARN] No trained weights found in {project}, falling back to: {fallback}")
        return fallback
    latest = max(candidates, key=os.path.getmtime)
    print(f"[OK] Using trained weights: {latest}")
    return latest

def step_aruco_generate(args: argparse.Namespace) -> None:
    """Step 1 — C++ ArUco dataset generator.

    Usage: ./batch_maker <input_folder> <overlay_folder> <output_name>
    Runs from inside args.output so the C++ binary writes dataset/ there.
    """
    # Ensure output dir exists before running
    Path(args.output).mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.output) / args.output_name
    if dataset_path.exists():
        print(f"[SKIP] Dataset already exists at: {dataset_path}")
        return

    print(f"{'='*60}")
    print(f"  STEP : ArUco Dataset Generation (C++)")
    print(f"  CMD  : {ARUCO_GENERATOR_BIN} {args.input_folder} {args.overlay_folder} {args.output_name}")
    print(f"  CWD  : {args.output}")
    print(f"{'='*60}")

    result = subprocess.run(
        [
            str(Path.cwd() / ARUCO_GENERATOR_BIN),
            args.input_folder,
            args.overlay_folder,
            args.output_name,
        ],
        cwd=args.output,
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] 'ArUco Dataset Generation (C++)' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"[OK] 'ArUco Dataset Generation (C++)' completed successfully.")


def step_data_split(args: argparse.Namespace) -> str:
    """Step 2 — split generated dataset into train / val / test and generate data.yaml."""
    split_path = Path(args.output) / "dataset_split"
    if split_path.exists():
        print(f"[SKIP] Dataset split already exists at: {split_path}")
        return str(split_path)

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"[ERROR] Split ratios must sum to 1.0 (got {total:.4f}).")
        sys.exit(1)

    split_output  = str(Path(args.output) / "dataset_split")
    class_map     = str(Path(args.output) / args.output_name / "overlay_index_map.txt")

    if not Path(class_map).exists():
        print(f"[ERROR] overlay_index_map.txt not found at: {class_map}")
        print(f"        Make sure step 1 (ArUco generation) completed successfully.")
        sys.exit(1)

    run(
        [
            sys.executable, DATA_SPLIT_SCRIPT,
            str(Path(args.output) / args.output_name), 
            "--val",       str(args.val_ratio * 100),
            "--test",      str(args.test_ratio * 100),
            "--out",       split_output,
            "--class-map", class_map,
        ],
        step_name="Train / Val / Test Split + data.yaml",
    )

    yaml_path = Path(split_output) / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml was not created at: {yaml_path}")
        sys.exit(1)

    print(f"[OK] data.yaml located at: {yaml_path}")
    return split_output


def step_run_model(args: argparse.Namespace) -> None:
    """Step 3 — single training run."""
    data_yaml = str(Path(args.split_output) / "data.yaml")
    cmd = [
        sys.executable, MODEL_SCRIPT,
        "--data",    data_yaml,
        "--weights", str(args.weights),
        "--epochs",  str(args.epochs),
        "--batch",   str(args.batch),
        "--lr0",     str(args.lr0),
        "--imgsz",   str(args.imgsz),
        "--project", str(args.output),
    ]
    if args.lrf is not None:
        cmd += ["--lrf", str(args.lrf)]

    run(cmd, step_name="Model Training")


def training_loop(args: argparse.Namespace) -> None:
    """Step 3 — train and loop until user confirms results are satisfactory.
    
    After each failed attempt, the latest trained weights are used as the
    starting point for the next run.
    """
    attempt = 1
    while True:
        print(f"\n  Training attempt #{attempt}")
        step_run_model(args)

        # After each run, update weights to the latest trained result
        args.weights = get_latest_weights(args.output, args.weights)

        print("\n" + "="*60)
        answer = input("  Are you satisfied with the results? [y/n]: ").strip().lower()
        if answer == "y":
            print("[OK] Training accepted. Continuing pipeline.")
            break

        print("  Looping back to training. Adjust your parameters if needed (Enter to keep current).")

        def prompt(name, current, cast):
            val = input(f"    {name} [{current}]: ").strip()
            return cast(val) if val else current

        args.epochs  = prompt("epochs",  args.epochs,  int)
        args.batch   = prompt("batch",   args.batch,   int)
        args.lr0     = prompt("lr0",     args.lr0,     float)
        args.imgsz   = prompt("imgsz",   args.imgsz,   int)
        lrf_input    = input(f"    lrf [{args.lrf if args.lrf is not None else 'not set'}]: ").strip()
        if lrf_input:
            args.lrf = float(lrf_input)

        attempt += 1


def step_run_optimizations(args: argparse.Namespace) -> None:
    """Step 4 — run optimizations (uncomment in main when ready)."""
    run(
        [
            sys.executable, OPTIMIZATION_SCRIPT,
            "--model-dir", args.output,
        ],
        step_name="Run Optimizations",
    )


def step_export_onnx(args: argparse.Namespace) -> None:
    """Step 5 — ONNX export using latest trained weights."""
    run(
        [
            sys.executable, ONNX_EXPORT_SCRIPT,
            "--weights", args.trained_weights,
            "--output",  args.output,
        ],
        step_name="ONNX Export",
    )


def step_convert(args: argparse.Namespace) -> None:
    """Step 6 — conversion + post-optimization."""
    run(
        [
            sys.executable, CONVERSION_SCRIPT,
            "--input-dir",  args.output,
            "--output-dir", args.output,
        ],
        step_name="Conversion + Post-optimization",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArUco dataset -> split -> train -> ONNX export -> convert pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data generation
    gen = parser.add_argument_group("Synthetic Data Generation")
    gen.add_argument(
        "--input-folder",
        required=True,
        help="Path to the input folder (initial dataset)",
    )
    gen.add_argument(
        "--overlay-folder",
        required=True,
        help="Path to the overlay folder",
    )

    # Data split
    split = parser.add_argument_group("Data Split")
    split.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of data used for training",
    )
    split.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data used for validation",
    )
    split.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data used for testing",
    )

    # Model training
    train = parser.add_argument_group("Model Training")
    train.add_argument(
        "--weights",
        type=str,
        default="yolov8/yolov8n.pt",
        help="Path to model weights (.pt). Defaults to yolov8n.pt",
    )
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
        help="Training batch size",
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
        help="Final LR as a fraction of lr0 (optional)",
    )
    train.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square)",
    )

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output",
        required=True,
        help="Working directory for generated dataset and model outputs",
    )

    return parser.parse_args()


# Entry point

def main() -> None:
    args = parse_args()

    # Resolve paths and hardcode output_name
    args.input_folder   = str(Path(args.input_folder).resolve())
    args.overlay_folder = str(Path(args.overlay_folder).resolve())
    args.output         = str(Path(args.output).resolve())
    args.output_name      = "dataset"

    print("\nPipeline starting")
    print(f"  Input folder  : {args.input_folder}")
    print(f"  Overlay folder: {args.overlay_folder}")
    print(f"  Working dir   : {args.output}")
    print(f"  Split         : train={args.train_ratio} / val={args.val_ratio} / test={args.test_ratio}")
    print(f"  Weights       : {args.weights}")
    print(f"  Training      : epochs={args.epochs}, batch={args.batch}, lr0={args.lr0}, lrf={args.lrf}, imgsz={args.imgsz}")

    step_aruco_generate(args)
    args.split_output = step_data_split(args)
    training_loop(args)
    args.trained_weights = get_latest_weights(args.output, args.weights)
    # step_run_optimizations(args)
    step_export_onnx(args)
    # step_convert(args)

    print(f"\n{'='*60}")
    print("  Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()