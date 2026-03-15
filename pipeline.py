# Pipeline:
# A script that accepts an initial dataset with aruco and dataset output location, put the generation path into the data.yaml file
# Run the model
# Run the optimizations
# Run the ONNX export
# Run the conversion (which also includes some optimization)

#what missing, arg to take in model options for finetuning
#the training model should loop until the training result is satisfactory, confirmed by the user, if not loop back and allows user to select a new parameter note: the pt file that need to be used after loop should be the weoght after trained

# list of all the args that user should be able to input:
# for C++ datagen: dataset location
# for split split percentage in float (0.7 0.15 0.15 ) 
# for model training, the parameters that you can adjust to tune
# for export, output location

# python pipeline.py \
#   --dataset   /path/to/aruco_data \
#   --output    /path/to/working_dir \
#   --export-output /path/to/final_model \
#   --epochs    150 \
#   --batch-size 32 \
#   --lr        0.0005 \
#   --imgsz     640 \
#   --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

"""
Pipeline Script
---------------
Steps:
  1. ArUco dataset generation (C++)
  2. Train / val / test split
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


# ── Configuration ─────────────────────────────────────────────────────────────

ARUCO_GENERATOR_BIN = "dataset_construction/batch_maker"
DATA_SPLIT_SCRIPT   = "yolov8/data_split_detection.py"
MODEL_SCRIPT        = "yolov8/ultralytics/train_detection.py"
# OPTIMIZATION_SCRIPT = "path/to/optimize.py"            # uncomment when ready
ONNX_EXPORT_SCRIPT  = "yolov8/ultralytics/export.py"
CONVERSION_SCRIPT   = "path/to/convert.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Pipeline stages ───────────────────────────────────────────────────────────

def step_aruco_generate(args: argparse.Namespace) -> None:
    """Step 1 — C++ ArUco dataset generator.

    Usage: ./batch_maker <input_folder> <overlay_folder> <output_name>
    """
    run(
        [
            ARUCO_GENERATOR_BIN,
            args.input_folder,
            args.overlay_folder,
            args.output_name,
        ],
        step_name="ArUco Dataset Generation (C++)",
    )


def step_data_split(args: argparse.Namespace) -> None:
    """Step 2 — split generated dataset into train / val / test."""
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"[ERROR] Split ratios must sum to 1.0 (got {total:.4f}).")
        sys.exit(1)

    split_output = f"{args.output_name}_split"
    run(
        [
            sys.executable, DATA_SPLIT_SCRIPT,
            args.output_name,      # positional: dataset_path (output of step 1)
            "--val",  args.val_ratio,
            "--test", args.test_ratio,
            "--out",  split_output,
            # --train is implicit (whatever remains after val and test)
        ],
        step_name="Train / Val / Test Split",
    )
    return split_output


def step_run_model(args: argparse.Namespace) -> None:
    """Step 3 — single training run."""
    cmd = [
        sys.executable, MODEL_SCRIPT,
        "--data",    str(Path(args.split_output) / "data.yaml"),
        "--weights", args.weights,
        "--epochs",  args.epochs,
        "--batch",   args.batch,
        "--lr0",     args.lr0,
        "--imgsz",   args.imgsz,
        "--project", args.output,
    ]
    if args.lrf is not None:
        cmd += ["--lrf", args.lrf]

    run(cmd, step_name="Model Training")


def training_loop(args: argparse.Namespace) -> None:
    """Step 3 — train and loop until user confirms results are satisfactory."""
    attempt = 1
    while True:
        print(f"\n  Training attempt #{attempt}")
        step_run_model(args)

        print("\n" + "="*60)
        answer = input("  Are you satisfied with the results? [y/n]: ").strip().lower()
        if answer == "y":
            print("[OK] Training accepted. Continuing pipeline.")
            break

        print("  Looping back to training. Adjust your parameters if needed (Enter to keep current).")

        def prompt(name, current, cast):
            val = input(f"    {name} [{current}]: ").strip()
            return cast(val) if val else current

        args.weights = prompt("weights", args.weights, str)
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
            # TODO: add optimisation-specific flags
        ],
        step_name="Run Optimizations",
    )


def step_export_onnx(args: argparse.Namespace) -> None:
    """Step 5 — ONNX export."""
    run(
        [
            sys.executable, ONNX_EXPORT_SCRIPT,
            "--model-dir", args.output,
            "--output",    args.export_output,
            # TODO: add ONNX export flags (opset version, dynamic axes, etc.)
        ],
        step_name="ONNX Export",
    )


def step_convert(args: argparse.Namespace) -> None:
    """Step 6 — conversion + post-optimization."""
    run(
        [
            sys.executable, CONVERSION_SCRIPT,
            "--input-dir",  args.export_output,
            "--output-dir", args.export_output,
            # TODO: add conversion / post-optimisation flags
        ],
        step_name="Conversion + Post-optimization",
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArUco dataset -> split -> train -> ONNX export -> convert pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data generation ───────────────────────────────────────────────────────
    gen = parser.add_argument_group("Data Generation (C++)")
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


    # ── Data split ────────────────────────────────────────────────────────────
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

    # ── Model training ────────────────────────────────────────────────────────
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


    # ── Export ────────────────────────────────────────────────────────────────
    export = parser.add_argument_group("Export")
    export.add_argument(
        "--output",
        required=True,
        help="Working directory for generated dataset and model outputs",
    )
    export.add_argument(
        "--export-output",
        required=True,
        help="Directory where the final ONNX / converted model will be saved",
    )

    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve paths
    args.input_folder   = str(Path(args.input_folder).resolve())
    args.overlay_folder = str(Path(args.overlay_folder).resolve())
    args.output_name    = "dataset"
    args.output         = str(Path(args.output).resolve())
    args.export_output  = str(Path(args.export_output).resolve())

    print("\nPipeline starting")
    print(f"  Input folder  : {args.input_folder}")
    print(f"  Overlay folder: {args.overlay_folder}")
    print(f"  Working dir   : {args.output}")
    print(f"  Export output : {args.export_output}")
    print(f"  Split         : train={args.train_ratio} / val={args.val_ratio} / test={args.test_ratio}")
    print(f"  Weights       : {args.weights}")
    print(f"  Training      : epochs={args.epochs}, batch={args.batch}, lr0={args.lr0}, lrf={args.lrf}, imgsz={args.imgsz}")
    print(f"  Project       : {args.output}")

    step_aruco_generate(args)             # Step 1
    args.split_output = step_data_split(args)                    # Step 2
    training_loop(args)                                            # Step 3 — loops until user is satisfied
    args.trained_weights = get_latest_weights(args.output, args.weights)  # resolve best.pt
    # step_run_optimizations(args)        # Step 4 — uncomment when ready
    step_export_onnx(args)                # Step 5
    step_convert(args)                    # Step 6

    print(f"\n{'='*60}")
    print("  Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()