# usage python split_yolo.py /path/to/dataset --test 15 --val 15 --out yolo_split --seed 123

#!/usr/bin/env python3
"""
Split a YOLO-style dataset into YOLOv8 folder structure.

Expected input structure:
dataset_root/
  images/
    img1.jpg|jpeg|png|bmp|webp ...
  labels/
    img1.txt ...

Output structure:
output_name/
  images/{train,val,test}/...
  labels/{train,val,test}/...

Rules enforced:
- #images must equal #labels (counts)
- every image must have a corresponding .txt label (same stem)
- randomly split into test% and val% (defaults 15/15); rest -> train
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Set


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def list_images(images_dir: Path) -> List[Path]:
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def list_label_stems(labels_dir: Path) -> Set[str]:
    labels = [p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    return {p.stem for p in labels}


def ensure_dirs(out_root: Path) -> None:
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def compute_split_sizes(n: int, test_pct: float, val_pct: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    test_n = int(round(n * (test_pct / 100.0)))
    val_n = int(round(n * (val_pct / 100.0)))

    if test_n + val_n > n:
        overflow = test_n + val_n - n
        reduce_val = min(val_n, overflow)
        val_n -= reduce_val
        overflow -= reduce_val
        test_n = max(0, test_n - overflow)

    train_n = n - test_n - val_n
    return train_n, val_n, test_n


def copy_pair(img_path: Path, lbl_path: Path, out_root: Path, split: str) -> None:
    dst_img = out_root / "images" / split / img_path.name
    dst_lbl = out_root / "labels" / split / lbl_path.name
    shutil.copy2(img_path, dst_img)
    shutil.copy2(lbl_path, dst_lbl)


def parse_class_map(class_map_path: Path) -> dict[int, str]:
    """Parse overlay_index_map.txt into {class_id: class_name} dict."""
    classes = {}
    with open(class_map_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                class_id = int(parts[0])
                class_name = Path(parts[1]).stem
                classes[class_id] = class_name
            except ValueError:
                continue
    return classes


def write_yaml(out_root: Path, classes: dict[int, str]) -> None:
    """Write data.yaml for YOLOv8 training."""
    yaml_path = out_root / "data.yaml"

    num_classes = max(classes.keys()) + 1
    names = [classes.get(i, f"class_{i}") for i in range(num_classes)]

    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {names}\n")

    print(f"[OK] data.yaml written to: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset into YOLOv8 structure (images/labels with train/val/test) and generate data.yaml."
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset root containing images/ and labels/")
    parser.add_argument("--test",      type=float, default=15.0,  help="Test split percentage (default: 15)")
    parser.add_argument("--val",       type=float, default=15.0,  help="Validation split percentage (default: 15)")
    parser.add_argument("--out",       type=str,   default="output", help='Output folder name (default: "output")')
    parser.add_argument("--seed",      type=int,   default=42,    help="Random seed (default: 42)")
    parser.add_argument(
        "--class-map",
        type=str,
        required=True,
        help="Path to overlay_index_map.txt mapping class_id to overlay filename",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path).expanduser().resolve()
    images_dir   = dataset_root / "images"
    labels_dir   = dataset_root / "labels"

    if not dataset_root.exists():
        die(f"Dataset path does not exist: {dataset_root}")
    if not images_dir.is_dir():
        die(f"Missing images/ folder: {images_dir}")
    if not labels_dir.is_dir():
        die(f"Missing labels/ folder: {labels_dir}")

    if args.test < 0 or args.val < 0:
        die("Split percentages must be non-negative.")
    if args.test + args.val >= 100:
        die("test% + val% must be < 100 (so train has something left).")

    class_map_path = Path(args.class_map).expanduser().resolve()
    if not class_map_path.exists():
        die(f"Class map file not found: {class_map_path}")

    classes = parse_class_map(class_map_path)
    if not classes:
        die(f"No valid class entries found in: {class_map_path}")

    print(f"[OK] Loaded {len(classes)} classes from {class_map_path.name}")
    for cid, cname in sorted(classes.items()):
        print(f"       {cid}: {cname}")

    images = list_images(images_dir)
    label_stems = list_label_stems(labels_dir)

    if len(images) == 0:
        die(f"No images found in: {images_dir}")

    num_labels = len(list(labels_dir.glob("*.txt")))
    if len(images) != num_labels:
        die(f"#images ({len(images)}) != #labels ({num_labels}).")

    missing = []
    for img in images:
        if img.stem not in label_stems:
            missing.append(img.name)
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        die(f"Some images do not have corresponding label .txt files (by stem). Example(s): {preview}")

    rng = random.Random(args.seed)
    indices = list(range(len(images)))
    rng.shuffle(indices)

    train_n, val_n, test_n = compute_split_sizes(len(images), args.test, args.val)
    train_idx = indices[:train_n]
    val_idx   = indices[train_n : train_n + val_n]
    test_idx  = indices[train_n + val_n : train_n + val_n + test_n]

    out_root = Path(args.out).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        die(f"Output directory already exists and is not empty: {out_root}")

    ensure_dirs(out_root)

    def do_split(idxs: List[int], split: str) -> None:
        for i in idxs:
            img = images[i]
            lbl = labels_dir / f"{img.stem}.txt"
            if not lbl.exists():
                die(f"Selected image missing corresponding label (should not happen): {img.name}")
            copy_pair(img, lbl, out_root, split)

    do_split(train_idx, "train")
    do_split(val_idx,   "val")
    do_split(test_idx,  "test")

    write_yaml(out_root, classes)

    print("Done.")
    print(f"Input:  {dataset_root}")
    print(f"Output: {out_root}")
    print(f"Counts: train={train_n}, val={val_n}, test={test_n}, total={len(images)}")
    print("Structure:")
    print(f"  {out_root}/images/{{train,val,test}}")
    print(f"  {out_root}/labels/{{train,val,test}}")
    print(f"  {out_root}/data.yaml")


if __name__ == "__main__":
    main()