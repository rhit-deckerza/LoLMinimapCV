#!/usr/bin/env python3
"""
Split a flat YOLO dataset (images/, labels/) into train/val folders and write dataset.yaml.

Layout in:
  dataset/
    images/   (img_000001.png, ...)
    labels/   (img_000001.txt, ...)  + classes.txt (optional, one class name per line)

Layout out:
  dataset_split/
    images/train/
    images/val/
    labels/train/
    labels/val/
    dataset.yaml    # uses absolute 'path' and relative 'train'/'val' under it

Usage (example):
  python split_yolo_dataset.py --images dataset/images --labels dataset/labels --out dataset_split --val_frac 0.2
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from random import shuffle, seed as rngseed

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def main():
    ap = argparse.ArgumentParser(description="Split YOLO dataset into train/val and write dataset.yaml")
    ap.add_argument("--images", required=True, help="Folder with images (flat)")
    ap.add_argument("--labels", required=True, help="Folder with labels (flat .txt)")
    ap.add_argument("--out",    required=True, help="Output dataset root")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction of images for val split")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--classes_txt", default="classes.txt", help="Optional class names file under labels/")
    args = ap.parse_args()

    imgs_dir = Path(args.images)
    lbls_dir = Path(args.labels)
    out_root = Path(args.out)

    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {imgs_dir}")
    if not lbls_dir.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {lbls_dir}")

    # Collect stems that have BOTH image and label
    imgs = [p for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    stems = []
    for ip in imgs:
        lp = lbls_dir / f"{ip.stem}.txt"
        if lp.exists():
            stems.append(ip.stem)

    if not stems:
        raise RuntimeError("No (image,label) pairs found. Ensure labels/*.txt match image stems.")

    rngseed(args.seed)
    shuffle(stems)

    n = len(stems)
    n_val = max(1, int(n * args.val_frac)) if n > 1 else 0
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:]) if n_val < n else set()

    # Create output dirs
    (out_root / "images/train").mkdir(parents=True, exist_ok=True)
    (out_root / "images/val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels/train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels/val").mkdir(parents=True, exist_ok=True)

    def copy_pair(stem: str, split: str):
        # copy image
        img_src = None
        for ext in IMG_EXTS:
            cand = imgs_dir / f"{stem}{ext}"
            if cand.exists():
                img_src = cand
                break
        if img_src is None:
            return
        shutil.copy2(img_src, out_root / f"images/{split}/{img_src.name}")
        # copy label
        lab_src = lbls_dir / f"{stem}.txt"
        if lab_src.exists():
            shutil.copy2(lab_src, out_root / f"labels/{split}/{lab_src.name}")

    for s in sorted(train_stems):
        copy_pair(s, "train")
    for s in sorted(val_stems):
        copy_pair(s, "val")

    # Load class names (optional)
    classes_file = lbls_dir / args.classes_txt
    if classes_file.exists():
        names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        names = []  # you can fill this later; Ultralytics can still train if labels have class ids

    # Write dataset.yaml (JSON is a valid YAML subset; Ultralytics can read this)
    ds = {
        "path": out_root.resolve().as_posix(),  # absolute base
        "train": "images/train",
        "val": "images/val",
        "nc": len(names) if names else None,
        "names": names if names else None,
    }
    # Remove keys with None (cleaner)
    ds = {k: v for k, v in ds.items() if v is not None}

    with open(out_root / "dataset.yaml", "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    print(f"[✓] Wrote {out_root/'dataset.yaml'}")
    print(f"[i] Train images: {len(train_stems)}, Val images: {len(val_stems)}")
    print(f"[i] Root: {out_root.resolve().as_posix()}")

if __name__ == "__main__":
    main()
