#!/usr/bin/env python3
"""
Generate a synthetic icon-detection dataset on a single background.
- Pastes each icon at a configurable scale (e.g., 0.5 = half size).
- Crops icons into circles before compositing.
- Class name = icon filename stem.
- Each image contains 5–10 DISTINCT classes (1 instance per selected class).
- YOLO labels (normalized xywh) use the icon's bounding rectangle.
"""

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def load_icons(icons_dir: Path) -> Dict[str, Image.Image]:
    icons = {}
    for p in sorted(icons_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            try:
                img = Image.open(p).convert("RGBA")
                icons[p.stem] = img
            except Exception:
                pass
    if not icons:
        raise ValueError(f"No icons found in {icons_dir}")
    return icons

def resize_circle(bg_size: Tuple[int,int], icon: Image.Image, scale_factor: float) -> Image.Image:
    """Resize icon by scale_factor, fit into bg, crop to circle."""
    bw, bh = bg_size
    iw, ih = icon.size

    # Apply scale factor
    target_w = max(1, int(iw * scale_factor))
    target_h = max(1, int(ih * scale_factor))

    # Further scale if larger than background
    scale = min(1.0, bw / max(1, target_w), bh / max(1, target_h))
    if scale < 1.0:
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))

    resized = icon.resize((target_w, target_h), Image.LANCZOS)

    # Pad to square for clean circle
    side = max(target_w, target_h)
    square = Image.new("RGBA", (side, side), (0,0,0,0))
    offset = ((side - target_w)//2, (side - target_h)//2)
    square.alpha_composite(resized, offset)

    # Circle mask
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, side-1, side-1], fill=255)
    circ = Image.new("RGBA", (side, side), (0,0,0,0))
    circ.paste(square, (0,0), mask)
    return circ

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def to_yolo_xywh_norm(box, W, H):
    x1,y1,x2,y2 = box
    w,h = max(0, x2-x1), max(0, y2-y1)
    cx, cy = x1 + w/2.0, y1 + h/2.0
    return (cx/W, cy/H, w/W, h/H)

def generate(background, icons_dir, out_images, out_labels,
             instances_per_class=100, min_classes=5, max_classes=10,
             iou_threshold=0.2, scale=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    bg = Image.open(background).convert("RGBA"); BW, BH = bg.size
    icons = load_icons(icons_dir)
    class_names = sorted(icons.keys()); cls_to_id = {c:i for i,c in enumerate(class_names)}

    # Helper files
    (out_labels / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    (out_labels / "dataset.yaml").write_text(
        f"path: {out_images.parent.as_posix()}\ntrain: {out_images.name}\nval: {out_images.name}\n"
        f"nc: {len(class_names)}\nnames: {class_names}\n", encoding="utf-8")

    counts = {c:0 for c in class_names}; image_index = 0
    while True:
        remaining = [c for c in class_names if counts[c] < instances_per_class]
        if not remaining: break
        k = random.randint(min_classes, max_classes)
        selected = set([random.choice(remaining)])
        pool = class_names.copy(); random.shuffle(pool)
        for c in pool:
            if len(selected) >= k: break
            selected.add(c)
        canvas = bg.copy(); placed = []
        for cls in selected:
            circ_icon = resize_circle((BW,BH), icons[cls], scale)
            iw, ih = circ_icon.size
            ok = False
            for _ in range(200):
                x1 = random.randint(0, max(0, BW - iw)); y1 = random.randint(0, max(0, BH - ih))
                x2, y2 = x1+iw, y1+ih; box = (x1,y1,x2,y2)
                if all(iou_xyxy(box, (a,b,c,d)) < iou_threshold for (_,a,b,c,d) in placed):
                    canvas.alpha_composite(circ_icon, (x1,y1))
                    placed.append((cls_to_id[cls], x1,y1,x2,y2)); ok = True; break
            if not ok: continue
        if not placed: continue
        image_index += 1
        img_name = f"img_{image_index:06d}.png"; lbl_name = f"img_{image_index:06d}.txt"
        canvas.convert("RGB").save(out_images / img_name)
        lines = []
        for (cid, x1,y1,x2,y2) in placed:
            x,y,w,h = to_yolo_xywh_norm((x1,y1,x2,y2), BW, BH)
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (out_labels / lbl_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        for (cid, *_rest) in placed:
            cname = class_names[cid]
            if counts[cname] < instances_per_class: counts[cname] += 1
    print(f"Done. Images: {image_index}, Instances: {sum(counts.values())}")
    for c in class_names: print(f"  {c}: {counts[c]}")

def main():
    ap = argparse.ArgumentParser(description="Generate circular icon dataset (YOLO labels).")
    ap.add_argument("--background", type=Path, required=True)
    ap.add_argument("--icons_dir", type=Path, required=True)
    ap.add_argument("--out_images", type=Path, required=True)
    ap.add_argument("--out_labels", type=Path, required=True)
    ap.add_argument("--instances_per_class", type=int, default=100)
    ap.add_argument("--min_classes", type=int, default=5)
    ap.add_argument("--max_classes", type=int, default=10)
    ap.add_argument("--iou_threshold", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=0.5, help="Resize factor for icons (e.g., 0.5 = half size)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    generate(args.background, args.icons_dir, args.out_images, args.out_labels,
             args.instances_per_class, args.min_classes, args.max_classes,
             args.iou_threshold, args.scale, args.seed)

if __name__ == "__main__":
    main()
