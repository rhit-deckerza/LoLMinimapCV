#!/usr/bin/env python3
"""
Train YOLOv8-s on your generated dataset.

Usage:
  python train_yolov8s.py --data /path/to/dataset.yaml --imgsz 960 --epochs 100

Notes:
- Exposes a few common knobs; sensible defaults otherwise.
- Writes results to runs/detect/<name> with best weights at weights/best.pt.
"""

import argparse
import sys
import subprocess

def ensure_ultralytics():
    try:
        import ultralytics  # noqa: F401
        return
    except Exception:
        print("[i] Installing ultralytics ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
        print("[i] ultralytics installed.")

def main():
    ensure_ultralytics()
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="Train YOLOv8-s")
    ap.add_argument("--data", required=True, help="Path to dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=960, help="Image size (helps small icons)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="LoLMinimapCV")
    ap.add_argument("--name", default="LoLMinimapCV-yolov8s")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--lr0", type=float, default=None, help="Base LR (optional)")
    ap.add_argument("--resume", action="store_true", help="Resume last run (in same project/name)")
    args = ap.parse_args()

    model = YOLO("yolov8s.pt")

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=0,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    # Train
    model.train(**train_kwargs)

    # Validate (uses best.pt)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, project=args.project, name=f"{args.name}-val")
    print("[✓] Validation metrics:", metrics)

    # Export ONNX (optional, comment out if not needed)
    model.export(format="onnx", opset=12, dynamic=True)

if __name__ == "__main__":
    main()
