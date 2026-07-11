#!/usr/bin/env python3
"""Train LATO-YOLO with the experimental settings described in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "configs" / "lato-yolo.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LATO-YOLO (YOLOv12n + AFPN + UTGhostConv + HPL).")
    parser.add_argument("--data", required=True, help="Dataset YAML path, e.g. datasets/visdrone.yaml")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model YAML path")
    parser.add_argument(
        "--weights",
        default="",
        help="Optional pretrained weights (e.g. yolo12n.pt). Matching layers are loaded; custom layers stay random.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1; omit for auto-selection")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="lato-yolo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3.0,
        # Manuscript augmentation settings.
        mosaic=0.5,
        mixup=0.1,
        fliplr=0.5,
        scale=0.5,  # random scale factor around 1.0 -> approximately [0.5, 1.5]
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        cache=args.cache,
        resume=args.resume,
    )
    if args.device is not None:
        train_args["device"] = args.device

    model.train(**train_args)


if __name__ == "__main__":
    main()
