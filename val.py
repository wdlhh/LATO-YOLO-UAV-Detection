#!/usr/bin/env python3
"""Validate a trained LATO-YOLO checkpoint."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LATO-YOLO.")
    parser.add_argument("--weights", required=True, help="Checkpoint path, e.g. runs/detect/lato-yolo/weights/best.pt")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    kwargs = dict(data=args.data, imgsz=args.imgsz, batch=args.batch)
    if args.device is not None:
        kwargs["device"] = args.device
    YOLO(args.weights).val(**kwargs)


if __name__ == "__main__":
    main()
