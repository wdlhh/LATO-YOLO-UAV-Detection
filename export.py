#!/usr/bin/env python3
"""Export a trained LATO-YOLO checkpoint."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LATO-YOLO.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine", "openvino"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    YOLO(args.weights).export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
