#!/usr/bin/env python3
"""Run LATO-YOLO inference on images, videos, directories, streams, or URLs."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with LATO-YOLO.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    kwargs = dict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, save=args.save)
    if args.device is not None:
        kwargs["device"] = args.device
    YOLO(args.weights).predict(**kwargs)


if __name__ == "__main__":
    main()
