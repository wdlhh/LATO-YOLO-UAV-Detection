# LATO-YOLO: UAV Traffic Object Detection

## Overview
LATO-YOLO is a lightweight and high-accuracy object detection framework designed for UAV-based traffic monitoring. It is optimized for real-time edge deployment and small object detection in dense aerial scenes.

## Key Contributions
- Asymptotic Feature Pyramid Network (AFPN)
- UTGhostConv lightweight convolution module
- Hybrid loss: WIoU + VFL + DFL
- Optimized YOLOv12-based architecture

## Performance
- VisDrone mAP@0.5: 41.6%
- UAV-DT mAP@0.5: 44.9%
- Speed: 185–235 FPS
- Params: 2.3M
- FLOPs: 5.7G

## Installation
```bash
git clone https://github.com/wdlhh/LATO-YOLO-UAV-Detection.git
cd LATO-YOLO-UAV-Detection
pip install -r requirements.txt
