# LATO-YOLO 完整代码包

本项目将论文中的三处改进完整接入 YOLOv12 检测框架：

- **3.2 AFPN 多尺度特征融合**：对 `C2/C3/C4/C5`（步长 4/8/16/32）执行渐进式融合，输出 `P2/P3/P4/P5` 四尺度检测特征。
- **3.3 UTGhostConv 轻量卷积**：按 3:1 的主通道/蒸馏通道进行非对称通道蒸馏，结合标准卷积、深度可分离卷积和 Channel Shuffle。
- **3.4 HPL 损失函数**：训练阶段组合 **WIoU + VFL + DFL**，推理阶段不增加额外参数和计算。

项目已经包含完整的本地 `ultralytics` 源码、模型配置、损失函数、训练/验证/推理/导出脚本以及 8 组消融实验配置，不需要再手工修改 site-packages。

## 目录

```text
LATO_YOLO_complete/
├── ultralytics/                     # 完整 YOLOv12-compatible 框架源码
│   ├── nn/modules/lato.py            # AFPN + UTGhostConv
│   ├── utils/lato_loss.py            # HPL: WIoU + VFL + DFL
│   └── cfg/models/12/lato-yolo.yaml  # 内置完整模型配置
├── configs/
│   ├── lato-yolo.yaml                # 完整 LATO-YOLO
│   └── ablation_1...8_*.yaml         # 8 组消融配置
├── datasets/                         # VisDrone / UAV-DT 数据 YAML 模板
├── train.py
├── val.py
├── predict.py
├── export.py
├── scripts/smoke_test.py
└── docs/IMPLEMENTATION_NOTES.md
```

## 1. 环境安装

建议新建独立环境：

```bash
conda create -n lato_yolo python=3.10 -y
conda activate lato_yolo
cd LATO_YOLO_complete
pip install -r requirements.txt
```

安装与你 CUDA 版本匹配的 PyTorch 后再安装其余依赖也可以。论文中的实验环境为 PyTorch 2.0、CUDA 11.8；本代码使用标准 PyTorch/Ultralytics 接口实现。

## 2. 数据准备

将 VisDrone 或 UAV-DT 转为标准 YOLO 检测格式：

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

修改：

```text
datasets/visdrone.yaml
datasets/uavdt.yaml
```

中的 `path`。

## 3. 训练完整 LATO-YOLO

从随机初始化训练：

```bash
python train.py \
  --data datasets/visdrone.yaml \
  --model configs/lato-yolo.yaml \
  --epochs 200 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

加载 YOLOv12n 预训练权重进行部分迁移：

```bash
python train.py \
  --data datasets/visdrone.yaml \
  --model configs/lato-yolo.yaml \
  --weights yolo12n.pt \
  --epochs 200 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

训练脚本默认采用论文实验设置：AdamW、初始学习率 0.01、权重衰减 0.0005、Cosine 学习率、3 个 warmup epoch、Mosaic 0.5、MixUp 0.1、水平翻转 0.5、随机缩放约 0.5–1.5 和 HSV 增强。

## 4. 验证

```bash
python val.py \
  --weights runs/detect/lato-yolo/weights/best.pt \
  --data datasets/visdrone.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

## 5. 推理

```bash
python predict.py \
  --weights runs/detect/lato-yolo/weights/best.pt \
  --source path/to/images_or_video \
  --imgsz 640 \
  --device 0
```

## 6. 导出 ONNX

```bash
python export.py \
  --weights runs/detect/lato-yolo/weights/best.pt \
  --format onnx \
  --imgsz 640
```

## 7. 复现 8 组消融实验

例如只使用 AFPN：

```bash
python train.py --data datasets/visdrone.yaml --model configs/ablation_2_afpn.yaml --device 0
```

完整模型：

```bash
python train.py --data datasets/visdrone.yaml --model configs/ablation_8_lato_yolo_full.yaml --device 0
```

其余配置对应论文消融表中的 1–8 组。

## 8. 快速自检

```bash
python scripts/smoke_test.py
```

自检会完成：模型解析、P2-P5 四尺度推理、HPL 前向计算以及一次反向传播。

## 9. 关键实现位置

```text
AFPN          -> ultralytics/nn/modules/lato.py::AFPN
空间自适应融合 -> ultralytics/nn/modules/lato.py::AdaptiveSpatialFusion
UTGhostConv   -> ultralytics/nn/modules/lato.py::UTGhostConv
WIoU v3       -> ultralytics/utils/lato_loss.py::WIoUv3BBoxLoss
HPL           -> ultralytics/utils/lato_loss.py::HPLDetectionLoss
模型解析接入    -> ultralytics/nn/tasks.py
完整模型 YAML   -> configs/lato-yolo.yaml
```

更详细的方法—代码对应关系见 `docs/IMPLEMENTATION_NOTES.md`。

## 说明

论文给出了三项改进的总体结构、通道比例和损失组合，但未公开所有仓库级工程细节，例如精确的 YOLOv12 提交版本、UTGhostConv 替换的每一个层号、全部 AFPN 通道宽度以及全部 WIoU-v3 常数。因此，本代码包对这些未明确项采用了可运行、可复现实验的工程实现，并将关键选择集中在 YAML 和模块参数中，便于后续按作者真实训练记录进一步对齐。

本项目中打包的 Ultralytics 源码遵循其 AGPL-3.0 许可证，许可证文本见 `LICENSE-ULTRALYTICS.txt`。
