# LLVIP 数据集训练配置

## 概述

本目录包含用于在 LLVIP (Low-Light Visible-Infrared Pair) 数据集上训练 RGB-IR 双模态目标检测模型的配置文件。

## 配置文件说明

| 配置文件 | IR Backbone | IR 输出通道 | 说明 |
|---------|-------------|------------|------|
| `yolow_v2_rgb_ir_llvip_text_correctionV4.py` | LiteFFTIRBackbone | [64, 128, 256] | 轻量级版本 |
| `yolow_v2_rgb_ir_llvip_text_correctionV4_V3backbone.py` | LiteFFTIRBackboneV3 | [128, 256, 512] | 通道对齐版本 |

## 使用的模块

### LLVIP 专用模块 (新增)
- **Dataset**: `LLVIPDataset` (`yolo_world/datasets/llvip_dataset.py`)
- **Transform**: `LoadIRImageFromFileLLVIP` (`yolo_world/datasets/transformers/llvip_transforms.py`)

### 复用的 FLIR 模块
- **Data Preprocessor**: `DualModalDataPreprocessor`
- **Augmentation**: `SyncMosaic`, `SyncRandomAffine`, `SyncLetterResize`, `SyncRandomFlip`, `DualModalityPhotometricDistortion`, `ThermalSpecificAugmentation`

## 数据集准备

### 目录结构
```
data/LLVIP/
├── visible/
│   ├── train/
│   │   ├── 010001.jpg
│   │   ├── 010002.jpg
│   │   └── ...
│   └── test/
│       └── ...
├── infrared/
│   ├── train/
│   │   ├── 010001.jpg  # 与 visible 同名
│   │   ├── 010002.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── coco_annotations/
    ├── train.json
    └── test.json
```

### 类别文本
已创建 `data/llvip/texts/llvip_class_texts.json`，内容为：
```json
[["person", "pedestrian", "human", "people", "walker", "man", "woman"]]
```

## 训练

### 版本1: LiteFFTIRBackbone (轻量级)
```bash
bash configs/custom_llvip/run_train_llvip.sh v1
```

### 版本2: LiteFFTIRBackboneV3 (通道对齐)
```bash
bash configs/custom_llvip/run_train_llvip.sh v3
```

### 手动训练命令
```bash
# 版本1
python tools/train.py configs/custom_llvip/yolow_v2_rgb_ir_llvip_text_correctionV4.py \
    --work-dir work_dirs/llvip/text_correction_v4/LiteFFTIRBackbone

# 版本2
python tools/train.py configs/custom_llvip/yolow_v2_rgb_ir_llvip_text_correctionV4_V3backbone.py \
    --work-dir work_dirs/llvip/text_correction_v4/LiteFFTIRBackboneV3
```

## 与 FLIR 配置的主要区别

| 项目 | FLIR | LLVIP |
|------|------|-------|
| 图像尺寸 | 640×512 → 640×640 | 1280×1024 → 1280×1280 |
| 类别数 | 4 (car, person, bicycle, dog) | 1 (person) |
| RGB/IR 区分方式 | 文件名后缀 (`*_RGB.jpg` / `*_PreviewData.jpeg`) | 目录 (`visible/` / `infrared/`) |
| Dataset 类 | `FLIRDataset` | `LLVIPDataset` |
| IR Loader | `LoadIRImageFromFile` | `LoadIRImageFromFileLLVIP` |
| Batch Size | 16 | 8 (因图像更大) |

## 显存注意事项

由于 LLVIP 训练尺寸为 1280×1280（是 FLIR 640×640 的 4 倍面积），显存消耗会显著增加：
- 建议使用至少 24GB 显存的 GPU
- 默认 batch_size=8，可根据显存情况调整
- 如显存不足，可尝试将 `img_scale` 改为 `(1024, 1024)` 或 `(640, 640)`
