# CDMDet: Bridging the RGB-IR Gap

**Bridging the RGB-IR Gap: Consensus and Discrepancy Modeling for Text-Guided Multispectral Detection**

## Overview

CDMDet is a novel framework for RGB-IR multispectral object detection that leverages natural language (text) semantics to bridge the modality gap between RGB and infrared imagery. Built upon YOLO-World, our approach introduces three key innovations:

1. **Consensus and Discrepancy Module (IR_RGB_Merr_Cons)**: A dual-branch module that explicitly models both cross-modal consensus (M_cons) and discrepancy (M_err) between RGB and IR attention maps, enabling targeted enhancement of IR features.

2. **LiteDCTGhost IR Backbone**: A lightweight IR feature extractor based on DCT (Discrete Cosine Transform) frequency decomposition and enhanced Ghost modules, achieving efficient multi-scale IR feature extraction with learnable frequency separation.

3. **TextUpdateConcatPool**: A multi-scale text update module that uses both max-pooling and average-pooling to extract complementary visual representations (18 tokens per level, 54 total across 3 FPN levels), updating text embeddings with richer visual context through multi-head cross-attention.

### Architecture

```
RGB Image ──> YOLOv8-CSPDarknet ──┐
                                  ├──> IR_RGB_Merr_Cons ──> RGB-IR Fusion ──> Text-Guided RGB Enhancement ──> Aggregator ──> Detection Head
IR Image  ──> LiteDCTGhost IR ────┘         ↑                                        ↑                                          ↑
                                            │                                        │                                          │
Text      ──> CLIP Text Encoder ────────────┴────────────────────────────────────────┴──────> TextUpdateConcatPool ─────────────┘
```

**Data Flow:**
- **RGB stream**: Extracts multi-scale features (P3/P4/P5) via YOLOv8 CSPDarknet backbone
- **IR stream**: Extracts frequency-decomposed features via LiteDCTGhost backbone with adaptive frequency separation
- **Text stream**: CLIP text encoder generates category-level text embeddings
- **Consensus & Discrepancy**: Text-guided attention maps from both modalities are used to compute M_err = A_ir * (1 - A_rgb) and M_cons = A_ir * A_rgb
- **Fusion & Enhancement**: Multi-level cross-modal fusion followed by text-guided RGB enhancement produces class-specific features
- **Text Update**: Visual features are pooled (concat of max + avg pooling) and used to update text embeddings via cross-attention
- **Detection**: Class dimension aggregation feeds into YOLO-World detection head

## Key Modules

| Module | File | Description |
|--------|------|-------------|
| DualStreamYOLOWorldDetector | `yolo_world/models/detectors/` | Main dual-stream detector |
| DualStreamMultiModalYOLOBackbone | `yolo_world/models/backbones/` | Dual-stream backbone with text guidance |
| LiteDCTGhostIRBackboneV2 | `yolo_world/models/backbones/IR_backbone/` | DCT-based lightweight IR backbone |
| IR_RGB_Merr_Cons | `yolo_world/models/necks/text_guided_ir_correction/` | Consensus & Discrepancy modeling |
| TextGuidedRGBEnhancementV2 | `yolo_world/models/necks/` | Text-guided RGB feature enhancement |
| TextUpdateConcatPool | `yolo_world/models/necks/` | Multi-scale text update with concat pooling |
| MultiLevelRGBIRFusion | `yolo_world/models/necks/` | Multi-level RGB-IR fusion |
| ClassDimensionAggregator | `yolo_world/models/necks/` | Class dimension aggregation |
| FLIRDataset | `yolo_world/datasets/` | FLIR aligned RGB-IR dataset |
| Synchronized Transforms | `yolo_world/datasets/transformers/` | Synchronized RGB-IR augmentation |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3

### Setup

```bash
# Clone the repository
git clone https://github.com/zhenwang5372/CDMDet.git
cd CDMDet

# Create conda environment
conda create -n cdmdet python=3.8 -y
conda activate cdmdet

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements/basic_requirements.txt

# Install mmcv
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Install in development mode
pip install -e .
```

## Data Preparation

### FLIR Aligned Dataset

Download the [FLIR Aligned Dataset](https://www.flir.com/oem/adas/adas-dataset-form/) and organize as follows:

```
data/
└── data/
    └── flir/
        ├── root/
        │   └── autodl-tmp/
        │       └── data/
        │           └── FLIR_V1_aligned/
        │               └── align/
        │                   ├── JPEGImages/
        │                   │   ├── *_RGB.jpg
        │                   │   └── *_PreviewData.jpeg
        │                   └── annotations_fixed/
        │                       ├── train_fixed.json
        │                       └── val_fixed.json
        └── texts/
            └── flir_class_texts.json
```

The dataset contains 4 categories: **car**, **person**, **bicycle**, **dog**.

### Pretrained Weights

Download YOLO-World v2 pretrained weights:

```bash
mkdir -p checkpoints
wget -P checkpoints/ https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth
```

## Training

### Single GPU

```bash
python tools/train.py configs/custom_flir/new_ircorrection/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py
```

### Multi-GPU (Distributed)

```bash
bash tools/dist_train.sh configs/custom_flir/new_ircorrection/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py <NUM_GPUS>
```

## Evaluation

```bash
python tools/test.py configs/custom_flir/new_ircorrection/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py <CHECKPOINT_PATH>
```

## Acknowledgements

This project is built upon the following open-source projects:

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) - Real-time Open-Vocabulary Object Detection
- [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab Detection Toolbox
- [MMYOLO](https://github.com/open-mmlab/mmyolo) - OpenMMLab YOLO Series Toolbox
- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-Training

## License

This project is released under the [Apache 2.0 License](LICENSE).
