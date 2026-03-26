# Bridging the RGB-IR Gap: Consensus and Discrepancy Modeling for Text-Guided Multispectral Detection

## Overview

Official implementation of **"Bridging the RGB-IR Gap: Consensus and Discrepancy Modeling for Text-Guided Multispectral Detection"**.

We propose a text-guided multispectral object detection framework that explicitly models the **consensus** and **discrepancy** between RGB and infrared (IR) modalities. Our method introduces:

- **Consensus and Discrepancy Module (IR_RGB_Merr_Cons):** A dual-branch module that explicitly models both cross-modal consensus (M_cons) and discrepancy (M_err) between RGB and IR attention maps, enabling targeted enhancement of IR features.

- **LiteDCTGhost IR Backbone:** A lightweight IR feature extractor based on DCT (Discrete Cosine Transform) frequency decomposition and enhanced Ghost modules, achieving efficient multi-scale IR feature extraction with learnable frequency separation.

- **TextUpdateConcatPool:** A multi-scale text update module that uses both max-pooling and average-pooling to extract complementary visual representations (18 tokens per level, 54 total across 3 FPN levels), updating text embeddings with richer visual context through multi-head cross-attention.

## Architecture

```
RGB Image → CSPDarknet ────────────┐
                                   ├─→ IR_RGB_Merr_Cons → Fusion → RGB Enhancement → Aggregator
IR Image  → DCT-Ghost IR Backbone ─┘         ↑                         ↑                 │
                                              │                         │          aggregated feats ──┐
Text      → CLIP Text Encoder ────────────────┴─────────────────────────┼─→ TextUpdateConcatPool     │
                                                                                    │                 │
                                                                          updated text embeddings     │
                                                                                    │                 │
                                                                                    └────── Head ─────┘
```

### Consensus and Discrepancy Module (Core)

Computes two complementary maps from text-guided attention:

- **Discrepancy Map:** `M_err = A_ir × (1 - A_rgb)` — IR-specific regions where RGB provides weak support
- **Consensus Map:** `M_cons = A_ir × A_rgb` — Regions where both modalities agree

Final correction:
```
IR_corrected = X_ir × (1 + β × Cons_map − α × Error_map)
```
where α and β are learnable parameters.

## Installation

```bash
conda create -n bridging-rgb-ir python=3.8 -y
conda activate bridging-rgb-ir

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install openmim
mim install mmcv==2.2.0 mmdet==3.0.0 mmengine==0.10.3
pip install transformers tokenizers opencv-python supervision==0.19.0

# Install MMYOLO
mkdir -p third_party && cd third_party
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo && pip install -e . && cd ../..

# Install this package
pip install -e .
```

## Dataset

### FLIR Aligned

Download [FLIR Aligned Dataset](https://www.flir.com/oem/adas/adas-dataset-form/) and organize:

```
data/FLIR_aligned/
├── JPEGImages/
│   ├── FLIR_00001_RGB.jpg
│   ├── FLIR_00001_PreviewData.jpeg
│   └── ...
└── annotations_fixed/
    ├── train_fixed.json
    └── val_fixed.json
```

### Pre-trained Weights

```bash
mkdir -p checkpoints
# Download pretrained weights to checkpoints/
```

## Training

```bash
# Single GPU
python tools/train.py configs/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py

# Multi-GPU
bash tools/dist_train.sh configs/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py <NUM_GPUS>
```

## Evaluation

```bash
python tools/test.py configs/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py <CHECKPOINT>
```

## Project Structure

```
├── configs/
│   ├── yolow_v2_..._DCTGhostV2.py          # Main config
│   └── ablation/                             # Ablation study configs
├── yolo_world/
│   ├── models/
│   │   ├── detectors/                        # Dual-stream detector
│   │   ├── backbones/                        # LiteDCTGhost IR, CLIP, dual-stream backbone
│   │   ├── necks/
│   │   │   ├── text_guided_ir_correction/    # Consensus & Discrepancy Module
│   │   │   ├── rgb_ir_fusion.py              # Multi-level RGB-IR fusion
│   │   │   ├── text_guided_rgb_enhancement_v2.py
│   │   │   ├── multiscale_text_update_v4.py  # TextUpdateConcatPool
│   │   │   └── class_dimension_aggregator.py
│   │   ├── dense_heads/                      # Detection head
│   │   ├── data_preprocessors/               # Dual-modal data preprocessing
│   │   └── layers/
│   ├── datasets/                             # FLIRDataset, transforms
│   └── engine/                               # Optimizer constructor
├── tools/                                    # train.py, test.py, dist scripts
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Key Modules

| Module | File | Description |
|--------|------|-------------|
| IR_RGB_Merr_Cons | `necks/text_guided_ir_correction/IR_RGB_Merr_Cons.py` | Consensus & discrepancy modeling |
| LiteDCTGhostIRBackboneV2 | `backbones/IR_backbone/lite_dct_ghost_ir_backbone_v2.py` | DCT + Ghost module lightweight IR backbone |
| TextUpdateConcatPool | `necks/multiscale_text_update_v4.py` | Multi-scale text update (max+avg pool concat) |
| TextGuidedRGBEnhancementV2 | `necks/text_guided_rgb_enhancement_v2.py` | Text-guided RGB enhancement |
| MultiLevelRGBIRFusion | `necks/rgb_ir_fusion.py` | Multi-level RGB-IR fusion |
| DualStreamBackboneV2 | `backbones/dual_stream_class_specific_backbone_v2.py` | Orchestrates all sub-modules |

## Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)
- [CLIP](https://github.com/openai/CLIP)

## License

[Apache 2.0](LICENSE)
