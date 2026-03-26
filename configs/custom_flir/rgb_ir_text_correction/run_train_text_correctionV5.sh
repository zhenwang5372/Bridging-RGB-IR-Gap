#!/bin/bash
# Text-Guided IR Correction V5 Training Script
# 
# V5 核心改动：
# - 移除 Key 投影的 bias
# - 对 Q、K 进行 L2 Normalize（余弦相似度）
# - 使用 scale=10.0 代替 1/sqrt(d_k)
# - Alpha 初始值 -0.3
# - 随机种子: 3407

# ==============================
# Configuration
# ==============================
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # 确定性训练所需

# Paths
ROOT_DIR="/home/ssd1/users/wangzhen01/YOLO-World-master_2"
CONFIG_FILE="${ROOT_DIR}/configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV5.py"
WORK_DIR="${ROOT_DIR}/work_dirs/text_guided_ir_correction/text_correction_v5/CosineScale10.0_Alpha-0.3"

# Training parameters
NUM_GPUS=1
BATCH_SIZE=16
MAX_EPOCHS=300

# ==============================
# Environment Setup
# ==============================
cd ${ROOT_DIR}

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate torch

# ==============================
# Training Information
# ==============================
echo "========================================"
echo "Training IR Correction V5 (Cosine Similarity)"
echo "========================================"
echo "Config: ${CONFIG_FILE}"
echo "Work Dir: ${WORK_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo ""
echo "V5 Features:"
echo "  - Cosine Similarity (L2 Normalized Q and K)"
echo "  - Scale = 10.0"
echo "  - Alpha init = -0.3"
echo "  - Random Seed = 3407"
echo "  - No bias in Key projection"
echo "========================================"
echo ""

# ==============================
# Create work directory
# ==============================
mkdir -p ${WORK_DIR}

# ==============================
# Start Training
# ==============================
python ${ROOT_DIR}/tools/train.py \
    ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --cfg-options \
        train_dataloader.batch_size=${BATCH_SIZE} \
        train_cfg.max_epochs=${MAX_EPOCHS} \
        randomness.seed=3407 \
        randomness.deterministic=True \
    2>&1 | tee ${WORK_DIR}/training.log

echo ""
echo "========================================"
echo "Training completed!"
echo "Work Dir: ${WORK_DIR}"
echo "========================================"
