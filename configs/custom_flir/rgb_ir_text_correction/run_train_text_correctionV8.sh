#!/bin/bash
# Text-Guided IR Correction V8 Training Script
# 
# V8 核心改动：
# - 用 M_err 增强 RGB（完全不使用 IR）
# - ⭐ M_err 应用双阈值硬注意力（Hard Attention）
# - threshold_low=0.2 (抑制噪声), threshold_high=0.7 (强化目标)

export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # 确定性训练所需

ROOT_DIR="/home/ssd1/users/wangzhen01/YOLO-World-master_2"
CONFIG_FILE="${ROOT_DIR}/configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV8.py"
WORK_DIR="${ROOT_DIR}/work_dirs/text_guided_ir_correction/text_correction_v8/Merr_HardAttention_Low0.2_High0.7"
BATCH_SIZE=16
LEARNING_RATE=0.0015
MAX_EPOCHS=300
cd ${ROOT_DIR}
source /opt/conda/etc/profile.d/conda.sh
conda activate torch

echo "========================================"
echo "Training IR Correction V8"
echo "  - M_err with Hard Attention (Dual Thresholds)"
echo "  - threshold_low=0.2, threshold_high=0.7"
echo "  - NO IR features in fusion (RGB only)"
echo "========================================"
echo "Config: ${CONFIG_FILE}"
echo "Work Dir: ${WORK_DIR}"
echo "========================================"

mkdir -p ${WORK_DIR}

python ${ROOT_DIR}/tools/train.py \
    ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --cfg-options \
        train_dataloader.batch_size=${BATCH_SIZE} \
        optim_wrapper.optimizer.lr=${LEARNING_RATE} \
        train_cfg.max_epochs=${MAX_EPOCHS} \
        randomness.seed=3407 \
        randomness.deterministic=True \
    --resume auto
    2>&1 | tee ${WORK_DIR}/training.log

echo "Training completed! Work Dir: ${WORK_DIR}"
