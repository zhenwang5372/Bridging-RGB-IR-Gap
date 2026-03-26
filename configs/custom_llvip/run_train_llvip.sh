#!/bin/bash

# ============================
# Text-Guided IR Correction Training Script
# ============================

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 激活环境
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch

# 设置路径
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/mmyolo
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
# 设置确定性训练所需的环境变量（CUDA >= 10.2）
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 训练参数
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_text_correctionV4.py"
# WORK_DIR="work_dirs/text_guided_ir_correction/20260114/seed3407"
WORK_DIR="work_dirs/LLVIP/text_correction_v4/ir_correction_v4/LiteFFTIRBackbone/batch32_1280"
BATCH_SIZE=16

LEARNING_RATE=0.0015
MAX_EPOCHS=300

echo "========================================="
echo "Text-Guided IR Correction Training"
echo "========================================="
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "========================================="

# 运行训练
python tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --cfg-options \
        train_dataloader.batch_size=${BATCH_SIZE} \
        optim_wrapper.optimizer.lr=${LEARNING_RATE} \
        train_cfg.max_epochs=${MAX_EPOCHS} \
        randomness.seed=3407 \
        randomness.deterministic=True \
    # --resume

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${WORK_DIR}"
echo "========================================="

