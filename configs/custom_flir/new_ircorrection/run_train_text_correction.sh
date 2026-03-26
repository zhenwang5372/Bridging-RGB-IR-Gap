#!/bin/bash

# ============================
# Text-Guided IR Correction Training Script
# ============================

# 设置GPU
export CUDA_VISIBLE_DEVICES=5

# 激活环境
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch

# 设置路径
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/mmyolo
export HF_ENDPOINT=https://hf-mirror.com

# 设置确定性训练所需的环境变量（CUDA >= 10.2）
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 训练参数
CONFIG="configs/custom_flir/new_ircorrection/yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons_DCTGhostV2.py"
# WORK_DIR="work_dirs/text_guided_ir_correction/20260114/seed3407"
WORK_DIR="work_dirs/new_ircorrection/IR_RGB_Merr_Cons/mean/Alpha_0.5/Beta_0.5/beixuanB/dctv2-4"
BATCH_SIZE=32   
LEARNING_RATE=0.002
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
        randomness.deterministic=False \
        # randomness.seed=3407 \
        
    # --resume

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${WORK_DIR}"
echo "========================================="

