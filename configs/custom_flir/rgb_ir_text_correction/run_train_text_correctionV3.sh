#!/bin/bash

# ============================
# Text-Guided IR Correction V3 Training Script
# ============================
# V3 新增：
#   - 训练时：每 50 个 iteration 打印 Alpha 值
#   - 验证时：每个 epoch 打印 Alpha 值

# 设置GPUs
export CUDA_VISIBLE_DEVICES=2

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
CONFIG="configs/custom_flir/rgb_ir_text_correction/yolow_v2_rgb_ir_flir_text_correctionV4_plus.py"
WORK_DIR="work_dirs/text_guided_ir_correction/text_correction_v4/ir_correction_v4_plus_Alpha_-0.3/LiteFFTIRBackboneV3"
BATCH_SIZE=16
LEARNING_RATE=0.0015
MAX_EPOCHS=300

echo "========================================="
echo "Text-Guided IR Correction V3 Training"
echo "========================================="
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo ""
echo "V3 新功能:"
echo "  - 训练时: 每 50 iter 打印一次 Alpha"
echo "  - 验证时: 每 epoch 打印一次 Alpha"
echo "日志中查找: [IR Correction V3]"
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
echo ""
echo "查看 Alpha 日志:"
echo "grep 'IR Correction V3' ${WORK_DIR}/*.log"
echo "========================================="

