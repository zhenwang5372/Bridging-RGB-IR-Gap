#!/bin/bash

# ============================
# LLVIP Multi-GPU Distributed Training Script
# ============================
# 使用 3 张 GPU (0, 1, 3) 进行分布式训练
# 总 batch_size = 每卡 batch_size × GPU 数量

# 设置 GPU (0, 1, 3 三张卡)
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# 激活环境
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch

# 设置路径
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/mmyolo
export HF_ENDPOINT=https://hf-mirror.com

# 设置确定性训练所需的环境变量
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling.py"
WORK_DIR="work_dirs/LLVIP/with_text_update/use_pretrain_false/maxpooling/concat"
# ⭐ 关键参数说明：
# - 每卡 batch_size = 11 (或 10)
# - 总 batch_size = 11 × 3 = 33 (接近 32)
# - 学习率可以适当增大（线性缩放）或保持不变
BATCH_SIZE_PER_GPU=8
LEARNING_RATE=2e-4   # 可以尝试 0.002 或 0.003（线性缩放）
MAX_EPOCHS=100

echo "========================================="
echo "LLVIP Multi-GPU Distributed Training"
echo "========================================="
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "Batch Size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total Batch Size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "========================================="

# 使用 torchrun 进行分布式训练 (PyTorch >= 1.9 推荐)
torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port=29503 \
    tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch \
    --cfg-options \
        train_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
        optim_wrapper.optimizer.lr=${LEARNING_RATE} \
        train_cfg.max_epochs=${MAX_EPOCHS} \
        randomness.seed=3407 \
        randomness.deterministic=False

# 或者使用 tools/dist_train.sh (取消注释以使用)
# ./tools/dist_train.sh \
#     ${CONFIG} \
#     ${NUM_GPUS} \
#     --work-dir ${WORK_DIR} \
#     --cfg-options \
#         train_dataloader.batch_size=${BATCH_SIZE_PER_GPU} \
#         optim_wrapper.optimizer.lr=${LEARNING_RATE} \
#         train_cfg.max_epochs=${MAX_EPOCHS}

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${WORK_DIR}"
echo "========================================="
