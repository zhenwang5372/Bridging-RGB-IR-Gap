#!/bin/bash
# ============================================================
# Scheme 2 V4 训练脚本
# 
# V4 新增功能：
#   1. param_constraint: β和γ参数约束方式
#   2. mask_center: mask 零中心化方式
# ============================================================

cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# ==================== V4 配置选项说明 ====================
#
# 【param_constraint】β和γ参数约束方式
# - 'softplus': 使用 softplus 确保 > 0（默认推荐）
# - 'abs': 使用绝对值确保 > 0
# - 'residual_alpha': 不约束，改用残差融合
# - 'none': 不约束（V3 行为）
#
# 【mask_center】mask 零中心化方式
# - 'spatial_mean': 减去空间均值（默认推荐）
# - 'tanh': 使用 tanh 激活
# - 'smap_center': S_map 阶段零中心化
# - 'none': 不零中心化（V3 行为）
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (V4 默认): softplus + spatial_mean
# 配置2 (tanh): softplus + tanh
# 配置3 (残差): residual_alpha + spatial_mean
# 配置4 (V3 兼容): none + none
#
# ==================== 训练配置 ====================

CONFIG="configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion_v4.py"

# 输出目录（根据配置命名）
# 默认配置: softplus + spatial_mean
WORK_DIR="work_dirs/ir_correction_rgb_fusion/scheme2_v4/softplus_tanh/residual_1x1"

export CUDA_VISIBLE_DEVICES=3

SEED=3407

# ==================== 开始训练 ====================

echo "============================================================"
echo "Starting Scheme 2 V4 Training"
echo "============================================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Seed: ${SEED}"
echo ""
echo "V4 新增功能："
echo "  1. param_constraint: β和γ参数约束方式"
echo "  2. mask_center: mask 零中心化方式"
echo ""
echo "默认配置："
echo "  param_constraint='softplus'"
echo "  mask_center='spatial_mean'"
echo "============================================================"

python tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --cfg-options randomness.seed=${SEED}

echo "============================================================"
echo "Training completed!"
echo "============================================================"
