#!/bin/bash
# ============================================================
# Scheme 2 V3 训练脚本
# 
# V3 新增功能：
#   1. smap_order: S_map 计算顺序（sum_first | multiply_first）
#   2. mask_method: Mask 生成方式（conv_gen | residual | dual_branch | se_spatial）
# ============================================================

cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# ==================== 配置选项说明 ====================
#
# 【smap_order】S_map 计算顺序
# - 'sum_first': 先按类别求和，再做 Hadamard 积（默认）
# - 'multiply_first': 先做 Hadamard 积，再按类别求和
#
# 【mask_method】Mask 生成方式
# - 'conv_gen': 轻量级卷积生成器（默认）
# - 'residual': 残差 Mask 细化
# - 'dual_branch': 双分支 Mask 生成
# - 'se_spatial': SE通道注意力 + 空间卷积
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (默认): smap_order='sum_first', mask_method='conv_gen'
# 配置2 (轻量): smap_order='sum_first', mask_method='residual'
# 配置3 (高精度): smap_order='multiply_first', mask_method='dual_branch'
# 配置4 (通道增强): smap_order='sum_first', mask_method='se_spatial'
#
# ==================== 训练配置 ====================

CONFIG="configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion_v3.py"

# 输出目录（根据配置命名）
# 默认配置: sum_first + conv_gen
WORK_DIR="work_dirs/ir_correction_rgb_fusion/scheme2_v3/sumfirst_refine_conv"

export CUDA_VISIBLE_DEVICES=3

SEED=3407

# ==================== 开始训练 ====================

echo "============================================================"
echo "Starting Scheme 2 V3 Training"
echo "============================================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Seed: ${SEED}"
echo ""
echo "V3 新增功能："
echo "  1. smap_order: S_map 计算顺序可配置"
echo "  2. mask_method: Mask 生成方式可配置（4种方案）"
echo "============================================================"

python tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --cfg-options randomness.seed=${SEED}

echo "============================================================"
echo "Training completed!"
echo "============================================================"
