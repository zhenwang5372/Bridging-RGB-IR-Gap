#!/bin/bash
# ============================================================
# Scheme 2 V2 训练脚本
# 
# 使用改进版的 TextGuidedRGBIRFusionV2 模块
# 修复了 w_c 全部相同和 S_map 值过小的问题
# ============================================================

# 项目根目录
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# ==================== 配置选项 ====================
# 
# 可以通过修改配置文件中的以下参数来切换不同方案：
#
# 【gap_method】类别权重 w_c 的计算方式
# --------------------------------------------------
# 'logits' (推荐 ⭐):
#   - 使用 softmax 之前的 logits 平均值
#   - 不同类别会有不同的 gap 值
#
# 'max':
#   - 使用 softmax 后的最大激活值
#   - 捕捉最强激活位置的强度
#
# 'entropy':
#   - 使用熵来度量 attention 分布的确定性
#   - 熵越低，类别越可能存在
#
# 【smap_method】S_map 的计算方式
# --------------------------------------------------
# 'sigmoid' (推荐 ⭐):
#   - 使用 sigmoid 归一化 logits
#   - 简单直接，值域可控
#
# 'sigmoid_temp':
#   - 带可学习温度参数的 sigmoid
#   - 更灵活，模型自适应调节
#
# 'normalized':
#   - 完整归一化流程
#   - 数值最稳定，语义最清晰
#
# ==================== 训练配置 ====================

# 配置文件
CONFIG="configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion_v2.py"

# 输出目录
WORK_DIR="work_dirs/ir_correction_rgb_fusion/scheme2_v2/logits_normalized"

# GPU 设置
export CUDA_VISIBLE_DEVICES=3

# 随机种子
SEED=3407

# ==================== 开始训练 ====================

echo "============================================================"
echo "Starting Scheme 2 V2 Training"
echo "============================================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Seed: ${SEED}"
echo ""
echo "V2 改进点："
echo "  1. 修复 w_c 全部相同的问题（使用 logits GAP）"
echo "  2. 修复 S_map 值过小的问题（使用 sigmoid 归一化）"
echo "============================================================"

python tools/train.py \
    ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --cfg-options randomness.seed=${SEED}

echo "============================================================"
echo "Training completed!"
echo "============================================================"
