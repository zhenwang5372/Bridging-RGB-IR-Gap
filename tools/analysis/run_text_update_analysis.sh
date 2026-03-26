#!/bin/bash
# 文本更新版本 mAP50 分析脚本
# 用法: bash tools/analysis/run_text_update_analysis.sh

set -e

# ======================== 配置 ========================
CONDA_PATH="/home/disk1/users/linsong/miniconda3"
CONDA_ENV="torch"
WORK_DIR="/home/ssd1/users/wangzhen01/YOLO-World-master_2"

# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# 配置文件和权重
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling.py"
CHECKPOINT="work_dirs/LLVIP/with_text_update/maxpooling/concat/best_coco_bbox_mAP_50_epoch_50.pth"

# 输出文件
PKL_SCORE0="work_dirs/analysis_results_text_update_score0.pkl"
PKL_SCORE0001="work_dirs/analysis_results_text_update_score0001.pkl"
REPORT_LOG="tools/analysis/map50_analysis_report_text_update.log"

# 临时配置文件（用于 score_thr=0）
CONFIG_SCORE0="configs/custom_llvip/yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling_score0.py"

# ======================== 初始化 ========================
cd "$WORK_DIR"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "========================================"
echo "文本更新版本 mAP50 分析"
echo "========================================"
echo "配置文件: $CONFIG"
echo "权重文件: $CHECKPOINT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# ======================== 创建 score_thr=0 的配置文件 ========================
echo "创建 score_thr=0 的配置文件..."
cat > "$CONFIG_SCORE0" << 'EOF'
# 临时配置文件：score_thr=0
_base_ = './yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling.py'

# 覆盖 test_cfg，设置 score_thr=0
model = dict(
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0,  # 完全无过滤
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300
    )
)
EOF
echo "已创建: $CONFIG_SCORE0"
echo ""

# ======================== 推理 score_thr=0 ========================
echo "========================================"
echo "步骤 1/3: 推理 score_thr=0"
echo "========================================"
echo "输出: $PKL_SCORE0"
echo ""

python tools/test.py \
    "$CONFIG_SCORE0" \
    "$CHECKPOINT" \
    --out "$PKL_SCORE0"

echo ""
echo "score_thr=0 推理完成!"
echo ""

# ======================== 推理 score_thr=0.001 ========================
echo "========================================"
echo "步骤 2/3: 推理 score_thr=0.001"
echo "========================================"
echo "输出: $PKL_SCORE0001"
echo ""

python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --out "$PKL_SCORE0001"

echo ""
echo "score_thr=0.001 推理完成!"
echo ""

# ======================== 清理可视化目录 ========================
echo "清理可视化目录..."
rm -rf tools/analysis/FP
rm -rf tools/analysis/FN
mkdir -p tools/analysis/FP
mkdir -p tools/analysis/FN

# ======================== 运行分析 ========================
echo "========================================"
echo "步骤 3/3: 分析并生成报告"
echo "========================================"

python tools/analysis/analyze_text_update.py \
    --pkl-score0 "$PKL_SCORE0" \
    --pkl-score0001 "$PKL_SCORE0001" \
    --ann-file "data/LLVIP/coco_annotations/test.json" \
    --img-dir "data/LLVIP/visible/test" \
    --ir-dir "data/LLVIP/infrared/test" \
    --output-log "$REPORT_LOG" \
    --fp-dir "tools/analysis/FP" \
    --fn-dir "tools/analysis/FN" \
    --fp-vis-thr 0.3

echo ""
echo "========================================"
echo "分析完成!"
echo "========================================"
echo "报告文件: $REPORT_LOG"
echo "FP 可视化: tools/analysis/FP/"
echo "FN 可视化: tools/analysis/FN/"
echo ""

# 清理临时配置文件
rm -f "$CONFIG_SCORE0"
echo "已清理临时配置文件"
