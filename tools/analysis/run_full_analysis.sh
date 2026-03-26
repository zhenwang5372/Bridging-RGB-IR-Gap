#!/bin/bash
# 完整的 mAP50 错误分析流程
#
# 步骤:
# 1. 使用官方 test.py 运行测试并保存结果
# 2. 从保存的结果分析 mAP50 错误
#
# 使用方法:
#   bash tools/analysis/run_full_analysis.sh

set -e

# 配置
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_64.pth"
RESULTS_PKL="work_dirs/analysis_results.pkl"
OUTPUT_DIR="data/LLVIP/map50"
ANN_FILE="data/LLVIP/coco_annotations/test.json"
IMG_DIR="data/LLVIP/visible/test/"
SCORE_THR=0.3
IOU_THR=0.5
DEVICE="cuda:2"

# 检查文件
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "=============================================="
echo "Step 1: Running test.py to save results"
echo "=============================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $RESULTS_PKL"
echo "=============================================="

# 运行测试并保存结果
CUDA_VISIBLE_DEVICES=${DEVICE#cuda:} python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --out "$RESULTS_PKL"

echo ""
echo "=============================================="
echo "Step 2: Analyzing results"
echo "=============================================="

# 清理旧的输出
rm -rf "$OUTPUT_DIR"/*

# 运行分析
python tools/analysis/analyze_from_pkl.py \
    --results "$RESULTS_PKL" \
    --ann-file "$ANN_FILE" \
    --img-dir "$IMG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --score-thr "$SCORE_THR" \
    --iou-thr "$IOU_THR"

echo ""
echo "=============================================="
echo "Analysis completed!"
echo ""
echo "Results saved to:"
echo "  漏检图片: $OUTPUT_DIR/missed/"
echo "  误检图片: $OUTPUT_DIR/false_pos/"
echo ""
echo "图例说明:"
echo "  蓝色框 = Ground Truth (真实标注)"
echo "  绿色框 = 正确检测 (IoU >= 0.5)"
echo "  红色框+X = 漏检 (Missed, False Negative)"
echo "  黄色框 = 误检 (False Positive)"
echo "=============================================="
