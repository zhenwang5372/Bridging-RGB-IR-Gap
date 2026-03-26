#!/bin/bash
# mAP50 错误分析脚本
#
# 分析 LLVIP 验证集上影响 mAP50 的图片（漏检和误检）
# 
# 使用方法:
#   bash tools/analysis/run_analyze_map50.sh
#

set -e

# 配置
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_64.pth"
OUTPUT_DIR="data/LLVIP/map50"
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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "mAP50 Error Analysis"
echo "=============================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Score threshold: $SCORE_THR"
echo "IoU threshold: $IOU_THR"
echo "=============================================="

# 运行分析
python tools/analysis/analyze_map50_errors.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --score-thr "$SCORE_THR" \
    --iou-thr "$IOU_THR" \
    --device "$DEVICE"

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
