#!/bin/bash

# 运行脚本：计算排除特定图片后的mAP
# 
# 使用方法：
# 1. 修改下面的路径配置
# 2. 运行: bash run_calculate_map.sh

# ==================== 配置区域 ====================

# 日志文件路径
LOG_FILE="tools/analysis/with_text_update/map50_analysis_report_text_update.log"

# PKL文件路径（需要根据实际情况修改）
PKL_SCORE0="work_dirs/analysis_results_text_update_score0.pkl"
PKL_SCORE0001="work_dirs/analysis_results_text_update_score0001.pkl"

# COCO标注文件路径
ANN_FILE="data/LLVIP/coco_annotations/test.json"

# 输出结果文件路径
OUTPUT_FILE="tools/analysis/with_text_update/filtered_map_results.txt"

# ==================== 运行脚本 ====================

echo "========================================"
echo "计算排除特定图片后的mAP"
echo "========================================"
echo ""
echo "配置信息："
echo "  日志文件: $LOG_FILE"
echo "  PKL (score=0): $PKL_SCORE0"
echo "  PKL (score=0.001): $PKL_SCORE0001"
echo "  标注文件: $ANN_FILE"
echo "  输出文件: $OUTPUT_FILE"
echo ""

# 检查文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件不存在: $LOG_FILE"
    exit 1
fi

if [ ! -f "$PKL_SCORE0" ]; then
    echo "错误: PKL文件不存在: $PKL_SCORE0"
    echo "请修改脚本中的 PKL_SCORE0 变量"
    exit 1
fi

if [ ! -f "$PKL_SCORE0001" ]; then
    echo "错误: PKL文件不存在: $PKL_SCORE0001"
    echo "请修改脚本中的 PKL_SCORE0001 变量"
    exit 1
fi

if [ ! -f "$ANN_FILE" ]; then
    echo "错误: 标注文件不存在: $ANN_FILE"
    exit 1
fi

# 运行Python脚本
python tools/analysis/with_text_update/calculate_map_after_filtering.py \
    --log-file "$LOG_FILE" \
    --pkl-score0 "$PKL_SCORE0" \
    --pkl-score0001 "$PKL_SCORE0001" \
    --ann-file "$ANN_FILE" \
    --output "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
