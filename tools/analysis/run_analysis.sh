#!/bin/bash
# 检测结果分析脚本
#
# 使用方法:
#   1. 先运行测试生成预测结果
#   2. 然后运行此脚本分析

# 配置
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_28.pth"
GT_ANN="data/LLVIP/coco_annotations/test.json"
OUTPUT_DIR="work_dirs/LLVIP/analysis"

# 环境设置
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/mmyolo
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Step 1: Run inference to generate predictions"
echo "============================================"

# 运行测试，保存预测结果
python tools/test.py ${CONFIG} ${CHECKPOINT} \
    --work-dir ${OUTPUT_DIR} \
    --out ${OUTPUT_DIR}/predictions.pkl

echo ""
echo "============================================"
echo "Step 2: Convert predictions to COCO format"
echo "============================================"

# 转换为 COCO 格式 (如果需要)
# mmdet 的 test.py 会自动生成 COCO 格式的结果

echo ""
echo "============================================"
echo "Step 3: Analyze results"
echo "============================================"

# 如果有 COCO 格式的预测结果
if [ -f "${OUTPUT_DIR}/predictions.bbox.json" ]; then
    python tools/analysis/analyze_detection_results.py \
        --gt-ann ${GT_ANN} \
        --dt-ann ${OUTPUT_DIR}/predictions.bbox.json \
        --output-dir ${OUTPUT_DIR}
else
    echo "Prediction file not found. Please check the test output."
    echo "Expected: ${OUTPUT_DIR}/predictions.bbox.json"
fi

echo ""
echo "============================================"
echo "Analysis complete!"
echo "Check results in: ${OUTPUT_DIR}"
echo "============================================"
