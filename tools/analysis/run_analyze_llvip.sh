#!/bin/bash
# 分析 LLVIP 检测结果，找出影响 mAP50 和 mAP75 的图像

# ======================== 配置 ========================
# Ground Truth 标注文件
GT_ANN="data/LLVIP/coco_annotations/test.json"

# 检测结果文件（需要先运行测试生成）
# 检测结果会保存在 work_dir 下，文件名类似：results.bbox.json
DT_ANN="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/results.bbox.json"

# 输出目录
OUTPUT_DIR="work_dirs/LLVIP/analysis"

# ======================== 环境设置 ========================
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
export PYTHONPATH=$PYTHONPATH:.

# ======================== Step 1: 生成检测结果 ========================
echo "============================================"
echo "Step 1: 生成检测结果文件"
echo "============================================"

CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_28.pth"
WORK_DIR="work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu"

if [ ! -f "${DT_ANN}" ]; then
    echo "检测结果文件不存在，正在生成..."
    
    python tools/test.py ${CONFIG} ${CHECKPOINT} \
        --work-dir ${WORK_DIR} \
        --cfg-options \
            test_evaluator.outfile_prefix=${WORK_DIR}/results
    
    echo "✅ 检测结果已生成: ${DT_ANN}"
else
    echo "✅ 检测结果文件已存在: ${DT_ANN}"
fi

# ======================== Step 2: 分析结果 ========================
echo ""
echo "============================================"
echo "Step 2: 分析检测结果"
echo "============================================"

python tools/analysis/analyze_detection_results.py \
    --gt-ann ${GT_ANN} \
    --dt-ann ${DT_ANN} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "============================================"
echo "分析完成！"
echo "============================================"
echo ""
echo "生成的文件:"
echo "  1. ${OUTPUT_DIR}/localization_problem_images.json"
echo "     → Top 20 定位不准的图像 (详细信息)"
echo ""
echo "  2. ${OUTPUT_DIR}/localization_problem_images_list.txt"
echo "     → Top 50 定位不准的图像 (文件名列表)"
echo ""
echo "  3. ${OUTPUT_DIR}/detection_problem_images.json"
echo "     → 漏检问题图像 (recall@50 < 0.5)"
echo ""
echo "  4. ${OUTPUT_DIR}/all_image_metrics.json"
echo "     → 所有图像的指标 (完整数据)"
echo ""
echo "============================================"
