#!/bin/bash
# Quick test script for fusion coefficient combinations
# This script modifies the fusion code and runs tests

FUSION_FILE="yolo_world/models/necks/rgb_ir_fusion.py"
CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_50.pth"
WORK_DIR="work_dirs/grid_search_results"
RESULTS_FILE="${WORK_DIR}/quick_test_results.txt"

# Create work directory
mkdir -p $WORK_DIR

# Backup original file
cp $FUSION_FILE ${FUSION_FILE}.backup

# RGB coefficients and fused multipliers to test
RGB_COEFS=(0.5 0.8 1.0 1.2 1.5)
FUSED_MULTS=(0.5 1.0 1.5 2.0 3.0)

# Initialize results file
echo "=== Fusion Coefficient Grid Search Results ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Checkpoint: $CHECKPOINT" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "rgb_coef | fused_mult | mAP50" >> $RESULTS_FILE
echo "---------|------------|------" >> $RESULTS_FILE

# Function to modify fusion coefficients
modify_fusion() {
    local rgb_coef=$1
    local fused_mult=$2
    
    # Replace the output line in the fusion file
    sed -i "s/output = .* \* x_rgb + .* \* self.gamma \* fused/output = ${rgb_coef} * x_rgb + ${fused_mult} * self.gamma * fused/" $FUSION_FILE
    sed -i "s/output = .* \* x_rgb + self.gamma \* fused/output = ${rgb_coef} * x_rgb + ${fused_mult} * self.gamma * fused/" $FUSION_FILE
}

# Run tests
for rgb_coef in "${RGB_COEFS[@]}"; do
    for fused_mult in "${FUSED_MULTS[@]}"; do
        echo ""
        echo "=============================================="
        echo "Testing: rgb_coef=${rgb_coef}, fused_mult=${fused_mult}"
        echo "=============================================="
        
        # Modify the fusion file
        modify_fusion $rgb_coef $fused_mult
        
        # Show the modified line
        echo "Modified line:"
        grep "output = " $FUSION_FILE | head -1
        
        # Run test
        python tools/test.py \
            $CONFIG \
            $CHECKPOINT \
            --work-dir ${WORK_DIR}/rgb${rgb_coef}_fused${fused_mult} \
            2>&1 | tee ${WORK_DIR}/log_rgb${rgb_coef}_fused${fused_mult}.txt
        
        # Extract mAP50 from log (adjust grep pattern based on your output format)
        MAP50=$(grep -oP "bbox_mAP_50: \K[0-9.]+" ${WORK_DIR}/log_rgb${rgb_coef}_fused${fused_mult}.txt | tail -1)
        
        if [ -z "$MAP50" ]; then
            MAP50="N/A"
        fi
        
        echo "${rgb_coef}     | ${fused_mult}        | ${MAP50}" >> $RESULTS_FILE
        echo "Result: rgb_coef=${rgb_coef}, fused_mult=${fused_mult}, mAP50=${MAP50}"
    done
done

# Restore original file
cp ${FUSION_FILE}.backup $FUSION_FILE

echo ""
echo "=============================================="
echo "Grid search complete!"
echo "Results saved to: $RESULTS_FILE"
echo "Original fusion file restored from backup"
echo "=============================================="
cat $RESULTS_FILE
