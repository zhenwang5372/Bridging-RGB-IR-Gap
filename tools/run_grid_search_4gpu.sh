#!/bin/bash
# Grid search using 4 GPUs in parallel
# A: 0.00 to 1.00, step 0.05 (21 values)
# B: 1.00 to 2.00, step 0.05 (21 values)
# Total: 21 x 21 = 441 combinations

CONFIG="configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py"
CHECKPOINT="work_dirs/LLVIP/no_update/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_50.pth"
WORK_DIR="work_dirs/grid_search_results"

# Create work directory
mkdir -p $WORK_DIR

echo "========================================"
echo "Starting Grid Search on 4 GPUs"
echo "========================================"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Work dir: $WORK_DIR"
echo ""
echo "A range: 0.00 to 1.00 (step 0.05, 21 values)"
echo "B range: 1.00 to 2.00 (step 0.05, 21 values)"
echo "Total combinations: 441"
echo ""
echo "GPU allocation:"
echo "  GPU 0: A = 0.00 ~ 0.25 (6 values x 21 B values = 126 tests)"
echo "  GPU 1: A = 0.30 ~ 0.50 (5 values x 21 B values = 105 tests)"
echo "  GPU 2: A = 0.55 ~ 0.75 (5 values x 21 B values = 105 tests)"
echo "  GPU 3: A = 0.80 ~ 1.00 (5 values x 21 B values = 105 tests)"
echo "========================================"
echo ""

# Launch 4 processes in parallel
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU 0 (A=0.00~0.25)..."
CUDA_VISIBLE_DEVICES=0 python tools/grid_search_fusion.py \
    $CONFIG $CHECKPOINT \
    --work-dir $WORK_DIR \
    --gpu-id 0 --a-start 0.00 --a-end 0.25 --a-step 0.05 \
    --b-start 1.00 --b-end 2.00 --b-step 0.05 \
    > ${WORK_DIR}/gpu0_stdout.log 2>&1 &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU 1 (A=0.30~0.50)..."
CUDA_VISIBLE_DEVICES=1 python tools/grid_search_fusion.py \
    $CONFIG $CHECKPOINT \
    --work-dir $WORK_DIR \
    --gpu-id 1 --a-start 0.30 --a-end 0.50 --a-step 0.05 \
    --b-start 1.00 --b-end 2.00 --b-step 0.05 \
    > ${WORK_DIR}/gpu1_stdout.log 2>&1 &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU 2 (A=0.55~0.75)..."
CUDA_VISIBLE_DEVICES=2 python tools/grid_search_fusion.py \
    $CONFIG $CHECKPOINT \
    --work-dir $WORK_DIR \
    --gpu-id 2 --a-start 0.55 --a-end 0.75 --a-step 0.05 \
    --b-start 1.00 --b-end 2.00 --b-step 0.05 \
    > ${WORK_DIR}/gpu2_stdout.log 2>&1 &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU 3 (A=0.80~1.00)..."
CUDA_VISIBLE_DEVICES=3 python tools/grid_search_fusion.py \
    $CONFIG $CHECKPOINT \
    --work-dir $WORK_DIR \
    --gpu-id 3 --a-start 0.80 --a-end 1.00 --a-step 0.05 \
    --b-start 1.00 --b-end 2.00 --b-step 0.05 \
    > ${WORK_DIR}/gpu3_stdout.log 2>&1 &

echo ""
echo "All 4 processes launched!"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${WORK_DIR}/gpu0_stdout.log"
echo "  tail -f ${WORK_DIR}/gpu1_stdout.log"
echo "  tail -f ${WORK_DIR}/gpu2_stdout.log"
echo "  tail -f ${WORK_DIR}/gpu3_stdout.log"
echo ""
echo "Or watch GPU usage: watch -n 1 nvidia-smi"
echo ""

# Wait for all background processes to finish
echo "Waiting for all processes to complete..."
wait

echo ""
echo "========================================"
echo "All processes completed!"
echo "========================================"
echo ""

# Merge results
echo "Merging results..."
python tools/merge_grid_search_results.py --work-dir $WORK_DIR

echo "Done!"
