#!/bin/bash
# 极简版Text-Only训练脚本
# 只更新Text，RGB和IR保持原样

cd /wangzhen/yolo-world-master/YOLO-World-master_2

# 训练命令
python tools/train.py \
    configs/custom_flir/yolow_v2_rgb_ir_flir_text_only.py \
    --work-dir work_dirs/text_only_fpn \
    --amp

# 说明:
# 1. RGB特征: 直接透传，不做任何修改
# 2. IR特征: 作为辅助信息，不做修改  
# 3. Text特征: 使用原始RGB+IR进行更新
# 4. 最简单、最稳定、最快收敛
#
# 预期效果:
# - 50 epoch: mAP_50 > 0.30
# - 100 epoch: mAP_50 > 0.35
#
# 如果效果好，说明:
# - Text动态调整是关键
# - IR/RGB作为稳定锚点更有效
# - 复杂的物理模型和RGB增强反而有害

