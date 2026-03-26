#!/bin/bash

# No-Neck Baseline训练脚本
# 完全跳过Neck，直接使用Fusion后的RGB特征 + 原始Text

cd /wangzhen/yolo-world-master/YOLO-World-master_2

export PYTHONPATH=/wangzhen/yolo-world-master/YOLO-World-master_2:$PYTHONPATH

# 训练命令
python tools/train.py \
    configs/custom_flir/yolow_v2_rgb_ir_flir_no_update.py \
    --work-dir work_dirs/no_neck_baseline \
    --amp

# 说明:
# 1. mm_neck=False: 不使用多模态Neck
# 2. SimpleChannelAlign: 只做通道对齐，不做任何融合
# 3. RGB特征来自Fusion模块输出（已融合IR信息）
# 4. Text特征直接使用CLIP编码的原始嵌入
# 5. Head中进行区域-文本相似度匹配

# 数据流:
# Backbone → Fusion(RGB+IR) → SimpleChannelAlign → Head(原始Text)

