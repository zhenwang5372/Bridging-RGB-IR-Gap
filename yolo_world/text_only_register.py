# Copyright (c) Tencent Inc. All rights reserved.
# 极简版Text-Only Neck模块注册文件
# 
# 使用方法：
# custom_imports = dict(
#     imports=['yolo_world', 'yolo_world.text_only_register'],
#     allow_failed_imports=False
# )

# 导入极简版Neck模块
from .models.necks.text_only_update import TextOnlyUpdateNeck

# 导入必要的工具模块
from .models.necks.trimodal_utils import *
from .models.necks.rgb_ir_fusion_v2 import (
    LightweightCrossFusionV2,
    MultiLevelRGBIRFusionV2
)

# 导入Detector模块
from .models.detectors.dual_stream_yolo_world_v2 import (
    DualStreamMultiModalYOLOBackboneV2,
    DualStreamYOLOWorldDetectorV2
)

__all__ = [
    'TextOnlyUpdateNeck',
    'LightweightCrossFusionV2',
    'MultiLevelRGBIRFusionV2',
    'DualStreamMultiModalYOLOBackboneV2',
    'DualStreamYOLOWorldDetectorV2',
]

print("[TextOnly] Successfully registered text-only update neck modules!")
print("[TextOnly] RGB and IR are kept UNCHANGED, only Text is updated.")
print("[TextOnly] This is the SIMPLEST and most STABLE version.")

