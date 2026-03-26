# Copyright (c) Tencent Inc. All rights reserved.
# 简化版三模态Neck模块注册文件
# 
# 使用方法：
# custom_imports = dict(
#     imports=['yolo_world', 'yolo_world.simplified_register'],
#     allow_failed_imports=False
# )

# 导入简化版Neck模块
from .models.necks.simplified_trimodal import SimplifiedTriModalNeck

# 导入必要的工具模块（从原始位置）
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
    'SimplifiedTriModalNeck',
    'LightweightCrossFusionV2',
    'MultiLevelRGBIRFusionV2',
    'DualStreamMultiModalYOLOBackboneV2',
    'DualStreamYOLOWorldDetectorV2',
]

print("[Simplified] Successfully registered simplified trimodal neck modules!")
print("[Simplified] IR update is SKIPPED, only RGB and Text are updated.")

