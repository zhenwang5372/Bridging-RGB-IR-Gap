# Copyright (c) Tencent Inc. All rights reserved.
# 三模态Neck模块注册文件
# 在配置文件中通过 custom_imports 导入此模块即可注册所有新模块
# 
# 使用方法：
# custom_imports = dict(
#     imports=['yolo_world', 'yolo_world.trimodal_register'],
#     allow_failed_imports=False
# )

# 导入新的Neck模块（会自动注册到MODELS）
from .models.necks.rgb_ir_fusion_v2 import (
    LightweightCrossFusionV2,
    MultiLevelRGBIRFusionV2
)
from .models.necks.trimodal_phased_neck import TriModalPhasedNeck
from .models.necks.ir_correction import IRCorrectionModule
from .models.necks.rgb_enhancement import RGBEnhancementModule
from .models.necks.text_update import TextUpdateModule
from .models.necks.text_update_multiscale import TextUpdateMultiScale
from .models.necks.text_update_fusion_first import TextUpdateFusionFirst

# 导入新的Detector模块
from .models.detectors.dual_stream_yolo_world_v2 import (
    DualStreamMultiModalYOLOBackboneV2,
    DualStreamYOLOWorldDetectorV2
)

__all__ = [
    'LightweightCrossFusionV2',
    'MultiLevelRGBIRFusionV2',
    'TriModalPhasedNeck',
    'IRCorrectionModule',
    'RGBEnhancementModule',
    'TextUpdateModule',
    'TextUpdateMultiScale',
    'TextUpdateFusionFirst',
    'DualStreamMultiModalYOLOBackboneV2',
    'DualStreamYOLOWorldDetectorV2',
]

print("[TriModal] Successfully registered trimodal neck modules!")

