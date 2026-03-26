# Copyright (c) Tencent Inc. All rights reserved.
# V2版本的__init__.py，包含三模态Neck模块
# 使用方法：将此文件重命名为__init__.py替换原文件，或在配置中通过custom_imports导入

from .yolo_world_pafpn import YOLOWorldPAFPN, YOLOWorldDualPAFPN
from .rgb_ir_fusion import LightweightCrossFusion, MultiLevelRGBIRFusion

# V2新增模块
from .rgb_ir_fusion_v2 import LightweightCrossFusionV2, MultiLevelRGBIRFusionV2
from .trimodal_phased_neck import TriModalPhasedNeck
from .ir_correction import IRCorrectionModule
from .rgb_enhancement import RGBEnhancementModule
from .text_update import TextUpdateModule
from .trimodal_utils import (
    AdditiveFusion,
    ChannelAttention,
    SpatialAttention,
    IRGuidedCBAM,
    CrossAttention,
    SmoothConv
)

__all__ = [
    # 原有模块
    'YOLOWorldPAFPN', 'YOLOWorldDualPAFPN',
    'LightweightCrossFusion', 'MultiLevelRGBIRFusion',
    # V2新增模块
    'LightweightCrossFusionV2', 'MultiLevelRGBIRFusionV2',
    'TriModalPhasedNeck',
    'IRCorrectionModule',
    'RGBEnhancementModule',
    'TextUpdateModule',
    'AdditiveFusion',
    'ChannelAttention',
    'SpatialAttention',
    'IRGuidedCBAM',
    'CrossAttention',
    'SmoothConv',
]

