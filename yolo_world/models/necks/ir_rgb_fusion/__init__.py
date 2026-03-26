# Copyright (c) Tencent Inc. All rights reserved.
# M_err Guided RGB-IR Fusion Modules

# V1: 用 M_err 替代 attention_map，仍然 concat IR 特征
from .Merr_attentionmapV1 import (
    MerrGuidedCrossFusionV1,
    MultiLevelMerrGuidedFusionV1
)

# V2: 只用 M_err 增强 RGB，完全不使用 IR 特征
from .Merr_attentionmapV2 import (
    MerrGuidedRGBEnhancementV2,
    MultiLevelMerrGuidedFusionV2
)

# V3: M_err + 双阈值硬注意力（Hard Attention with Dual Thresholds）
from .Merr_attentionmapV3 import (
    MerrGuidedRGBEnhancementV3,
    MultiLevelMerrGuidedFusionV3
)

__all__ = [
    # V1: M_err + IR concat
    'MerrGuidedCrossFusionV1',
    'MultiLevelMerrGuidedFusionV1',
    # V2: M_err only (no IR)
    'MerrGuidedRGBEnhancementV2',
    'MultiLevelMerrGuidedFusionV2',
    # V3: M_err + Hard Attention
    'MerrGuidedRGBEnhancementV3',
    'MultiLevelMerrGuidedFusionV3',
]
