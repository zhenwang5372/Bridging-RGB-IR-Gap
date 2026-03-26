# Copyright (c) Tencent Inc. All rights reserved.
"""
IR Backbone V2/V3 Module

V2 版本改动：
1. base_channels 默认值: 32 → 64
2. Spatial branch: Depthwise Conv → Standard Conv (更强的特征提取能力)
3. 所有类名添加 V2 后缀，避免与原版本注册冲突

V3 版本改动：
1. base_channels 默认值: 64 (与 V2 相同，让 IR 和 RGB 通道对齐)
2. Spatial branch: 恢复使用 Depthwise Separable Conv (参照 V1，参数效率更高)
3. 所有类名添加 V3 后缀
"""

from .lite_fft_ir_backboneV2 import (
    SELayerV2,
    SpectralBlockV2,
    SpectralBlockPreSEV2,
    SpectralBlockPostSEV2,
    LiteFFTIRBackboneV2,
    LiteFFTIRBackbonePreSEV2,
    LiteFFTIRBackbonePostSEV2,
)

from .lite_dct_ghost_ir_backbone_v2 import LiteDCTGhostIRBackboneV2

from .lite_fft_ir_backboneV3 import (
    SELayerV3,
    SpectralBlockV3,
    SpectralBlockPreSEV3,
    SpectralBlockPostSEV3,
    LiteFFTIRBackboneV3,
    LiteFFTIRBackbonePreSEV3,
    LiteFFTIRBackbonePostSEV3,
)

__all__ = [
    # V2
    'SELayerV2',
    'SpectralBlockV2',
    'SpectralBlockPreSEV2',
    'SpectralBlockPostSEV2',
    'LiteFFTIRBackboneV2',
    'LiteFFTIRBackbonePreSEV2',
    'LiteFFTIRBackbonePostSEV2',
    # V3
    'SELayerV3',
    'SpectralBlockV3',
    'SpectralBlockPreSEV3',
    'SpectralBlockPostSEV3',
    'LiteFFTIRBackboneV3',
    'LiteFFTIRBackbonePreSEV3',
    'LiteFFTIRBackbonePostSEV3',
    # DCT Ghost V2
    'LiteDCTGhostIRBackboneV2',
]
