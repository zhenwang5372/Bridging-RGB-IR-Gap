# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone)
from .lite_fft_ir_backbone import LiteFFTIRBackbone
from .dual_stream_class_specific_backbone import DualStreamMultiModalYOLOBackboneWithClassSpecific
from .dual_stream_class_specific_backbone_v2 import DualStreamMultiModalYOLOBackboneWithClassSpecificV2

# Import IR_backbone V2 module to register classes
from .IR_backbone import (
    LiteFFTIRBackboneV2,
    LiteFFTIRBackbonePreSEV2,
    LiteFFTIRBackbonePostSEV2,
    LiteDCTGhostIRBackboneV2,
)
from .ir_correction_rgb_fusion import (
    DualStreamMultiModalYOLOBackboneWithTextGuidedFusion,
)

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'LiteFFTIRBackbone',
    'DualStreamMultiModalYOLOBackboneWithClassSpecific',
    'DualStreamMultiModalYOLOBackboneWithClassSpecificV2',
    # V2 IR Backbones
    'LiteFFTIRBackboneV2',
    'LiteFFTIRBackbonePreSEV2',
    'LiteFFTIRBackbonePostSEV2',
    'LiteDCTGhostIRBackboneV2',
    # Text-Guided Fusion Backbone (Scheme 2)
    'DualStreamMultiModalYOLOBackboneWithTextGuidedFusion',
]
