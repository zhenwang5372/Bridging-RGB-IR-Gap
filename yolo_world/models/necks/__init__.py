# Copyright (c) Tencent Inc. All rights reserved.
from .yolo_world_pafpn import YOLOWorldPAFPN, YOLOWorldDualPAFPN
from .rgb_ir_fusion import LightweightCrossFusion, MultiLevelRGBIRFusion
from .rgb_ir_fusion_v3 import LightweightCrossFusionV3, MultiLevelRGBIRFusionV3
from .simple_channel_align import SimpleChannelAlign, NoNeckPassThrough
from .text_guided_ir_correction import (
    TextGuidedIRCorrection,
    DualStreamMultiModalYOLOBackboneWithCorrection,
    TextGuidedIRCorrectionV2,
    DualStreamMultiModalYOLOBackboneWithCorrectionV2,
    TextGuidedIRCorrectionV3,
    DualStreamMultiModalYOLOBackboneWithCorrectionV3,
    TextGuidedIRCorrectionV4,
    DualStreamMultiModalYOLOBackboneWithCorrectionV4,
    TextGuidedIRCorrectionV5,
    DualStreamMultiModalYOLOBackboneWithCorrectionV5,
    TextGuidedIRCorrectionV6,
    DualStreamMultiModalYOLOBackboneWithCorrectionV6,
)
from .ir_rgb_fusion import (
    MerrGuidedCrossFusionV1,
    MultiLevelMerrGuidedFusionV1,
    MerrGuidedRGBEnhancementV2,
    MultiLevelMerrGuidedFusionV2,
)
from .text_guided_rgb_enhancement import TextGuidedRGBEnhancement
from .text_guided_rgb_enhancement_v2 import TextGuidedRGBEnhancementV2
from .multiscale_text_update import MultiScaleTextUpdate
from .multiscale_text_update_v2 import MultiScaleTextUpdateV2
from .multiscale_text_update_v3 import MultiScaleTextUpdateV3
from .multiscale_text_update_v4 import MultiScaleTextUpdateV4
from .multiscale_text_update_v5 import MultiScaleTextUpdateV5
from .class_dimension_aggregator import ClassDimensionAggregator
from .ir_correction_rgb_fusion import (
    TextGuidedRGBIRFusion,
    SingleLevelTextGuidedFusion,
    TextGuidedRGBIRFusionV2,
    SingleLevelTextGuidedFusionV2,
    TextGuidedRGBIRFusionV3,
    SingleLevelTextGuidedFusionV3,
)

__all__ = [
    'YOLOWorldPAFPN', 'YOLOWorldDualPAFPN',
    'LightweightCrossFusion', 'MultiLevelRGBIRFusion',
    'LightweightCrossFusionV3', 'MultiLevelRGBIRFusionV3',
    'SimpleChannelAlign', 'NoNeckPassThrough',
    'TextGuidedIRCorrection',
    'DualStreamMultiModalYOLOBackboneWithCorrection',
    'TextGuidedIRCorrectionV2',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV2',
    'TextGuidedIRCorrectionV3',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV3',
    'TextGuidedIRCorrectionV4',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV4',
    'TextGuidedIRCorrectionV5',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV5',
    'TextGuidedIRCorrectionV6',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV6',
    # M_err Guided Fusion
    'MerrGuidedCrossFusionV1',
    'MultiLevelMerrGuidedFusionV1',
    'MerrGuidedRGBEnhancementV2',
    'MultiLevelMerrGuidedFusionV2',
    # Text-Guided RGB-IR Fusion (Scheme 2)
    'TextGuidedRGBIRFusion',
    'SingleLevelTextGuidedFusion',
    'TextGuidedRGBIRFusionV2',
    'SingleLevelTextGuidedFusionV2',
    'TextGuidedRGBIRFusionV3',
    'SingleLevelTextGuidedFusionV3',
    # Others
    'TextGuidedRGBEnhancement',
    'TextGuidedRGBEnhancementV2',
    'MultiScaleTextUpdate',
    'MultiScaleTextUpdateV2',
    'MultiScaleTextUpdateV3',
    'MultiScaleTextUpdateV4',
    'MultiScaleTextUpdateV5',
    'ClassDimensionAggregator',
]
