# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module (V1, V2, V3, V4, V5, V6)

# V1 版本
from .text_guided_ir_correction_v1 import (
    TextGuidedIRCorrection,
    DualStreamMultiModalYOLOBackboneWithCorrection
)

# V2 版本
from .text_guided_ir_correction_v2 import (
    TextGuidedIRCorrectionV2,
    DualStreamMultiModalYOLOBackboneWithCorrectionV2
)

# V3 版本 (带 Alpha 日志输出)
from .text_guided_ir_correction_v3 import (
    TextGuidedIRCorrectionV3,
    DualStreamMultiModalYOLOBackboneWithCorrectionV3
)

# V4 版本 (Alpha 初始值 -0.3)
from .text_guided_ir_correction_v4 import (
    TextGuidedIRCorrectionV4,
    DualStreamMultiModalYOLOBackboneWithCorrectionV4
)

# V4 Plus 版本 (去掉 error_estimator，直接门控)
from .text_guided_ir_correction_v4_plus import (
    TextGuidedIRCorrectionV4Plus,
    DualStreamMultiModalYOLOBackboneWithCorrectionV4Plus
)

# V5 版本 (Cosine Similarity + Alpha -0.3)
from .text_guided_ir_correction_v5 import (
    TextGuidedIRCorrectionV5,
    DualStreamMultiModalYOLOBackboneWithCorrectionV5
)

# V5 Debug 版本 (添加权重相似度分析)
from .text_guided_ir_correction_v5_debug import (
    TextGuidedIRCorrectionV5Debug,
    DualStreamMultiModalYOLOBackboneWithCorrectionV5Debug
)

# V6 版本 (输出 M_err 用于融合模块)
from .text_guided_ir_correction_v6 import (
    TextGuidedIRCorrectionV6,
    DualStreamMultiModalYOLOBackboneWithCorrectionV6
)

# V5_Plus 版本 (动态投影维度 + 余弦相似度)
from .text_guided_ir_correction_v5_plus import (
    TextGuidedIRCorrectionV5Plus,
    DualStreamMultiModalYOLOBackboneWithCorrectionV5Plus
)

# Softmax-G-Smap-SumLast 版本 (G求积归一化加卷积)
from .Softmax_G_Smap_Sumlast import (
    SoftmaxGSmapSumlast,
)

# Softmax-G-Smap-SumFirst 版本 (G加权先求和后求积)
from .Softmax_G_Smap_Sumfirst import (
    SoftmaxGSmapSumfirst,
)

# IR-Only Unsupported by RGB (G-weighted version)
from .IR_Only_Unsupported_by_RGB_G import (
    IR_Only_Unsupported_by_RGB_G,
    DualStreamMultiModalYOLOBackboneWithIROnlyUnsupportedByRGB_G,
)

# IR-Only Unsupported by RGB (Mean version)
from .IR_Only_Unsupported_by_RGB_mean import (
    IR_Only_Unsupported_by_RGB_mean,
    DualStreamMultiModalYOLOBackboneWithIROnlyUnsupportedByRGB_mean,
)

# IR RGB M_err M_cons (双分支：纠错 + 共识增强)
from .IR_RGB_Merr_Cons import (
    IR_RGB_Merr_Cons,
    DualStreamMultiModalYOLOBackboneWithIR_RGB_Merr_Cons,
)

# 消融实验: No Text Anchor (cross_attn / cosine_sim / full_cross_attn)
from .IR_RGB_CrossModal_NoText import (
    IR_RGB_CrossModal_CrossAttn,
    IR_RGB_CrossModal_CosineSim,
    IR_RGB_CrossModal_FullCrossAttn,
)

# 消融实验: Mcons/Mdis 分离 (only_mcons / only_mdis / fixed coeffs)
from .IR_RGB_Merr_Cons_Ablation import (
    IR_RGB_Merr_Cons_Ablation,
)

__all__ = [
    # V1
    'TextGuidedIRCorrection',
    'DualStreamMultiModalYOLOBackboneWithCorrection',
    # V2
    'TextGuidedIRCorrectionV2',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV2',
    # V3 (带 Alpha 日志)
    'TextGuidedIRCorrectionV3',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV3',
    # V4 (Alpha 初始值 -0.3)
    'TextGuidedIRCorrectionV4',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV4',
    # V4 Plus (去掉 error_estimator)
    'TextGuidedIRCorrectionV4Plus',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV4Plus',
    # V5 (Cosine Similarity + Alpha -0.3)
    'TextGuidedIRCorrectionV5',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV5',
    # V5 Debug (带权重相似度分析)
    'TextGuidedIRCorrectionV5Debug',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV5Debug',
    # V6 (输出 M_err)
    'TextGuidedIRCorrectionV6',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV6',
    # V5_Plus (动态投影维度)
    'TextGuidedIRCorrectionV5Plus',
    'DualStreamMultiModalYOLOBackboneWithCorrectionV5Plus',
    # Softmax-G-Smap-SumLast (G求积归一化加卷积)
    'SoftmaxGSmapSumlast',
    # Softmax-G-Smap-SumFirst (G加权先求和后求积)
    'SoftmaxGSmapSumfirst',
    # IR-Only Unsupported by RGB (G-weighted)
    'IR_Only_Unsupported_by_RGB_G',
    'DualStreamMultiModalYOLOBackboneWithIROnlyUnsupportedByRGB_G',
    # IR-Only Unsupported by RGB (Mean)
    'IR_Only_Unsupported_by_RGB_mean',
    'DualStreamMultiModalYOLOBackboneWithIROnlyUnsupportedByRGB_mean',
    # IR RGB M_err M_cons (双分支：纠错 + 共识增强)
    'IR_RGB_Merr_Cons',
    'DualStreamMultiModalYOLOBackboneWithIR_RGB_Merr_Cons',
    # 消融实验: No Text Anchor
    'IR_RGB_CrossModal_CrossAttn',
    'IR_RGB_CrossModal_CosineSim',
    'IR_RGB_CrossModal_FullCrossAttn',
    # 消融实验: Mcons/Mdis 分离
    'IR_RGB_Merr_Cons_Ablation',
]

