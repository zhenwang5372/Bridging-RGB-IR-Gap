# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream Multi-Modal YOLO Backbone with Class-Specific Features V2

import torch
import torch.nn as nn
from typing import Tuple, Union, List
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithClassSpecificV2(BaseModule):
    """
    带类别特定特征生成的双流多模态YOLO Backbone V2
    
    改进：
        - 阶段4: RGB Enhancement使用Fused作为Value
        - 阶段5: Text Update完全独立，直接从Fused提取
    
    工作流程:
        1. RGB Backbone提取RGB特征
        2. IR Backbone提取IR特征
        3. Text Model提取文本特征
        4. IR Correction纠正IR特征
        5. RGB-IR Fusion融合特征
        6. ⭐ RGB Enhancement生成类别特定特征 (Q=Text, K=Fused, V=Fused)
        7. ⭐ Text Update更新文本特征 (独立Cross-Attention from Fused)
    
    Args:
        image_model (dict): RGB backbone配置
        ir_model (dict): IR backbone配置
        fusion_module (dict): RGB-IR融合模块配置
        text_model (dict): 文本模型配置
        ir_correction (dict): IR纠错模块配置
        rgb_enhancement (dict): RGB增强模块配置 (V2)
        text_update (dict): Text更新模块配置 (V2)
        frozen_stages (int): 冻结阶段数
        with_text_model (bool): 是否使用文本模型
        init_cfg (dict): 初始化配置
    """
    
    def __init__(
        self,
        image_model: dict,
        ir_model: dict,
        fusion_module: dict,
        text_model: dict = None,
        ir_correction: dict = None,
        rgb_enhancement: dict = None,
        text_update: dict = None,
        frozen_stages: int = -1,
        with_text_model: bool = True,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # 构建原有组件
        self.image_model = MODELS.build(image_model)
        self.ir_model = MODELS.build(ir_model)
        self.fusion_module = MODELS.build(fusion_module)
        
        if with_text_model:
            self.text_model = MODELS.build(text_model)
        
        if ir_correction is not None:
            self.ir_correction = MODELS.build(ir_correction)
        else:
            self.ir_correction = None
        
        # ⭐ 构建新增组件 (V2版本)
        if rgb_enhancement is not None:
            self.rgb_enhancement = MODELS.build(rgb_enhancement)
        else:
            self.rgb_enhancement = None
        
        if text_update is not None:
            self.text_update = MODELS.build(text_update)
        else:
            self.text_update = None
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """冻结指定阶段"""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, f'stage{i}', None)
                if m is not None:
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        texts: Union[List[str], torch.Tensor],
        x_ir: torch.Tensor = None,
        gt_labels: List[torch.Tensor] = None,  # ⭐ 新增：用于 V5 的 class mask
    ) -> Tuple:
        """
        Args:
            x_rgb: RGB图像 [B, 3, H, W]
            texts: 文本输入
            x_ir: IR图像 [B, 3, H, W]
            gt_labels: List of [N_i] - 每张图片的 GT 类别索引（训练时用于 V5 mask）
        
        Returns:
            rgb_class_specific: List of [B, num_cls, C, H, W]
            text_updated: (text_updated, None) tuple for compatibility
        """
        # 1. 提取RGB特征
        rgb_feats = self.image_model(x_rgb)
        # Tuple of [P3, P4, P5]
        
        # 2. 提取IR特征
        ir_feats = self.ir_model(x_ir)
        # Tuple of [P3, P4, P5]
        
        # 3. 提取Text特征
        if self.with_text_model:
            text_output = self.text_model(texts)
            # [num_cls, text_dim] 或 (text_feats, txt_masks)
            
            # 处理可能的tuple返回
            if isinstance(text_output, tuple):
                text_feats, txt_masks = text_output
            else:
                text_feats = text_output
                txt_masks = None
        else:
            text_feats = texts
            txt_masks = None
        
        # 4. IR Correction
        if self.ir_correction is not None:
            rgb_feats, ir_corrected_feats = self.ir_correction(
                rgb_feats, ir_feats, text_feats
            )
        else:
            ir_corrected_feats = ir_feats
        
        # 5. RGB-IR Fusion
        fused_feats = self.fusion_module(rgb_feats, ir_corrected_feats)
        # Tuple of [P3, P4, P5]
        
        # 6. ⭐ RGB Enhancement V2 (阶段4)
        # 改进: 使用Fused作为Value
        if self.rgb_enhancement is not None:
            rgb_class_specific = self.rgb_enhancement(
                rgb_feats=rgb_feats,  # 保留接口兼容，但V2版本不使用
                fused_feats=fused_feats,
                text_feats=text_feats
            )
            # List of [B, num_cls, C, H, W]
        else:
            # 如果没有enhancement，返回原始fused特征
            rgb_class_specific = list(fused_feats)
        
        # 7. ⭐ Text Update (阶段5)
        # V3/V4: 不需要 gt_labels
        # V5: 使用 gt_labels 创建 class_mask
        if self.text_update is not None:
            # 检查 text_update 是否支持 gt_labels 参数（只检查一次）
            if not hasattr(self, '_text_update_supports_gt_labels'):
                import inspect
                sig = inspect.signature(self.text_update.forward)
                self._text_update_supports_gt_labels = 'gt_labels' in sig.parameters
            
            if self._text_update_supports_gt_labels:
                # V5: 支持 gt_labels
                text_updated = self.text_update(
                    fused_feats=fused_feats,
                    text_feats=text_feats,
                    gt_labels=gt_labels
                )
            else:
                # V3/V4: 不支持 gt_labels
                text_updated = self.text_update(
                    fused_feats=fused_feats,
                    text_feats=text_feats
                )
            # [B, num_cls, text_dim] or [num_cls, text_dim]
        else:
            text_updated = text_feats
        
        # 返回格式：(img_feats, (txt_feats, txt_masks), fused_feats)
        # 为了兼容detector，img_feats使用rgb_class_specific
        # ⭐ 新增：返回fused_feats用于aggregator中与聚合后的特征融合
        return rgb_class_specific, (text_updated, txt_masks), fused_feats
    
    def forward_image(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        仅提取图像特征（用于推理时文本特征已缓存的情况）
        
        Args:
            x_rgb: RGB图像 [B, 3, H, W]
            x_ir: IR图像 [B, 3, H, W]
        
        Returns:
            rgb_feats: RGB特征 [P3, P4, P5]
        """
        # 1. 提取RGB特征
        rgb_feats = self.image_model(x_rgb)
        
        if x_ir is None or self.ir_model is None:
            # 如果没有IR，直接返回RGB特征
            return rgb_feats
        
        # 2. 提取IR特征
        ir_feats = self.ir_model(x_ir)
        
        # 3. 简单融合RGB和IR（不需要text）
        if self.fusion_module is not None:
            fused_feats = self.fusion_module(rgb_feats, ir_feats)
            return fused_feats
        else:
            return rgb_feats

