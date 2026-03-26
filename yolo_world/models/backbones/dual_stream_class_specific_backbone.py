# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream Multi-Modal YOLO Backbone with Class-Specific Features

import torch
import torch.nn as nn
from typing import Tuple, Union, List
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithClassSpecific(BaseModule):
    """
    带类别特定特征生成的双流多模态YOLO Backbone
    
    在原有的IR Correction和Fusion基础上，新增:
        - 阶段2: Text-guided RGB Enhancement
        - 阶段3: Multi-scale Text Update
    
    工作流程:
        1. RGB Backbone提取RGB特征
        2. IR Backbone提取IR特征
        3. Text Model提取文本特征
        4. IR Correction纠正IR特征
        5. RGB-IR Fusion融合特征
        6. ⭐ RGB Enhancement生成类别特定特征
        7. ⭐ Text Update更新文本特征
    
    Args:
        image_model (dict): RGB backbone配置
        ir_model (dict): IR backbone配置
        fusion_module (dict): RGB-IR融合模块配置
        text_model (dict): 文本模型配置
        ir_correction (dict): IR纠错模块配置
        rgb_enhancement (dict): RGB增强模块配置 (新增)
        text_update (dict): Text更新模块配置 (新增)
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
        
        # ⭐ 构建新增组件
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
        x_ir: torch.Tensor = None
    ) -> Tuple:
        """
        Args:
            x_rgb: RGB图像 [B, 3, H, W]
            texts: 文本输入
            x_ir: IR图像 [B, 3, H, W]
        
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
        
        # 6. ⭐ RGB Enhancement (阶段2)
        if self.rgb_enhancement is not None:
            rgb_class_specific = self.rgb_enhancement(
                rgb_feats=rgb_feats,
                fused_feats=fused_feats,
                text_feats=text_feats
            )
            # List of [B, num_cls, C, H, W]
        else:
            # 如果没有enhancement，返回原始fused特征
            rgb_class_specific = list(fused_feats)
        
        # 7. ⭐ Text Update (阶段3)
        if self.text_update is not None:
            text_updated = self.text_update(
                rgb_class_specific=rgb_class_specific,
                text_feats=text_feats
            )
            # [num_cls, text_dim]
        else:
            text_updated = text_feats
        
        # 返回格式：(img_feats, (txt_feats, txt_masks))
        # 为了兼容detector，img_feats使用rgb_class_specific
        return rgb_class_specific, (text_updated, txt_masks)
    
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

