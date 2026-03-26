# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream Multi-Modal YOLO Backbone with Text-Guided Fusion (Scheme 2)
# 
# 架构说明：
# - RGB Stream: YOLOv8 CSPDarknet
# - IR Stream: LiteFFTIRBackbone
# - Fusion: TextGuidedRGBIRFusion（方案二：文本引导的逐类哈达姆门控）
# - Text Model: CLIP Text Encoder

from typing import List, Tuple, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithTextGuidedFusion(BaseModule):
    """
    带文本引导融合的双流多模态 YOLO Backbone（方案二）
    
    与方案一的区别：
    - 方案一: IR纠错（ir_correction）+ RGB-IR融合（fusion_module）分离
    - 方案二: 统一的文本引导融合（text_guided_fusion），同时完成语义引导和融合
    
    架构流程：
        Input: RGB Image + IR Image + Text
          ↓
        RGB Backbone (YOLOv8) → RGB Features (P3, P4, P5)
        IR Backbone (LiteFFT)  → IR Features (P3, P4, P5)
        Text Model (CLIP)      → Text Features [B, N, 512]
          ↓
        Text-Guided Fusion (方案二)
          - Step 1: 文本引导的语义激活
          - Step 2: 类别重要性计算
          - Step 3: 加权哈达姆对齐
          - Step C: 门控生成
          - Step D: 特征融合
          ↓
        Fused Features (P3, P4, P5) → 送入 Neck
    
    Args:
        image_model (dict): RGB backbone 配置（YOLOv8CSPDarknet）
        ir_model (dict): IR backbone 配置（LiteFFTIRBackbone）
        text_guided_fusion (dict): 文本引导融合模块配置（TextGuidedRGBIRFusion）
        text_model (dict, optional): 文本模型配置（HuggingCLIPLanguageBackbone）
        frozen_stages (int): 冻结阶段数，默认 -1（不冻结）
        with_text_model (bool): 是否使用文本模型，默认 True
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        image_model: dict,
        ir_model: dict,
        text_guided_fusion: dict,
        text_model: dict = None,
        frozen_stages: int = -1,
        with_text_model: bool = True,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # Build RGB image backbone
        self.image_model = MODELS.build(image_model)
        
        # Build IR backbone
        self.ir_model = MODELS.build(ir_model)
        
        # Build text-guided fusion module (方案二)
        self.text_guided_fusion = MODELS.build(text_guided_fusion)
        
        # Build text model
        if self.with_text_model and text_model is not None:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze the parameters of specified stages."""
        if self.frozen_stages >= 0:
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
    
    def forward(
        self,
        image: torch.Tensor,
        text: List[List[str]],
        img_ir: torch.Tensor = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            image: RGB 图像 [B, 3, H, W]
            text: 文本列表 List[List[str]]
            img_ir: IR 图像 [B, 3, H, W]
        
        Returns:
            fused_feats: 融合后的特征 (P3, P4, P5)
            txt_feats: 文本特征 [B, N, 512] 或 (txt_feats, text_mask)
        """
        # 如果没有提供 IR 图像，使用 RGB 图像
        if img_ir is None:
            img_ir = image
        
        # 提取 RGB 特征
        rgb_feats = self.image_model(image)  # (P3, P4, P5)
        
        # 提取 IR 特征
        ir_feats = self.ir_model(img_ir)     # (P3, P4, P5)
        
        # 提取文本特征
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)  # [B, N, 512] or (txt_feats, text_mask)
        else:
            txt_feats = None
        
        # 文本引导融合（方案二）
        if txt_feats is not None:
            # text_model 可能返回 (txt_feats, text_mask) tuple
            # 需要提取实际的 text features tensor
            if isinstance(txt_feats, tuple):
                txt_feats_tensor = txt_feats[0]  # 第一个元素是 txt_feats [B, N, 512]
            else:
                txt_feats_tensor = txt_feats
            
            fused_feats = self.text_guided_fusion(rgb_feats, ir_feats, txt_feats_tensor)
        else:
            # 如果没有文本特征，直接返回 RGB 特征
            # （理论上不应该发生，因为方案二依赖文本）
            fused_feats = rgb_feats
        
        return fused_feats, txt_feats
    
    def forward_text(self, text: List[List[str]]):
        """
        Forward text only.
        
        Args:
            text: 文本列表 List[List[str]]
        
        Returns:
            txt_feats: 文本特征 [B, N, 512] 或 (txt_feats, text_mask)
        """
        assert self.with_text_model and self.text_model is not None, \
            "forward_text() requires a text model"
        return self.text_model(text)
    
    def forward_image(
        self, 
        image: torch.Tensor, 
        img_ir: torch.Tensor = None,
        text: List[List[str]] = None
    ):
        """
        Forward image only.
        
        Args:
            image: RGB 图像 [B, 3, H, W]
            img_ir: IR 图像 [B, 3, H, W]
            text: 文本列表 List[List[str]]（用于融合）
        
        Returns:
            fused_feats: 融合后的特征 (P3, P4, P5)
        """
        if img_ir is None:
            img_ir = image
        
        # 提取特征
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        
        # 如果提供了文本，进行融合
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
            fused_feats = self.text_guided_fusion(rgb_feats, ir_feats, txt_feats)
        else:
            # 否则返回 RGB 特征
            fused_feats = rgb_feats
        
        return fused_feats
