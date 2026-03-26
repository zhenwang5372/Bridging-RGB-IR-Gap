# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream Multi-Modal YOLO Backbone with Text-Guided Fusion V2
# 
# ==================== V2 版本核心改进 ====================
#
# 相比 V1 的改进：
# 1. 支持返回 S_map 中间结果，用于辅助损失计算
# 2. 提供 get_s_maps() 方法获取各尺度的 S_map
#
# ==================== 使用场景 ====================
#
# 配合 DualStreamYOLOWorldDetectorV2 使用，支持 mask 监督损失
#
# ==================== 架构说明 ====================
#
# - RGB Stream: YOLOv8 CSPDarknet
# - IR Stream: LiteFFTIRBackbone
# - Fusion: TextGuidedRGBIRFusionV5（方案二：文本引导的逐类哈达姆门控）
# - Text Model: CLIP Text Encoder

from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithTextGuidedFusionV2(BaseModule):
    """
    带文本引导融合的双流多模态 YOLO Backbone V2
    
    相比 V1 的改进：
    1. 支持返回 S_map 中间结果，用于辅助损失计算
    2. 提供 get_s_maps() 方法获取各尺度的 S_map
    
    架构流程：
        Input: RGB Image + IR Image + Text
          ↓
        RGB Backbone (YOLOv8) → RGB Features (P3, P4, P5)
        IR Backbone (LiteFFT)  → IR Features (P3, P4, P5)
        Text Model (CLIP)      → Text Features [B, N, 512]
          ↓
        Text-Guided Fusion (V5)
          ↓
        Fused Features (P3, P4, P5) → 送入 Neck
          ↓
        [新增] S_maps (P3, P4, P5) → 用于辅助损失
    
    Args:
        image_model (dict): RGB backbone 配置（YOLOv8CSPDarknet）
        ir_model (dict): IR backbone 配置（LiteFFTIRBackbone）
        text_guided_fusion (dict): 文本引导融合模块配置（TextGuidedRGBIRFusionV5）
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
        self._s_maps = None  # 存储 S_map 用于损失计算
        
        # Build RGB image backbone
        self.image_model = MODELS.build(image_model)
        
        # Build IR backbone
        self.ir_model = MODELS.build(ir_model)
        
        # Build text-guided fusion module (V5)
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
        
        # 文本引导融合
        if txt_feats is not None:
            # text_model 可能返回 (txt_feats, text_mask) tuple
            # 需要提取实际的 text features tensor
            if isinstance(txt_feats, tuple):
                txt_feats_tensor = txt_feats[0]  # 第一个元素是 txt_feats [B, N, 512]
            else:
                txt_feats_tensor = txt_feats
            
            fused_feats = self.text_guided_fusion(rgb_feats, ir_feats, txt_feats_tensor)
            
            # V2 新增：存储 S_map
            if hasattr(self.text_guided_fusion, 'get_s_maps'):
                self._s_maps = self.text_guided_fusion.get_s_maps()
        else:
            # 如果没有文本特征，直接返回 RGB 特征
            fused_feats = rgb_feats
            self._s_maps = None
        
        return fused_feats, txt_feats
    
    def get_s_maps(self) -> Optional[List[torch.Tensor]]:
        """
        获取各尺度的 S_map（用于辅助损失）
        
        Returns:
            s_maps: List[Tensor], 每个元素形状为 [B, 1, H, W]
        """
        return self._s_maps
    
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
            if isinstance(txt_feats, tuple):
                txt_feats = txt_feats[0]
            fused_feats = self.text_guided_fusion(rgb_feats, ir_feats, txt_feats)
            
            # V2 新增：存储 S_map
            if hasattr(self.text_guided_fusion, 'get_s_maps'):
                self._s_maps = self.text_guided_fusion.get_s_maps()
        else:
            # 否则返回 RGB 特征
            fused_feats = rgb_feats
            self._s_maps = None
        
        return fused_feats
