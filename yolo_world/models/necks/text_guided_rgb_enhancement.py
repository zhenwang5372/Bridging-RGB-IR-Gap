# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB Enhancement Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedRGBEnhancement(BaseModule):
    """
    Text-guided RGB Enhancement Module (阶段2)
    
    为每个尺度独立生成类别特定的RGB特征。
    
    工作流程（对每个尺度）:
        1. 使用Fused features计算Text-guided Attention
        2. 用Attention权重调制原始RGB特征
        3. 输出类别特定的特征 [B, num_cls, C, H, W]
    
    Args:
        rgb_channels (List[int]): RGB通道数 [P3, P4, P5]
        text_dim (int): Text embedding维度，默认512
        num_classes (int): 类别数，默认4
        d_k (int): Attention的key/query维度，默认128
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        d_k: int = 128,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.rgb_channels = rgb_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.d_k = d_k
        self.num_levels = len(rgb_channels)
        
        # 为每个尺度创建独立的模块
        self.level_modules = nn.ModuleList([
            SingleLevelRGBEnhancement(
                rgb_channels=ch,
                text_dim=text_dim,
                num_classes=num_classes,
                d_k=d_k
            )
            for ch in rgb_channels
        ])
    
    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        fused_feats: Tuple[torch.Tensor, ...],
        text_feats: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            rgb_feats: 原始RGB特征 (P3, P4, P5)
                每个: [B, C, H, W]
            fused_feats: Fusion后的特征 (P3, P4, P5)
                每个: [B, C, H, W]
            text_feats: Text embedding [num_cls, text_dim]
        
        Returns:
            rgb_class_specific: List of [B, num_cls, C, H, W]
        """
        rgb_class_specific = []
        
        for rgb_feat, fused_feat, module in zip(
            rgb_feats, fused_feats, self.level_modules
        ):
            rgb_cs = module(
                rgb_feat=rgb_feat,
                fused_feat=fused_feat,
                text_feat=text_feats
            )
            rgb_class_specific.append(rgb_cs)
        
        return rgb_class_specific


class SingleLevelRGBEnhancement(nn.Module):
    """
    单尺度的RGB增强模块
    
    核心机制:
        1. Text作为Query查询Fused features
        2. 生成每个类别的空间Attention map
        3. 用Attention调制原始RGB特征
    """
    
    def __init__(
        self,
        rgb_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        d_k: int = 128
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.d_k = d_k
        
        # Query projection: Text -> d_k
        self.query_proj = nn.Linear(text_dim, d_k)
        
        # Key projection: Fused features -> d_k
        self.key_conv = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        
        # 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.zeros_(self.key_conv.bias)
    
    def forward(
        self,
        rgb_feat: torch.Tensor,
        fused_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat: [B, C, H, W] 原始RGB特征
            fused_feat: [B, C, H, W] Fused特征
            text_feat: [num_cls, text_dim] Text embedding
        
        Returns:
            rgb_class_specific: [B, num_cls, C, H, W]
        """
        B, C, H, W = rgb_feat.shape
        
        # 处理text_feat的不同形状
        # 期望: [num_cls, text_dim]
        # 可能收到: [1, B, num_cls, text_dim] 或 [B, num_cls, text_dim] 或 [num_cls, text_dim]
        if text_feat.dim() == 4:
            # [1, B, num_cls, text_dim] -> [num_cls, text_dim]
            text_feat = text_feat[0, 0]  # 取第一个batch的第一个元素
        elif text_feat.dim() == 3:
            # [B, num_cls, text_dim] -> [num_cls, text_dim]
            text_feat = text_feat[0]  # 取第一个batch
        
        num_cls = text_feat.shape[0]
        
        # Step 1: 扩展Text到batch维度
        text_expanded = text_feat.unsqueeze(0).expand(B, -1, -1)
        # [B, num_cls, text_dim]
        
        # Step 2: 生成Query (来自Text)
        Q = self.query_proj(text_expanded)
        # [B, num_cls, d_k]
        
        # Step 3: 生成Key (来自Fused features)
        K = self.key_conv(fused_feat)
        # [B, d_k, H, W]
        
        # Step 4: Flatten spatial维度
        K_flat = K.flatten(2)
        # [B, d_k, H*W]
        
        # Step 5: 计算Attention logits
        attn_logits = torch.bmm(Q, K_flat)
        # [B, num_cls, H*W]
        
        # Step 6: Scale和Softmax
        attn_logits = attn_logits / math.sqrt(self.d_k)
        A = F.softmax(attn_logits, dim=-1)
        # [B, num_cls, H*W]
        
        # Step 7: Reshape到空间维度
        A_spatial = A.view(B, num_cls, H, W)
        # [B, num_cls, H, W]
        
        # Step 8: 生成类别特定特征 (用原始RGB)
        rgb_expanded = rgb_feat.unsqueeze(1)
        # [B, 1, C, H, W]
        
        A_expanded = A_spatial.unsqueeze(2)
        # [B, num_cls, 1, H, W]
        
        rgb_class_specific = rgb_expanded * A_expanded
        # [B, num_cls, C, H, W]
        
        return rgb_class_specific

