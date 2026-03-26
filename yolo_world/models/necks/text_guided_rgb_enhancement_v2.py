# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB Enhancement Module V2

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedRGBEnhancementV2(BaseModule):
    """
    Text-guided RGB Enhancement Module V2 (阶段4)
    
    改进：使用Fused特征作为Value，标准QKV Attention
    
    为每个尺度独立生成类别特定的RGB特征。
    
    工作流程（对每个尺度）:
        1. Text → Query
        2. Fused → Key
        3. Fused → Value  (改进点)
        4. Attention = Softmax(Q @ K / √d)
        5. Output = Attention @ Value
    
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
            SingleLevelRGBEnhancementV2(
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
            rgb_feats: 原始RGB特征 (P3, P4, P5) - 未使用，保留接口兼容
                每个: [B, C, H, W]
            fused_feats: Fusion后的特征 (P3, P4, P5)
                每个: [B, C, H, W]
            text_feats: Text embedding [num_cls, text_dim]
        
        Returns:
            rgb_class_specific: List of [B, num_cls, C, H, W]
        """
        rgb_class_specific = []
        
        for fused_feat, module in zip(fused_feats, self.level_modules):
            rgb_cs = module(
                fused_feat=fused_feat,
                text_feat=text_feats
            )
            rgb_class_specific.append(rgb_cs)
        
        return rgb_class_specific


class SingleLevelRGBEnhancementV2(nn.Module):
    """
    单尺度的RGB增强模块 V2
    
    核心机制 (标准QKV Attention):
        1. Text作为Query
        2. Fused作为Key
        3. Fused作为Value (改进点)
        4. Attention加权求和Value
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
        
        # Value projection: Fused features -> rgb_channels (保持原通道数)
        self.value_conv = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1)
        
        # 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.zeros_(self.key_conv.bias)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.zeros_(self.value_conv.bias)
    
    def forward(
        self,
        fused_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feat: [B, C, H, W] Fused特征
            text_feat: [num_cls, text_dim] Text embedding
        
        Returns:
            rgb_class_specific: [B, num_cls, C, H, W]
        """
        B, C, H, W = fused_feat.shape
        
        # 处理text_feat的不同形状
        if text_feat.dim() == 4:
            text_feat = text_feat[0, 0]
        elif text_feat.dim() == 3:
            text_feat = text_feat[0]
        
        num_cls = text_feat.shape[0]
        
        # Step 1: 生成Query (来自Text)
        text_expanded = text_feat.unsqueeze(0).expand(B, -1, -1)
        # [B, num_cls, text_dim]
        
        Q = self.query_proj(text_expanded)
        # [B, num_cls, d_k]
        
        # Step 2: 生成Key (来自Fused)
        K = self.key_conv(fused_feat)
        # [B, d_k, H, W]
        K_flat = K.flatten(2)
        # [B, d_k, H*W]
        
        # Step 3: 生成Value (来自Fused)
        V = self.value_conv(fused_feat)
        # [B, C, H, W]
        V_flat = V.flatten(2)
        # [B, C, H*W]
        
        # Step 4: 计算Attention weights
        attn_logits = torch.bmm(Q, K_flat)
        # [B, num_cls, H*W]
        
        attn_logits = attn_logits / math.sqrt(self.d_k)
        A = F.softmax(attn_logits, dim=-1)
        # [B, num_cls, H*W]
        
        # Step 5: Attention加权求和Value
        # A: [B, num_cls, H*W]
        # V_flat: [B, C, H*W]
        # 需要对每个类别独立计算
        
        rgb_class_specific_list = []
        for cls_idx in range(num_cls):
            A_cls = A[:, cls_idx:cls_idx+1, :]  # [B, 1, H*W]
            # 加权求和
            weighted_V = torch.bmm(V_flat, A_cls.transpose(1, 2))
            # [B, C, 1]
            weighted_V = weighted_V.squeeze(2)  # [B, C]
            
            # 广播到空间维度
            weighted_V_spatial = weighted_V.view(B, C, 1, 1).expand(B, C, H, W)
            # [B, C, H, W]
            
            # 使用attention map调制
            A_spatial = A[:, cls_idx, :].view(B, 1, H, W)  # [B, 1, H, W]
            rgb_cs_cls = V * A_spatial
            # [B, C, H, W]
            
            rgb_class_specific_list.append(rgb_cs_cls)
        
        # Stack所有类别
        rgb_class_specific = torch.stack(rgb_class_specific_list, dim=1)
        # [B, num_cls, C, H, W]
        
        return rgb_class_specific

