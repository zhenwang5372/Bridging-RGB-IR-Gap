# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Scale Text Update Module V2

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextUpdateV2(BaseModule):
    """
    Multi-Scale Text Update Module V2 (阶段5)
    
    改进：完全独立的Cross-Attention，直接从Fused特征提取
    
    从多尺度的Fused特征中提取视觉证据，更新Text embedding。
    
    工作流程:
        1. 对每个尺度，执行标准Cross-Attention
           - Q = Text
           - K, V = Fused (不依赖阶段4的rgb_class_specific)
        2. 多尺度融合（加权平均）
        3. 跨batch聚合
        4. 残差更新Text (YOLO-World风格)
    
    Args:
        in_channels (List[int]): Fused特征的通道数 [P3, P4, P5]
        text_dim (int): Text embedding维度，默认512
        num_classes (int): 类别数，默认4
        hidden_dim (int): Cross-Attention的隐藏维度，默认256
        scale_init (float): 残差缩放初始值，默认0.0
        fusion_method (str): 多尺度融合方法，'learned_weight'或'equal'
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
        scale_init: float = 0.0,
        fusion_method: str = 'learned_weight',
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_levels = len(in_channels)
        self.fusion_method = fusion_method
        
        # 为每个尺度创建Cross-Attention模块
        self.level_modules = nn.ModuleList([
            SingleLevelTextUpdate(
                in_channels=ch,
                text_dim=text_dim,
                hidden_dim=hidden_dim
            )
            for ch in in_channels
        ])
        
        # 多尺度融合权重
        if fusion_method == 'learned_weight':
            self.scale_weights = nn.Parameter(
                torch.ones(self.num_levels)
            )
        
        # 残差缩放参数 (YOLO-World风格)
        self.scale = nn.Parameter(torch.tensor(scale_init))
    
    def forward(
        self,
        fused_feats: Tuple[torch.Tensor, ...],
        text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feats: Tuple of [B, C, H, W] - (P3, P4, P5)
                来自阶段3的RGB-IR融合特征
            text_feats: [B, num_cls, text_dim] 或 [num_cls, text_dim] 原始Text embedding
        
        Returns:
            text_updated: [B, num_cls, text_dim] 更新后的Text
            
        工作流程:
            1. 每张图片用自己的视觉特征更新 text_feats → [B, num_cls, text_dim]
            2. 跨Batch聚合更新量 → [num_cls, text_dim]
            3. 残差更新原始text → [num_cls, text_dim]
            4. 广播回 [B, num_cls, text_dim]，所有图片共享同一套更新后的text
        """
        # 获取 batch size
        B = fused_feats[0].shape[0]
        
        # 处理text_feats的不同形状，统一为 [num_cls, text_dim] 用于 Cross-Attention
        if text_feats.dim() == 3:
            # [B, num_cls, text_dim] -> 取第一个batch作为原始text
            # 训练时，batch中所有样本的text是相同的
            text_feats_2d = text_feats[0]  # [num_cls, text_dim]
        elif text_feats.dim() == 2:
            # [num_cls, text_dim] - 已经是正确形状
            text_feats_2d = text_feats
        
        # Step 1: 从每个尺度提取视觉证据
        # 每张图片用自己的视觉特征去更新text
        Y_text_list = []
        
        for fused_feat, module in zip(fused_feats, self.level_modules):
            # fused_feat: [B, C, H, W]
            Y_text_l = module(
                fused_feat=fused_feat,
                text_feat=text_feats_2d  # [num_cls, text_dim]
            )
            # Y_text_l: [B, num_cls, text_dim] - B张图片各自的更新量
            Y_text_list.append(Y_text_l)
        
        # Step 2: 多尺度融合
        if self.fusion_method == 'learned_weight':
            # Softmax归一化权重
            weights = F.softmax(self.scale_weights, dim=0)
            
            # 加权平均
            Y_text_fused = sum(
                w * Y for w, Y in zip(weights, Y_text_list)
            )
        else:  # 'equal'
            # 等权重平均
            Y_text_fused = sum(Y_text_list) / self.num_levels
        
        # Y_text_fused: [B, num_cls, text_dim] - B张图片各自的更新量
        
        # Step 3: 跨Batch聚合更新量
        # 将B张图片的更新量聚合为一个共享的更新量
        Y_text_avg = Y_text_fused.mean(dim=0)
        # Y_text_avg: [num_cls, text_dim]
        
        # Step 4: 残差更新 (YOLO-World风格)
        text_updated_2d = text_feats_2d + self.scale * Y_text_avg
        # text_updated_2d: [num_cls, text_dim]
        
        # Step 5: 广播回 [B, num_cls, text_dim]
        # 所有图片共享同一套更新后的text_feats
        text_updated = text_updated_2d.unsqueeze(0).expand(B, -1, -1)
        # text_updated: [B, num_cls, text_dim]
        
        return text_updated


class SingleLevelTextUpdate(nn.Module):
    """
    单尺度的Text Update模块
    
    标准Cross-Attention:
        Q = Text
        K, V = Fused
    """
    
    def __init__(
        self,
        in_channels: int,
        text_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Query projection: Text -> hidden_dim
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        
        # Key projection: Fused -> hidden_dim
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Value projection: Fused -> hidden_dim
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Output projection: hidden_dim -> text_dim
        self.out_proj = nn.Linear(hidden_dim, text_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.zeros_(self.key_conv.bias)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.zeros_(self.value_conv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        fused_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-Attention: Text attends to Fused features
        
        Args:
            fused_feat: [B, C, H, W] - Fused RGB-IR features
            text_feat: [num_cls, text_dim] - Text embeddings
        
        Returns:
            Y_text: [B, num_cls, text_dim] - Updated text features
        """
        B, C, H, W = fused_feat.shape
        num_cls = text_feat.shape[0]
        
        # Query from Text: [num_cls, text_dim] -> [num_cls, hidden_dim]
        Q = self.query_proj(text_feat)  # [num_cls, hidden_dim]
        
        # Key from Fused: [B, C, H, W] -> [B, hidden_dim, H, W]
        K = self.key_conv(fused_feat)  # [B, hidden_dim, H, W]
        K = K.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Value from Fused: [B, C, H, W] -> [B, hidden_dim, H, W]
        V = self.value_conv(fused_feat)  # [B, hidden_dim, H, W]
        V = V.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Expand Q for batch dimension
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, hidden_dim]
        
        # Attention: Q @ K^T
        # [B, num_cls, hidden_dim] @ [B, hidden_dim, H*W] -> [B, num_cls, H*W]
        attn_scores = torch.bmm(Q_expanded, K) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_cls, H*W]
        
        # Attention: attn_weights @ V
        # [B, num_cls, H*W] @ [B, H*W, hidden_dim] -> [B, num_cls, hidden_dim]
        V_transposed = V.transpose(1, 2)  # [B, H*W, hidden_dim]
        attn_output = torch.bmm(attn_weights, V_transposed)  # [B, num_cls, hidden_dim]
        
        # Output projection: [B, num_cls, hidden_dim] -> [B, num_cls, text_dim]
        Y_text = self.out_proj(attn_output)  # [B, num_cls, text_dim]
        
        return Y_text
