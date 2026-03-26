# Copyright (c) Tencent Inc. All rights reserved.
# Class Dimension Aggregator Module
# 
# 支持与Fused特征融合的类别维度聚合器
# 方案1: Concat + Conv 融合

import torch
import torch.nn as nn
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class ClassDimensionAggregator(BaseModule):
    """
    类别维度聚合器（支持与Fused特征融合）
    
    作用：将类别特定特征聚合为标准特征，并可选地与Fused特征融合
    
    输入：
        - class_specific_feats: List of [B, num_cls, C, H, W] - 类别特定特征
        - fused_feats: List of [B, C, H, W] - Fused特征（可选）
    输出：List of [B, C, H, W] - 标准特征
    
    Args:
        in_channels (List[int]): 每个尺度的通道数，例如 [128, 256, 512]
        num_classes (int): 类别数，默认4
        aggregation_method (str): 聚合方法
            - 'conv': 1x1卷积聚合（推荐，简单高效）
            - 'mlp': 两层MLP聚合
            - 'attention': Attention加权聚合
            - 'max': Max pooling（原始方法）
            - 'avg': Average pooling
        fusion_type (str): 与Fused特征的融合类型
            - 'none': 不融合（原始行为）
            - 'add': 简单相加
            - 'concat': Concat + Conv（推荐）
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 4,
        aggregation_method: str = 'conv',
        fusion_type: str = 'none',
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.aggregation_method = aggregation_method
        self.fusion_type = fusion_type
        self.num_levels = len(in_channels)
        
        # 为每个尺度创建聚合模块
        self.aggregators = nn.ModuleList()
        
        for C in in_channels:
            if aggregation_method == 'conv':
                # 简单的1x1卷积聚合
                aggregator = nn.Sequential(
                    nn.Conv2d(
                        C * num_classes,  # 输入：类别维度展平后的通道数
                        C,  # 输出：原始通道数
                        kernel_size=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True)
                )
                
            elif aggregation_method == 'mlp':
                # 两层MLP with bottleneck
                hidden_dim = C * 2
                aggregator = nn.Sequential(
                    nn.Conv2d(C * num_classes, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, C, kernel_size=1, bias=False),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True)
                )
                
            elif aggregation_method == 'attention':
                # Attention加权聚合
                aggregator = AttentionAggregator(C, num_classes)
                
            elif aggregation_method == 'max':
                # Max pooling（原始方法）
                aggregator = MaxPoolAggregator()
                
            elif aggregation_method == 'avg':
                # Average pooling
                aggregator = AvgPoolAggregator()
                
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            self.aggregators.append(aggregator)
        
        # ⭐ 新增：融合模块（用于与Fused特征融合）
        if fusion_type == 'concat':
            self.fusion_convs = nn.ModuleList()
            for C in in_channels:
                # Concat后通道数翻倍，用1x1卷积降回原通道数
                fusion_conv = nn.Sequential(
                    nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True)
                )
                self.fusion_convs.append(fusion_conv)
    
    def forward(
        self,
        class_specific_feats: Tuple[torch.Tensor, ...],
        fused_feats: Tuple[torch.Tensor, ...] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        聚合类别维度，并可选地与Fused特征融合
        
        Args:
            class_specific_feats: Tuple of [B, num_cls, C, H, W]
                来自Backbone的类别特定特征
            fused_feats: Tuple of [B, C, H, W]（可选）
                来自Backbone的Fused特征，用于融合
        
        Returns:
            aggregated_feats: Tuple of [B, C, H, W]
                聚合后的标准特征，可以输入传统YOLO Head
        """
        aggregated_feats = []
        
        for i, (feat, aggregator) in enumerate(zip(class_specific_feats, self.aggregators)):
            # feat: [B, num_cls, C, H, W]
            
            # Step 1: 聚合类别维度
            if self.aggregation_method in ['max', 'avg']:
                # Pooling方法直接处理5D tensor
                aggregated = aggregator(feat)
            else:
                # Conv/MLP/Attention需要先reshape
                B, num_cls, C, H, W = feat.shape
                # 展平类别维度: [B, num_cls, C, H, W] → [B, num_cls*C, H, W]
                feat_flat = feat.reshape(B, num_cls * C, H, W)
                aggregated = aggregator(feat_flat)
            
            # Step 2: ⭐ 与Fused特征融合
            if fused_feats is not None and self.fusion_type != 'none':
                fused = fused_feats[i]
                
                if self.fusion_type == 'add':
                    # 简单相加
                    aggregated = aggregated + fused
                    
                elif self.fusion_type == 'concat':
                    # Concat + Conv（方案1推荐方式）
                    # Combined = Concat([Aggregated, Fused], dim=1)  # [B, 2C, H, W]
                    # Final = Conv1x1(Combined)                       # [B, C, H, W]
                    combined = torch.cat([aggregated, fused], dim=1)
                    aggregated = self.fusion_convs[i](combined)
            
            # aggregated: [B, C, H, W]
            aggregated_feats.append(aggregated)
        
        return tuple(aggregated_feats)


class AttentionAggregator(nn.Module):
    """
    使用Attention机制聚合类别维度
    
    学习每个位置上不同类别的重要性权重，然后加权聚合
    """
    
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        
        self.channels = channels
        self.num_classes = num_classes
        
        # 生成attention权重的网络
        self.attention_net = nn.Sequential(
            nn.Conv2d(
                channels * num_classes,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                num_classes,  # 输出每个类别的权重
                kernel_size=1
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_cls*C, H, W] - 展平后的特征
        
        Returns:
            out: [B, C, H, W] - 聚合后的特征
        """
        B, _, H, W = x.shape
        
        # 计算attention权重
        attn = self.attention_net(x)  # [B, num_cls, H, W]
        attn = torch.softmax(attn, dim=1)  # Softmax归一化
        
        # 将x reshape回5D来应用attention
        C = self.channels
        num_cls = self.num_classes
        x_5d = x.reshape(B, num_cls, C, H, W)
        
        # 加权聚合
        attn = attn.unsqueeze(2)  # [B, num_cls, 1, H, W]
        out = (x_5d * attn).sum(dim=1)  # [B, C, H, W]
        
        return out


class MaxPoolAggregator(nn.Module):
    """Max Pooling聚合（原始方法）"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_cls, C, H, W]
        
        Returns:
            out: [B, C, H, W]
        """
        return x.max(dim=1)[0]


class AvgPoolAggregator(nn.Module):
    """Average Pooling聚合"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_cls, C, H, W]
        
        Returns:
            out: [B, C, H, W]
        """
        return x.mean(dim=1)
