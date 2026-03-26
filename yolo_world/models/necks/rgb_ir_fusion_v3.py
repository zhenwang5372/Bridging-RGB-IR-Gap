# Copyright (c) Tencent Inc. All rights reserved.
# RGB-IR Cross-Modal Fusion Module - V3 with multiple fusion strategies
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    
    在 concat 之后使用，让网络学习选择 IR 还是 RGB 的通道。
    
    优点：简单有效，能学习通道重要性
    缺点：只有通道注意力，没有空间注意力
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    """
    Spatial and Channel Squeeze-Excitation Block.
    结合空间和通道注意力，比纯 SENet 更强。
    
    优点：同时考虑空间和通道
    缺点：计算量稍大
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel SE
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # Spatial SE
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cse_out = x * self.cse(x)
        sse_out = x * self.sse(x)
        return cse_out + sse_out


class GatedFusion(nn.Module):
    """
    门控融合：学习每个空间位置 IR 和 RGB 的融合比例。
    
    公式: fused = gate * x_ir + (1 - gate) * x_rgb
    gate 是从两个模态学习得到的空间权重图
    
    优点：
    - 每个像素位置可以有不同的融合比例
    - 目标区域可以更多使用 IR，背景可以更多使用 RGB
    - 非常灵活
    
    缺点：需要额外学习 gate 网络
    
    推荐度：★★★★★ (你的场景最推荐)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()  # Output: [0, 1] gate map
        )
    
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_rgb, x_ir], dim=1)
        gate = self.gate_conv(combined)  # [B, 1, H, W]
        # gate 接近 1 时使用更多 IR，接近 0 时使用更多 RGB
        fused = gate * x_ir + (1 - gate) * x_rgb
        return fused, gate


class ChannelGatedFusion(nn.Module):
    """
    通道级门控融合：为每个通道学习不同的 IR/RGB 融合比例。
    
    公式: fused = gate * x_ir + (1 - gate) * x_rgb
    gate 是 [B, C, 1, 1] 的通道权重
    
    优点：比空间门控更轻量，同时保持灵活性
    缺点：不能做空间级别的自适应
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_rgb, x_ir], dim=1)
        gate = self.gate(combined)  # [B, C, 1, 1]
        fused = gate * x_ir + (1 - gate) * x_rgb
        return fused, gate


@MODELS.register_module()
class LightweightCrossFusionV3(BaseModule):
    """
    改进的 RGB-IR 融合模块，支持多种融合策略。
    
    主要改进：
    1. 支持 SENet/SCSE/Gated Fusion 等多种融合方式
    2. 支持多种输出模式：RGB-dominant / IR-dominant / fused-only
    3. 可视化友好，保留中间特征供分析
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        reduction (int): 注意力模块的通道缩减比例
        fusion_type (str): 融合类型
            - 'senet': SE Block on combined features
            - 'scse': Spatial-Channel SE Block  
            - 'gated': Spatial gated fusion (推荐)
            - 'channel_gated': Channel-wise gated fusion
            - 'none': No attention, just conv
        output_type (str): 输出类型
            - 'rgb_dominant': output = x_rgb + gamma * fused (原设计)
            - 'ir_dominant': output = x_ir + gamma * fused (推荐夜间场景)
            - 'fused_only': output = fused (直接输出融合特征)
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int, 
                 reduction: int = 4,
                 fusion_type: str = 'gated',
                 output_type: str = 'ir_dominant',
                 init_cfg: Optional[dict] = None):
        super(LightweightCrossFusionV3, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.fusion_type = fusion_type
        self.output_type = output_type
        
        assert output_type in ['rgb_dominant', 'ir_dominant', 'fused_only'], \
            f"output_type must be 'rgb_dominant', 'ir_dominant' or 'fused_only', got {output_type}"
        
        # Channel alignment
        self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False) \
                        if ir_channels != rgb_channels else nn.Identity()
        
        # Attention generation from IR
        mid_channels = max(rgb_channels // reduction, 8)
        self.attention_gen = nn.Sequential(
            nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Fusion module based on type
        if fusion_type == 'senet':
            self.fusion_attention = SEBlock(rgb_channels * 2, reduction=reduction)
            self.cross_conv = nn.Sequential(
                nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True)
            )
        elif fusion_type == 'scse':
            self.fusion_attention = SCSEBlock(rgb_channels * 2, reduction=reduction)
            self.cross_conv = nn.Sequential(
                nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True)
            )
        elif fusion_type == 'gated':
            self.gated_fusion = GatedFusion(rgb_channels)
            self.cross_conv = nn.Sequential(
                nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True)
            )
        elif fusion_type == 'channel_gated':
            self.gated_fusion = ChannelGatedFusion(rgb_channels, reduction=reduction)
            self.cross_conv = nn.Sequential(
                nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True)
            )
        else:  # 'none'
            self.cross_conv = nn.Sequential(
                nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(rgb_channels),
                nn.SiLU(inplace=True)
            )
        
        # Learnable residual weight
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        # Align IR channels
        x_ir_aligned = self.ir_align(x_ir)
        
        # Resize if needed
        if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
            x_ir_aligned = F.interpolate(x_ir_aligned, size=x_rgb.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        # Generate attention from IR
        attention_map = self.attention_gen(x_ir_aligned)
        
        # Apply attention to RGB
        x_rgb_attended = x_rgb * attention_map
        
        # Fusion based on type
        if self.fusion_type in ['senet', 'scse']:
            combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
            combined = self.fusion_attention(combined)  # Apply attention
            fused = self.cross_conv(combined)
        elif self.fusion_type in ['gated', 'channel_gated']:
            fused, gate = self.gated_fusion(x_rgb_attended, x_ir_aligned)
            fused = self.cross_conv(fused)
        else:
            combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
            fused = self.cross_conv(combined)
        
        # Output based on output_type
        if self.output_type == 'fused_only':
            # 直接输出融合特征
            output = fused
        elif self.output_type == 'ir_dominant':
            # IR 为主体，fused 作为增强
            output = x_ir_aligned + self.gamma * fused
        else:  # 'rgb_dominant'
            # RGB 为主体（原设计）
            output = x_rgb + self.gamma * fused
        
        return output


@MODELS.register_module()
class MultiLevelRGBIRFusionV3(BaseModule):
    """
    Multi-level RGB-IR fusion with selectable fusion strategy.
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表
        ir_channels (List[int]): IR 特征通道数列表
        reduction (int): 注意力模块的通道缩减比例
        fusion_type (str): 融合类型 ('senet', 'scse', 'gated', 'channel_gated', 'none')
        output_type (str): 输出类型 ('rgb_dominant', 'ir_dominant', 'fused_only')
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int],
                 reduction: int = 4,
                 fusion_type: str = 'gated',
                 output_type: str = 'ir_dominant',
                 init_cfg: Optional[dict] = None):
        super(MultiLevelRGBIRFusionV3, self).__init__(init_cfg)
        
        assert len(rgb_channels) == len(ir_channels)
        
        self.num_levels = len(rgb_channels)
        self.fusion_modules = nn.ModuleList()
        
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.fusion_modules.append(
                LightweightCrossFusionV3(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    reduction=reduction,
                    fusion_type=fusion_type,
                    output_type=output_type
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        fused_feats = []
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](rgb_feats[i], ir_feats[i])
            fused_feats.append(fused)
        return tuple(fused_feats)
