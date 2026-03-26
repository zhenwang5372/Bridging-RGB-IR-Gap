# Copyright (c) Tencent Inc. All rights reserved.
# Trimodal Neck Shared Utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule


class AdditiveFusion(nn.Module):
    """加法融合：上采样后与当前层相加，保持通道数不变"""
    
    def __init__(self, high_channels: int, low_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.align = nn.Conv2d(high_channels, low_channels, 1) if high_channels != low_channels else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_high: 高层特征（分辨率小）[B, C_high, H, W]
            x_low: 低层特征（分辨率大）[B, C_low, 2H, 2W]
        Returns:
            融合后特征 [B, C_low, 2H, 2W]
        """
        x_up = self.upsample(x_high)
        x_up = self.align(x_up)
        return x_low + self.alpha * x_up


class ChannelAttention(nn.Module):
    """通道注意力模块（CBAM风格）"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            通道注意力权重 [B, C, 1, 1]
        """
        B, C, _, _ = x.shape
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        combined = torch.cat([avg_out, max_out], dim=-1)
        attn = self.mlp(combined).view(B, C, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    """空间注意力模块（CBAM风格）"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            空间注意力权重 [B, 1, H, W]
        """
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(combined)
        return attn


class IRGuidedCBAM(nn.Module):
    """IR引导的CBAM模块：通道注意力来自RGB，空间注意力来自IR"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB特征 [B, C, H, W]
            x_ir: IR特征 [B, C, H, W]
        Returns:
            增强后的RGB特征 [B, C, H, W]
        """
        ca = self.channel_attn(x_rgb)
        sa = self.spatial_attn(x_ir)
        x_rgb_ca = x_rgb * ca
        x_rgb_enhanced = x_rgb + self.alpha * (x_rgb_ca * sa)
        return x_rgb_enhanced


class CrossAttention(nn.Module):
    """Cross-Attention模块"""
    
    def __init__(self, 
                 query_dim: int, 
                 key_dim: int, 
                 value_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(value_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> tuple:
        """
        Args:
            query: [num_queries, query_dim] 或 [B, num_queries, query_dim]
            key: [B, N, key_dim]
            value: [B, N, value_dim]
        Returns:
            attn_output: [B, num_queries, hidden_dim]
            attn_weights: [B, num_queries, N]
        """
        if query.dim() == 2:
            query = query.unsqueeze(0).expand(key.shape[0], -1, -1)
            
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        attn_weights = torch.bmm(Q, K.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.bmm(attn_weights, V)
        
        return attn_output, attn_weights


class SmoothConv(nn.Module):
    """平滑卷积：将注意力图平滑化"""
    
    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W]
        Returns:
            平滑后的mask [B, 1, H, W]
        """
        return self.conv(x)


def compute_ir_statistics(x_ir: torch.Tensor) -> torch.Tensor:
    """计算IR特征的统计量
    
    Args:
        x_ir: IR特征 [B, C, H, W]
    Returns:
        统计量 [B, 5]
    """
    B = x_ir.shape[0]
    x_flat = x_ir.view(B, -1)
    
    mean_val = x_flat.mean(dim=-1)
    std_val = x_flat.std(dim=-1)
    max_val = x_flat.amax(dim=-1)
    min_val = x_flat.amin(dim=-1)
    dynamic_range = max_val - min_val
    
    stats = torch.stack([mean_val, std_val, max_val, min_val, dynamic_range], dim=-1)
    return stats


def compute_attention_consistency(A_rgb: torch.Tensor, A_ir: torch.Tensor) -> torch.Tensor:
    """计算两路注意力的一致性（余弦相似度）
    
    Args:
        A_rgb: RGB注意力 [B, num_cls, N]
        A_ir: IR注意力 [B, num_cls, N]
    Returns:
        一致性 [B, num_cls]
    """
    A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)
    A_ir_norm = F.normalize(A_ir, p=2, dim=-1)
    G = (A_rgb_norm * A_ir_norm).sum(dim=-1)
    return G

