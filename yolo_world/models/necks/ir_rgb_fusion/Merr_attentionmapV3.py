# Copyright (c) Tencent Inc. All rights reserved.
# M_err Guided RGB Enhancement Module V3 - Hard Attention with Dual Thresholds
# 
# 核心改动（相比 V2）：
# - 对 M_err 应用双阈值硬注意力（Hard Attention）
# - 低于 threshold_low 的区域 → 置为 0（抑制噪声）
# - 高于 threshold_high 的区域 → 置为 1（强化目标）
# - 中间区域保持原值（平滑过渡）
# 
# 设计理念：
# - V2 使用软注意力（M_err 原值），可能包含噪声
# - V3 使用硬注意力，明确区分"目标 vs 非目标"
# - 双阈值策略平衡"去噪"和"保留边界"
#
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class MerrGuidedRGBEnhancementV3(BaseModule):
    """
    M_err Guided RGB Enhancement Module V3 - Hard Attention with Dual Thresholds
    
    核心改动（相比 V2）：
    - 对 M_err 应用双阈值硬注意力
    - 更强的目标区域聚焦能力
    
    融合公式：
        M_err_hard = hard_threshold(M_err, low, high)
            - M_err < low  → 0
            - M_err > high → 1
            - else         → M_err (保持原值)
        x_rgb_attended = x_rgb * M_err_hard
        enhancement = conv(x_rgb_attended)
        output = x_rgb + gamma * enhancement
    
    Args:
        rgb_channels (int): Number of RGB feature channels
        ir_channels (int): Number of IR feature channels (仅用于兼容接口)
        threshold_low (float): 低阈值，低于此值置为 0（默认 0.2）
        threshold_high (float): 高阈值，高于此值置为 1（默认 0.7）
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int = None,
                 threshold_low: float = 0.2,
                 threshold_high: float = 0.7,
                 init_cfg: Optional[dict] = None):
        super(MerrGuidedRGBEnhancementV3, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
        # RGB 特征增强模块（轻量级）
        self.enhancement_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
        )
        
        # 可学习的残差权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def apply_hard_threshold(self, M_err: torch.Tensor) -> torch.Tensor:
        """
        对 M_err 应用双阈值硬注意力
        
        Args:
            M_err: [B, 1, H, W], 范围 [0, 1]
        
        Returns:
            M_err_hard: [B, 1, H, W]
                - M_err < threshold_low  → 0
                - M_err > threshold_high → 1
                - else                   → M_err (保持原值)
        """
        M_err_hard = M_err.clone()
        
        # 低于低阈值 → 置为 0（抑制噪声）
        low_mask = M_err < self.threshold_low
        M_err_hard[low_mask] = 0.0
        
        # 高于高阈值 → 置为 1（强化目标）
        high_mask = M_err > self.threshold_high
        M_err_hard[high_mask] = 1.0
        
        # 中间区域保持原值（平滑过渡）
        
        return M_err_hard
        
    def forward(self, 
                x_rgb: torch.Tensor, 
                x_ir: torch.Tensor = None,
                M_err: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB features [B, C_rgb, H, W]
            x_ir: IR features [B, C_ir, H, W] (不使用)
            M_err: Semantic error map [B, 1, H_m, W_m]
        
        Returns:
            Enhanced RGB features [B, C_rgb, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        
        # 如果没有 M_err，返回原始 RGB
        if M_err is None:
            return x_rgb
        
        # 将 M_err 插值到当前尺度
        if M_err.shape[-2:] != (H, W):
            M_err_resized = F.interpolate(M_err, size=(H, W), 
                                          mode='bilinear', align_corners=False)
        else:
            M_err_resized = M_err
        
        # ⭐ V3 核心：应用双阈值硬注意力
        M_err_hard = self.apply_hard_threshold(M_err_resized)
        
        # 用硬注意力加权 RGB
        x_rgb_attended = x_rgb * M_err_hard  # [B, C_rgb, H, W]
        
        # 提取增强特征
        enhancement = self.enhancement_conv(x_rgb_attended)
        
        # 残差连接
        output = x_rgb + self.gamma * enhancement
        
        return output


@MODELS.register_module()
class MultiLevelMerrGuidedFusionV3(BaseModule):
    """
    Multi-level M_err Guided RGB Enhancement Module V3
    
    对每个尺度（P3, P4, P5）应用 MerrGuidedRGBEnhancementV3
    
    ⭐ 核心特点：
    - 完全不使用 IR 特征
    - 对 M_err 应用双阈值硬注意力
    - 更强的目标区域聚焦能力
    
    Args:
        rgb_channels (List[int]): RGB feature channels at each level.
        ir_channels (List[int]): IR feature channels at each level (保留兼容).
        threshold_low (float): 低阈值（默认 0.2）
        threshold_high (float): 高阈值（默认 0.7）
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int] = None,
                 threshold_low: float = 0.2,
                 threshold_high: float = 0.7,
                 init_cfg: Optional[dict] = None):
        super(MultiLevelMerrGuidedFusionV3, self).__init__(init_cfg)
        
        self.num_levels = len(rgb_channels)
        self.rgb_channels = rgb_channels
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
        # Build enhancement modules for each pyramid level
        self.fusion_modules = nn.ModuleList()
        for rgb_ch in rgb_channels:
            self.fusion_modules.append(
                MerrGuidedRGBEnhancementV3(
                    rgb_channels=rgb_ch,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...] = None,
                M_err_list: Tuple[torch.Tensor, ...] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: Tuple of RGB feature tensors (P3, P4, P5)
            ir_feats: Tuple of IR feature tensors (不使用)
            M_err_list: Tuple of M_err tensors (P3, P4, P5)
            
        Returns:
            Tuple of enhanced RGB feature tensors (P3, P4, P5)
        """
        assert len(rgb_feats) == self.num_levels
        
        # 如果没有提供 M_err，直接返回 RGB
        if M_err_list is None:
            return rgb_feats
        
        enhanced_feats = []
        for i in range(self.num_levels):
            M_err = M_err_list[i] if M_err_list is not None else None
            enhanced = self.fusion_modules[i](rgb_feats[i], None, M_err)
            enhanced_feats.append(enhanced)
        
        return tuple(enhanced_feats)
