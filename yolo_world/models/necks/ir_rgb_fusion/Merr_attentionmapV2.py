# Copyright (c) Tencent Inc. All rights reserved.
# M_err Guided RGB Enhancement Module V2
# 
# 核心改动（相比 V1）：
# - 完全不使用 IR 特征！
# - 只用 M_err（语义错误图）来增强 RGB 特征
# - M_err 明确指出目标区域，用于加权 RGB
# 
# 设计理念：
# - M_err 已经通过 RGB-IR-Text 对比准确定位了目标区域
# - IR 特征语义可能错误（如 P3/P4 背景高亮问题）
# - 直接用 M_err 增强 RGB 目标区域，避免 IR 语义污染
#
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class MerrGuidedRGBEnhancementV2(BaseModule):
    """
    M_err Guided RGB Enhancement Module V2
    
    核心改动（相比 V1）：
    - 完全不使用 IR 特征
    - 只用 M_err 加权 RGB 特征
    - 通过残差连接增强目标区域
    
    融合公式：
        attention = M_err_resized                   (直接用 M_err 作为注意力)
        x_rgb_attended = x_rgb * attention          (目标区域被增强)
        enhancement = conv(x_rgb_attended)          (提取增强特征)
        output = x_rgb + gamma * enhancement        (残差连接)
    
    理论优势：
    - 避免 IR 语义错误污染
    - M_err 明确指出目标区域
    - 简洁高效
    
    Args:
        rgb_channels (int): Number of RGB feature channels
        ir_channels (int): Number of IR feature channels (仅用于兼容接口，实际不使用 IR)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int = None,  # 保留接口兼容，但不使用
                 init_cfg: Optional[dict] = None):
        super(MerrGuidedRGBEnhancementV2, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        
        # ⭐ V2: 不再需要 IR 通道对齐
        # 移除：self.ir_align = ...
        
        # ⭐ V2: 不再需要 attention_gen
        # 移除：self.attention_gen = ...
        
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
        
    def forward(self, 
                x_rgb: torch.Tensor, 
                x_ir: torch.Tensor = None,  # 保留接口兼容，但不使用
                M_err: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB features [B, C_rgb, H, W]
            x_ir: IR features [B, C_ir, H, W] (⭐ V2: 不使用！)
            M_err: Semantic error map [B, 1, H_m, W_m]
        
        Returns:
            Enhanced RGB features [B, C_rgb, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        
        # ⭐ V2: 如果没有 M_err，返回原始 RGB
        if M_err is None:
            return x_rgb
        
        # 将 M_err 插值到当前尺度
        if M_err.shape[-2:] != (H, W):
            M_err_resized = F.interpolate(M_err, size=(H, W), 
                                          mode='bilinear', align_corners=False)
        else:
            M_err_resized = M_err
        
        # ⭐ V2 核心：用 M_err 加权 RGB（不使用 IR）
        # M_err 高的区域是目标区域，被增强
        x_rgb_attended = x_rgb * M_err_resized  # [B, C_rgb, H, W]
        
        # 提取增强特征
        enhancement = self.enhancement_conv(x_rgb_attended)
        
        # 残差连接
        output = x_rgb + self.gamma * enhancement
        
        return output


@MODELS.register_module()
class MultiLevelMerrGuidedFusionV2(BaseModule):
    """
    Multi-level M_err Guided RGB Enhancement Module V2
    
    对每个尺度（P3, P4, P5）应用 MerrGuidedRGBEnhancementV2
    
    ⭐ 核心特点：
    - 完全不使用 IR 特征
    - 只用 M_err 增强 RGB
    
    Args:
        rgb_channels (List[int]): RGB feature channels at each level.
        ir_channels (List[int]): IR feature channels at each level (保留兼容).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int] = None,  # 保留兼容，但不使用
                 init_cfg: Optional[dict] = None):
        super(MultiLevelMerrGuidedFusionV2, self).__init__(init_cfg)
        
        self.num_levels = len(rgb_channels)
        self.rgb_channels = rgb_channels
        
        # Build enhancement modules for each pyramid level
        self.fusion_modules = nn.ModuleList()
        for rgb_ch in rgb_channels:
            self.fusion_modules.append(
                MerrGuidedRGBEnhancementV2(
                    rgb_channels=rgb_ch,
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...] = None,  # ⭐ V2: 不使用
                M_err_list: Tuple[torch.Tensor, ...] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: Tuple of RGB feature tensors (P3, P4, P5)
            ir_feats: Tuple of IR feature tensors (⭐ V2: 不使用！保留接口兼容)
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
            enhanced = self.fusion_modules[i](rgb_feats[i], None, M_err)  # 不传 IR
            enhanced_feats.append(enhanced)
        
        return tuple(enhanced_feats)
