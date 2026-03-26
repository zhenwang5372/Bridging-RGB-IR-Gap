# Copyright (c) Tencent Inc. All rights reserved.
# M_err Guided RGB-IR Fusion Module V1
# 
# 核心改动（相比原 LightweightCrossFusion）：
# - 用 M_err（语义错误图）替代从 IR 学习的 attention_map
# - M_err 来自 TextGuidedIRCorrectionV6，能够准确定位目标区域
# - 仍然 concat IR 特征进行跨模态融合
# 
# M_err 的优势：
# - 基于文本引导的语义感知注意力
# - 可视化显示能够准确定位目标区域
# - 不需要额外学习注意力生成网络
#
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class MerrGuidedCrossFusionV1(BaseModule):
    """
    M_err Guided Cross-Modal Fusion Module V1
    
    核心改动：
    - 用 M_err 替代 attention_map（移除 attention_gen 模块）
    - M_err 维度 [B, 1, H, W]，需要广播到通道维度
    - 仍然 concat IR 特征进行融合
    
    融合公式：
        x_rgb_attended = x_rgb * M_err_resized  (用 M_err 加权 RGB)
        combined = cat([x_rgb_attended, x_ir_aligned], dim=1)
        fused = cross_conv(combined)
        output = x_rgb + gamma * fused
    
    Args:
        rgb_channels (int): Number of RGB feature channels
        ir_channels (int): Number of IR feature channels
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int, 
                 init_cfg: Optional[dict] = None):
        super(MerrGuidedCrossFusionV1, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        # Channel alignment (if RGB and IR have different channels)
        self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False) \
                        if ir_channels != rgb_channels else nn.Identity()
        
        # ⭐ V1: 移除 attention_gen，改用 M_err
        # 不再需要：self.attention_gen = ...
        
        # Cross-modality interaction
        self.cross_conv = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True)
        )
        
        # Residual weight for stability (learnable)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, 
                x_rgb: torch.Tensor, 
                x_ir: torch.Tensor,
                M_err: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB features [B, C_rgb, H, W]
            x_ir: IR features [B, C_ir, H, W]
            M_err: Semantic error map [B, 1, H_m, W_m]  ⭐ 来自 IR Correction V6
        
        Returns:
            Enhanced RGB features [B, C_rgb, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        
        # Align IR channels to RGB
        x_ir_aligned = self.ir_align(x_ir)
        
        # Resize IR features if spatial dimensions don't match
        if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
            x_ir_aligned = F.interpolate(x_ir_aligned, size=(H, W), 
                                        mode='bilinear', align_corners=False)
        
        # ⭐ V1 核心改动：用 M_err 替代 attention_map
        # M_err 来自 IR Correction，表示目标区域
        # 需要插值到当前尺度
        if M_err.shape[-2:] != (H, W):
            M_err_resized = F.interpolate(M_err, size=(H, W), 
                                          mode='bilinear', align_corners=False)
        else:
            M_err_resized = M_err
        
        # M_err: [B, 1, H, W] → 广播到所有通道
        # Apply M_err as attention to RGB features
        x_rgb_attended = x_rgb * M_err_resized  # [B, C_rgb, H, W]
        
        # Cross-modality fusion (仍然使用 IR 特征)
        combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
        fused = self.cross_conv(combined)
        
        # Residual connection with learnable weight
        output = x_rgb + self.gamma * fused
        
        return output


@MODELS.register_module()
class MultiLevelMerrGuidedFusionV1(BaseModule):
    """
    Multi-level M_err Guided RGB-IR Fusion Module V1
    
    对每个尺度（P3, P4, P5）应用 MerrGuidedCrossFusionV1
    
    Args:
        rgb_channels (List[int]): RGB feature channels at each level.
        ir_channels (List[int]): IR feature channels at each level.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int],
                 init_cfg: Optional[dict] = None):
        super(MultiLevelMerrGuidedFusionV1, self).__init__(init_cfg)
        
        assert len(rgb_channels) == len(ir_channels)
        
        self.num_levels = len(rgb_channels)
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        # Build fusion modules for each pyramid level
        self.fusion_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.fusion_modules.append(
                MerrGuidedCrossFusionV1(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...],
                M_err_list: Tuple[torch.Tensor, ...] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: Tuple of RGB feature tensors (P3, P4, P5)
            ir_feats: Tuple of IR feature tensors (P3, P4, P5)
            M_err_list: Tuple of M_err tensors (P3, P4, P5)  ⭐ 来自 IR Correction V6
            
        Returns:
            Tuple of fused feature tensors (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels
        
        # 如果没有提供 M_err，使用全 1（相当于原始融合）
        if M_err_list is None:
            M_err_list = [torch.ones(rgb_feats[i].shape[0], 1, 
                                     rgb_feats[i].shape[2], rgb_feats[i].shape[3],
                                     device=rgb_feats[i].device)
                          for i in range(self.num_levels)]
        
        fused_feats = []
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](rgb_feats[i], ir_feats[i], M_err_list[i])
            fused_feats.append(fused)
        
        return tuple(fused_feats)
