# Copyright (c) Tencent Inc. All rights reserved.
# RGB-IR Cross-Modal Fusion Module V2
# 与原版的区别：同时输出融合特征和align后的IR特征
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class LightweightCrossFusionV2(BaseModule):
    """
    Lightweight fusion module V2: 同时返回融合特征和align后的IR特征
    
    与原版的区别：forward返回(fused_feat, ir_aligned)
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int, 
                 reduction: int = 4,
                 init_cfg: Optional[dict] = None):
        super(LightweightCrossFusionV2, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False) \
                        if ir_channels != rgb_channels else nn.Identity()
        
        mid_channels = max(rgb_channels // reduction, 8)
        self.attention_gen = nn.Sequential(
            nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.cross_conv = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_rgb: RGB features [B, C_rgb, H, W]
            x_ir: IR features [B, C_ir, H, W]
        
        Returns:
            output: Enhanced RGB features [B, C_rgb, H, W]
            x_ir_aligned: Aligned IR features [B, C_rgb, H, W]
        """
        x_ir_aligned = self.ir_align(x_ir)
        
        if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
            x_ir_aligned = F.interpolate(x_ir_aligned, size=x_rgb.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        attention_map = self.attention_gen(x_ir_aligned)
        
        x_rgb_attended = x_rgb * attention_map
        
        combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
        fused = self.cross_conv(combined)
        
        output = x_rgb + self.gamma * fused
        
        return output, x_ir_aligned


@MODELS.register_module()
class MultiLevelRGBIRFusionV2(BaseModule):
    """
    Multi-level RGB-IR fusion module V2: 同时返回融合特征和align后的IR特征
    
    与原版的区别：forward返回(fused_feats, ir_aligned_feats)
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int],
                 reduction: int = 4,
                 init_cfg: Optional[dict] = None):
        super(MultiLevelRGBIRFusionV2, self).__init__(init_cfg)
        
        assert len(rgb_channels) == len(ir_channels), \
            f"rgb_channels and ir_channels must have same length, got {len(rgb_channels)} vs {len(ir_channels)}"
        
        self.num_levels = len(rgb_channels)
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        self.fusion_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.fusion_modules.append(
                LightweightCrossFusionV2(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    reduction=reduction
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Args:
            rgb_feats: Tuple of RGB feature tensors (P3, P4, P5)
            ir_feats: Tuple of IR feature tensors (P3, P4, P5)
            
        Returns:
            fused_feats: Tuple of fused feature tensors (P3, P4, P5)
            ir_aligned_feats: Tuple of aligned IR feature tensors (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Expected {self.num_levels} levels, got RGB:{len(rgb_feats)}, IR:{len(ir_feats)}"
        
        fused_feats = []
        ir_aligned_feats = []
        for i in range(self.num_levels):
            fused, ir_aligned = self.fusion_modules[i](rgb_feats[i], ir_feats[i])
            fused_feats.append(fused)
            ir_aligned_feats.append(ir_aligned)
        
        return tuple(fused_feats), tuple(ir_aligned_feats)

