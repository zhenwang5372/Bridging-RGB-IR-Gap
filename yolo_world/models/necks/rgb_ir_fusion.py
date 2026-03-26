# Copyright (c) Tencent Inc. All rights reserved.
# RGB-IR Cross-Modal Fusion Module
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class LightweightCrossFusion(BaseModule):
    """
    Lightweight fusion module using IR features to generate spatial attention for RGB.
    
    Mechanism:
        1. IR features → Spatial attention map (via conv + sigmoid)
        2. RGB features × attention map → Enhanced RGB features
        3. Residual connection for stability
    
    Args:
        rgb_channels (int): Number of RGB feature channels
        ir_channels (int): Number of IR feature channels
        reduction (int): Channel reduction ratio for attention (default: 4)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 rgb_channels: int, 
                 ir_channels: int, 
                 reduction: int = 4,
                 init_cfg: Optional[dict] = None):
        super(LightweightCrossFusion, self).__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        # Channel alignment (if RGB and IR have different channels)
        self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False) \
                        if ir_channels != rgb_channels else nn.Identity()
        
        # Lightweight attention generation from IR features
        mid_channels = max(rgb_channels // reduction, 8)
        self.attention_gen = nn.Sequential(
            nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # Spatial attention weights [0, 1]
        )
        
        # Cross-modality interaction (lightweight)
        self.cross_conv = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True)
        )
        
        # Residual weight for stability (learnable)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB features [B, C_rgb, H, W]
            x_ir: IR features [B, C_ir, H, W]
        
        Returns:
            Enhanced RGB features [B, C_rgb, H, W]
        """
        # Align IR channels to RGB
        x_ir_aligned = self.ir_align(x_ir)
        
        # Resize IR features if spatial dimensions don't match
        if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
            x_ir_aligned = F.interpolate(x_ir_aligned, size=x_rgb.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        # Generate spatial attention from IR features
        attention_map = self.attention_gen(x_ir_aligned)
        
        # Apply attention to RGB features
        x_rgb_attended = x_rgb * attention_map
        
        # Cross-modality fusion
        combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
        fused = self.cross_conv(combined)
        
        # Residual connection with learnable weight
        output = x_rgb + self.gamma * fused

        return output


@MODELS.register_module()
class MultiLevelRGBIRFusion(BaseModule):
    """
    Multi-level RGB-IR fusion module for FPN features.
    
    Applies LightweightCrossFusion at each pyramid level (P3, P4, P5).
    
    Args:
        rgb_channels (List[int]): RGB feature channels at each level.
        ir_channels (List[int]): IR feature channels at each level.
        reduction (int): Channel reduction ratio for attention. Defaults to 4.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 rgb_channels: List[int],
                 ir_channels: List[int],
                 reduction: int = 4,
                 init_cfg: Optional[dict] = None):
        super(MultiLevelRGBIRFusion, self).__init__(init_cfg)
        
        assert len(rgb_channels) == len(ir_channels), \
            f"rgb_channels and ir_channels must have same length, got {len(rgb_channels)} vs {len(ir_channels)}"
        
        self.num_levels = len(rgb_channels)
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        
        # Build fusion modules for each pyramid level
        self.fusion_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.fusion_modules.append(
                LightweightCrossFusion(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    reduction=reduction
                )
            )
    
    def forward(self, 
                rgb_feats: Tuple[torch.Tensor, ...],
                ir_feats: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: Tuple of RGB feature tensors (P3, P4, P5)
            ir_feats: Tuple of IR feature tensors (P3, P4, P5)
            
        Returns:
            Tuple of fused feature tensors (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Expected {self.num_levels} levels, got RGB:{len(rgb_feats)}, IR:{len(ir_feats)}"
        
        fused_feats = []
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](rgb_feats[i], ir_feats[i])
            fused_feats.append(fused)
        
        return tuple(fused_feats)
