# Copyright (c) Tencent Inc. All rights reserved.
# Simple Channel Alignment Module - 只做通道对齐，不做特征融合
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class SimpleChannelAlign(BaseModule):
    """
    Simple Channel Alignment Module
    
    只做通道对齐，不做任何特征融合或更新。
    用于在没有Neck的情况下，将Backbone特征对齐到Head期望的通道数。
    
    Args:
        in_channels (list[int]): Backbone输出通道数 (P3, P4, P5)
        out_channels (list[int]): Head期望的输入通道数 (P3, P4, P5)
    """
    
    def __init__(self,
                 in_channels: Tuple[int, int, int],
                 out_channels: Tuple[int, int, int],
                 init_cfg=None):
        super(SimpleChannelAlign, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 为每个尺度创建1x1卷积进行通道对齐
        self.align_convs = nn.ModuleList()
        for in_c, out_c in zip(in_channels, out_channels):
            if in_c != out_c:
                # 需要对齐
                self.align_convs.append(
                    nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
                )
            else:
                # 通道数已对齐，使用Identity
                self.align_convs.append(nn.Identity())
    
    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Forward pass - 只做通道对齐
        
        Args:
            feats: Tuple of feature tensors (P3, P4, P5)
                - P3: [B, in_channels[0], H/8, W/8]
                - P4: [B, in_channels[1], H/16, W/16]
                - P5: [B, in_channels[2], H/32, W/32]
        
        Returns:
            aligned_feats: Tuple of aligned feature tensors (P3, P4, P5)
                - P3: [B, out_channels[0], H/8, W/8]
                - P4: [B, out_channels[1], H/16, W/16]
                - P5: [B, out_channels[2], H/32, W/32]
        """
        aligned_feats = []
        for feat, align_conv in zip(feats, self.align_convs):
            aligned_feats.append(align_conv(feat))
        
        return tuple(aligned_feats)


@MODELS.register_module()
class NoNeckPassThrough(BaseModule):
    """
    Pass-through module for no-neck architecture
    
    直接传递特征，不做任何处理。
    用于完全跳过Neck的情况。
    """
    
    def __init__(self, init_cfg=None):
        super(NoNeckPassThrough, self).__init__(init_cfg)
    
    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """直接返回输入特征"""
        return feats

