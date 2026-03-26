# Copyright (c) Tencent Inc. All rights reserved.
# Simplified Trimodal Neck: 跳过IR更新，只更新RGB和Text
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS

from ..trimodal_utils import AdditiveFusion
from .rgb_enhancement import RGBEnhancementModule
from .text_update import TextUpdateModule
from .text_update_multiscale import TextUpdateMultiScale
from .text_update_fusion_first import TextUpdateFusionFirst


@MODELS.register_module()
class SimplifiedTriModalNeck(BaseModule):
    """简化版三模态Neck：跳过IR更新
    
    两阶段更新顺序：
        阶段1: RGB更新 → 使用原始IR + 原始Text
        阶段2: Text更新 → 使用原始IR + RGB_new + 原始Text
    
    优势：
        - 减少复杂度，降低累积误差
        - IR作为稳定的"锚点"，不被破坏
        - 保留IR-Guided CBAM等有效机制
        - 训练更稳定
    
    Args:
        in_channels: 各尺度输入通道数 [P3, P4, P5]
        text_dim: 文本特征维度
        hidden_dim: 隐藏层维度
        text_update_scale: Text更新使用的尺度 ('P3', 'P4', 'P5')
        use_multiscale_text_update: 是否使用多尺度Text更新
        multiscale_fusion_first: 多尺度时是否先融合再更新
        rgb_enhancement_cfg: RGB增强模块配置
        text_update_cfg: Text更新模块配置
        init_cfg: 初始化配置
    """
    
    def __init__(self,
                 in_channels: List[int] = [128, 256, 512],
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 text_update_scale: str = 'P4',
                 use_multiscale_text_update: bool = True,
                 multiscale_fusion_first: bool = False,
                 rgb_enhancement_cfg: ConfigType = None,
                 text_update_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.text_update_scale = text_update_scale
        self.use_multiscale_text_update = use_multiscale_text_update
        self.multiscale_fusion_first = multiscale_fusion_first
        self.num_levels = len(in_channels)
        
        rgb_enhancement_cfg = rgb_enhancement_cfg or {}
        text_update_cfg = text_update_cfg or {}
        
        # 跨尺度融合（用于RGB更新的top-down）
        self.rgb_fuse = nn.ModuleList([
            AdditiveFusion(in_channels[1], in_channels[0]),
            AdditiveFusion(in_channels[2], in_channels[1]),
        ])
        
        # RGB增强模块（每个尺度）
        self.rgb_enhance = nn.ModuleList()
        for ch in in_channels:
            self.rgb_enhance.append(
                RGBEnhancementModule(
                    channels=ch,
                    text_dim=text_dim,
                    hidden_dim=hidden_dim,
                    **rgb_enhancement_cfg
                )
            )
        
        # Text更新模块
        if use_multiscale_text_update:
            if multiscale_fusion_first:
                # 先融合多尺度，再更新Text
                self.text_update = TextUpdateFusionFirst(
                    in_channels=in_channels,
                    text_dim=text_dim,
                    hidden_dim=hidden_dim,
                    **text_update_cfg
                )
            else:
                # 多尺度分别更新，再融合增量
                self.text_update = TextUpdateMultiScale(
                    in_channels=in_channels,
                    text_dim=text_dim,
                    hidden_dim=hidden_dim,
                    **text_update_cfg
                )
            self._text_update_idx = None
        else:
            # 单尺度Text更新
            scale_idx = {'P3': 0, 'P4': 1, 'P5': 2}[text_update_scale]
            self.text_update = TextUpdateModule(
                channels=in_channels[scale_idx],
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                **text_update_cfg
            )
            self._text_update_idx = scale_idx
        
    def forward(self,
                rgb_feats: Tuple[Tensor, ...],
                ir_feats: Tuple[Tensor, ...],
                text_feats: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """前向传播
        
        Args:
            rgb_feats: RGB特征 (P3, P4, P5)
            ir_feats: IR特征 (P3, P4, P5)，已对齐到与RGB相同通道数
            text_feats: 文本特征 [B, num_cls, text_dim] 或 [num_cls, text_dim]
            
        Returns:
            rgb_new: 更新后的RGB特征 (P3, P4, P5)
            text_new: 更新后的Text特征 [B, num_cls, text_dim]
        """
        if text_feats.dim() == 3:
            text = text_feats[0]
        else:
            text = text_feats
        
        # 阶段1: RGB更新（使用原始IR + 原始Text）
        rgb_new = self._phase1_rgb_update(rgb_feats, ir_feats, text)
        
        # 阶段2: Text更新（使用原始IR + RGB_new + 原始Text）
        text_new = self._phase2_text_update(rgb_new, ir_feats, text)
        
        return tuple(rgb_new), text_new
    
    def _phase1_rgb_update(self,
                           rgb_feats: Tuple[Tensor, ...],
                           ir_feats: Tuple[Tensor, ...],
                           text: Tensor) -> List[Tensor]:
        """阶段1：RGB更新（所有尺度，使用原始IR + 原始Text）
        
        采用top-down方式：P5 → P4 → P3
        """
        rgb_new = [None, None, None]
        
        # P5: 直接更新
        rgb_new[2] = self.rgb_enhance[2](rgb_feats[2], ir_feats[2], text)
        
        # P4: 融合P5的信息后更新
        rgb4_in = self.rgb_fuse[1](rgb_new[2], rgb_feats[1])
        rgb_new[1] = self.rgb_enhance[1](rgb4_in, ir_feats[1], text)
        
        # P3: 融合P4的信息后更新
        rgb3_in = self.rgb_fuse[0](rgb_new[1], rgb_feats[0])
        rgb_new[0] = self.rgb_enhance[0](rgb3_in, ir_feats[0], text)
        
        return rgb_new
    
    def _phase2_text_update(self,
                            rgb_new: List[Tensor],
                            ir_feats: Tuple[Tensor, ...],
                            text: Tensor) -> Tensor:
        """阶段2：Text更新（使用原始IR + RGB_new）"""
        if self.use_multiscale_text_update:
            # 多尺度更新：传入所有尺度
            text_new = self.text_update(rgb_new, list(ir_feats), text)
        else:
            # 单尺度更新：使用指定尺度
            idx = self._text_update_idx
            text_new = self.text_update(rgb_new[idx], ir_feats[idx], text)
        return text_new

