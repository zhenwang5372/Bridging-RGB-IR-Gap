# Copyright (c) Tencent Inc. All rights reserved.
# Trimodal Phased Neck: Three-stage RGB-IR-Text fusion
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS

from .trimodal_utils import AdditiveFusion
from .ir_correction import IRCorrectionModule
from .rgb_enhancement import RGBEnhancementModule
from .text_update import TextUpdateModule
from .text_update_multiscale import TextUpdateMultiScale
from .text_update_fusion_first import TextUpdateFusionFirst


@MODELS.register_module()
class TriModalPhasedNeck(BaseModule):
    """三模态分阶段更新Neck
    
    三阶段更新顺序：
        阶段1: IR更新 → 使用原始RGB + 原始Text
        阶段2: RGB更新 → 使用IR_new + 原始Text
        阶段3: Text更新 → 使用RGB_new + IR_new + 原始Text
    
    跨尺度融合使用加法融合避免通道不对齐。
    
    Args:
        in_channels: 各尺度输入通道数 [P3, P4, P5]
        text_dim: 文本特征维度
        hidden_dim: 隐藏层维度
        text_update_scale: Text更新使用的尺度 ('P3', 'P4', 'P5')
        ir_correction_cfg: IR纠错模块配置
        rgb_enhancement_cfg: RGB增强模块配置
        text_update_cfg: Text更新模块配置
        init_cfg: 初始化配置
    """
    
    def __init__(self,
                 in_channels: List[int] = [128, 256, 512],
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 text_update_scale: str = 'P4',
                 use_multiscale_text_update: bool = True,  # 是否使用多尺度Text更新
                 multiscale_fusion_first: bool = False,  # 是否先融合再更新（更高效）
                 ir_correction_cfg: ConfigType = None,
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
        
        ir_correction_cfg = ir_correction_cfg or {}
        rgb_enhancement_cfg = rgb_enhancement_cfg or {}
        text_update_cfg = text_update_cfg or {}
        
        self.ir_fuse = nn.ModuleList([
            AdditiveFusion(in_channels[1], in_channels[0]),
            AdditiveFusion(in_channels[2], in_channels[1]),
        ])
        
        self.rgb_fuse = nn.ModuleList([
            AdditiveFusion(in_channels[1], in_channels[0]),
            AdditiveFusion(in_channels[2], in_channels[1]),
        ])
        
        self.ir_correct = nn.ModuleList()
        for ch in in_channels:
            self.ir_correct.append(
                IRCorrectionModule(
                    channels=ch,
                    text_dim=text_dim,
                    hidden_dim=hidden_dim,
                    **ir_correction_cfg
                )
            )
        
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
        
        if use_multiscale_text_update:
            if multiscale_fusion_first:
                # 先融合多尺度，再更新Text（计算量更小）
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
            # 使用单尺度Text更新
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
            
        ir_new, masks = self._phase1_ir_update(ir_feats, rgb_feats, text)
        
        rgb_new = self._phase2_rgb_update(rgb_feats, ir_new, text)
        
        text_new = self._phase3_text_update(rgb_new, ir_new, text)
        
        return tuple(rgb_new), text_new
    
    def _phase1_ir_update(self,
                          ir_feats: Tuple[Tensor, ...],
                          rgb_feats: Tuple[Tensor, ...],
                          text: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """阶段1：IR更新（所有尺度，使用原始RGB + 原始Text）"""
        ir_new = [None, None, None]
        masks = [None, None, None]
        
        ir_new[2], masks[2] = self.ir_correct[2](ir_feats[2], rgb_feats[2], text)
        
        ir4_in = self.ir_fuse[1](ir_new[2], ir_feats[1])
        ir_new[1], masks[1] = self.ir_correct[1](ir4_in, rgb_feats[1], text)
        
        ir3_in = self.ir_fuse[0](ir_new[1], ir_feats[0])
        ir_new[0], masks[0] = self.ir_correct[0](ir3_in, rgb_feats[0], text)
        
        return ir_new, masks
    
    def _phase2_rgb_update(self,
                           rgb_feats: Tuple[Tensor, ...],
                           ir_new: List[Tensor],
                           text: Tensor) -> List[Tensor]:
        """阶段2：RGB更新（所有尺度，使用IR_new + 原始Text）"""
        rgb_new = [None, None, None]
        
        rgb_new[2] = self.rgb_enhance[2](rgb_feats[2], ir_new[2], text)
        
        rgb4_in = self.rgb_fuse[1](rgb_new[2], rgb_feats[1])
        rgb_new[1] = self.rgb_enhance[1](rgb4_in, ir_new[1], text)
        
        rgb3_in = self.rgb_fuse[0](rgb_new[1], rgb_feats[0])
        rgb_new[0] = self.rgb_enhance[0](rgb3_in, ir_new[0], text)
        
        return rgb_new
    
    def _phase3_text_update(self,
                            rgb_new: List[Tensor],
                            ir_new: List[Tensor],
                            text: Tensor) -> Tensor:
        """阶段3：Text更新"""
        if self.use_multiscale_text_update:
            # 多尺度更新：传入所有尺度
            text_new = self.text_update(rgb_new, ir_new, text)
        else:
            # 单尺度更新：使用指定尺度
            idx = self._text_update_idx
            text_new = self.text_update(rgb_new[idx], ir_new[idx], text)
        return text_new

