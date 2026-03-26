# Copyright (c) Tencent Inc. All rights reserved.
# Text-Only Update Neck: 极简版，只更新Text，RGB和IR保持原样
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS

from .text_update import TextUpdateModule
from .text_update_multiscale import TextUpdateMultiScale
from .text_update_fusion_first import TextUpdateFusionFirst


@MODELS.register_module()
class TextOnlyUpdateNeck(BaseModule):
    """极简版Neck：只更新Text
    
    核心思路：
        - RGB特征：直接透传，不做任何修改
        - IR特征：作为辅助信息，不做修改
        - Text特征：使用原始RGB + 原始IR进行更新
    
    优势：
        - 最简单，最稳定
        - 没有复杂的物理模型
        - 没有RGB增强的额外计算
        - 只有Text更新这一个学习目标
        - 训练速度最快
    
    Args:
        in_channels: 各尺度输入通道数 [P3, P4, P5]
        text_dim: 文本特征维度
        hidden_dim: 隐藏层维度
        text_update_scale: Text更新使用的尺度 ('P3', 'P4', 'P5')
        use_multiscale_text_update: 是否使用多尺度Text更新
        multiscale_fusion_first: 多尺度时是否先融合再更新
        text_update_cfg: Text更新模块配置
        init_cfg: 初始化配置
    """
    
    def __init__(self,
                 in_channels: List[int] = [128, 256, 512],
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 text_update_scale: str = 'P4',
                 use_multiscale_text_update: bool = True,
                 multiscale_fusion_first: bool = True,
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
        
        text_update_cfg = text_update_cfg or {}
        
        # Text更新模块（唯一的学习模块）
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
            rgb_feats: 原始RGB特征（未修改，直接透传）
            text_new: 更新后的Text特征 [B, num_cls, text_dim]
        """
        if text_feats.dim() == 3:
            text = text_feats[0]
        else:
            text = text_feats
        
        # Text更新（使用原始RGB + 原始IR）
        if self.use_multiscale_text_update:
            # 多尺度更新：传入所有尺度
            text_new = self.text_update(list(rgb_feats), list(ir_feats), text)
        else:
            # 单尺度更新：使用指定尺度
            idx = self._text_update_idx
            text_new = self.text_update(rgb_feats[idx], ir_feats[idx], text)
        
        # RGB特征直接透传，不做任何修改
        return rgb_feats, text_new

