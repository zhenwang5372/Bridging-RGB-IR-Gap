# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Scale Text Update Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextUpdate(BaseModule):
    """
    Multi-Scale Text Update Module (阶段3)
    
    从多尺度的RGB_class_specific中提取视觉证据，更新Text embedding。
    
    工作流程:
        1. 对每个尺度，从RGB_class_specific提取视觉证据
        2. 多尺度融合（加权平均）
        3. 跨batch聚合
        4. 残差更新Text (YOLO-World风格)
    
    Args:
        in_channels (List[int]): RGB_class_specific的通道数 [P3, P4, P5]
        text_dim (int): Text embedding维度，默认512
        num_classes (int): 类别数，默认4
        scale_init (float): 残差缩放初始值，默认0.0
        fusion_method (str): 多尺度融合方法，'learned_weight'或'equal'
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        scale_init: float = 0.0,
        fusion_method: str = 'learned_weight',
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(in_channels)
        self.fusion_method = fusion_method
        
        # 为每个尺度创建投影层 (C -> text_dim)
        self.level_projs = nn.ModuleList([
            nn.Linear(ch, text_dim)
            for ch in in_channels
        ])
        
        # 多尺度融合权重
        if fusion_method == 'learned_weight':
            # 可学习的尺度权重
            self.scale_weights = nn.Parameter(
                torch.ones(self.num_levels)
            )
        
        # 残差缩放参数 (YOLO-World风格)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        # 初始化投影层
        for proj in self.level_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        rgb_class_specific: List[torch.Tensor],
        text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_class_specific: List of [B, num_cls, C, H, W]
                来自阶段2的输出
            text_feats: [B, num_cls, text_dim] 或 [num_cls, text_dim] 原始Text embedding
        
        Returns:
            text_updated: [num_cls, text_dim] 更新后的Text
        """
        # 处理text_feats的不同形状
        if text_feats.dim() == 3:
            # [B, num_cls, text_dim] -> 取第一个batch或平均
            # 通常在训练时，batch中所有样本的text是相同的
            text_feats = text_feats[0]  # [num_cls, text_dim]
        elif text_feats.dim() == 2:
            # [num_cls, text_dim] - 正确形状
            pass
        
        # Step 1: 从每个尺度提取视觉证据
        Y_text_list = []
        
        for rgb_cs, proj in zip(rgb_class_specific, self.level_projs):
            # rgb_cs: [B, num_cls, C, H, W]
            
            # 全局平均池化
            Y_text_l = rgb_cs.mean(dim=[3, 4])
            # [B, num_cls, C]
            
            # 投影到text_dim
            Y_text_l = proj(Y_text_l)
            # [B, num_cls, text_dim]
            
            Y_text_list.append(Y_text_l)
        
        # Step 2: 多尺度融合
        if self.fusion_method == 'learned_weight':
            # Softmax归一化权重
            weights = F.softmax(self.scale_weights, dim=0)
            
            # 加权平均
            Y_text_fused = sum(
                w * Y for w, Y in zip(weights, Y_text_list)
            )
        else:  # 'equal'
            # 等权重平均
            Y_text_fused = sum(Y_text_list) / self.num_levels
        
        # [B, num_cls, text_dim]
        
        # Step 3: 跨batch聚合
        Y_text_avg = Y_text_fused.mean(dim=0)
        # [num_cls, text_dim]
        
        # Step 4: 残差更新 (YOLO-World风格)
        text_updated = text_feats + self.scale * Y_text_avg
        # [num_cls, text_dim]
        
        return text_updated

