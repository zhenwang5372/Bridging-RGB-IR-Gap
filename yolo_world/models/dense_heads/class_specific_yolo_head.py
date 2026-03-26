# Copyright (c) Tencent Inc. All rights reserved.
# Class-Specific YOLO Head Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule


@MODELS.register_module()
class ClassSpecificYOLOHeadModule(YOLOv8HeadModule):
    """
    类别特定的YOLO Head Module
    
    处理类别特定特征 [B, num_cls, C, H, W]，输出:
        - 分类分数: Region-Text相似度
        - 边界框: 每个类别独立预测
    
    Args:
        embed_dims (int): Text embedding维度
        num_classes (int): 类别数
        temperature (float): 分类的温度参数
        use_bn_head (bool): 是否使用BN
        其他参数继承自YOLOv8HeadModule
    """
    
    def __init__(
        self,
        *args,
        embed_dims: int = 512,
        num_classes: int = 4,
        temperature: float = 0.07,
        use_bn_head: bool = True,
        **kwargs
    ):
        # 确保 num_classes 传递给父类
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = num_classes
        
        super().__init__(*args, **kwargs)
        
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_bn_head = use_bn_head
        
        # 重新构建分类和回归头
        self._build_class_specific_heads()
    
    def _build_class_specific_heads(self):
        """构建类别特定的分类和回归头"""
        
        # 分类投影层: C -> embed_dims
        self.cls_projs = nn.ModuleList()
        for ch in self.in_channels:
            if self.use_bn_head:
                proj = nn.Sequential(
                    nn.Conv2d(ch, self.embed_dims, 1),
                    nn.BatchNorm2d(self.embed_dims),
                )
            else:
                proj = nn.Conv2d(ch, self.embed_dims, 1)
            self.cls_projs.append(proj)
        
        # 回归头: 共享的卷积层
        self.reg_preds = nn.ModuleList()
        for ch in self.in_channels:
            reg_pred = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, 4 * self.reg_max, 3, padding=1)
            )
            self.reg_preds.append(reg_pred)
    
    def forward(
        self,
        rgb_class_specific: List[Tensor],
        text_updated: Tensor,
        txt_masks: Tensor = None
    ) -> Tuple[List]:
        """
        Args:
            rgb_class_specific: List of [B, num_cls, C, H, W]
            text_updated: [num_cls, embed_dims]
            txt_masks: Optional text masks (for compatibility)
        
        Returns:
            cls_scores: List of [B, num_cls, H, W]
            bbox_preds: List of [B, num_cls, 4*reg_max, H, W]
            bbox_dist_preds: List of [B, num_cls, 4*reg_max, H, W] (训练时)
        """
        assert len(rgb_class_specific) == self.num_levels
        
        cls_scores = []
        bbox_preds = []
        bbox_dist_preds_list = []
        
        for level_idx, rgb_cs in enumerate(rgb_class_specific):
            # rgb_cs: [B, num_cls, C, H, W]
            B, num_cls, C, H, W = rgb_cs.shape
            
            # === 分类分支 ===
            cls_score = self._forward_cls_single(
                rgb_cs, text_updated, level_idx
            )
            # [B, num_cls, H, W]
            
            # === 回归分支 ===
            bbox_dist_pred = self._forward_reg_single(
                rgb_cs, level_idx
            )
            # [B, num_cls, 4*reg_max, H, W]
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_dist_pred)
            bbox_dist_preds_list.append(bbox_dist_pred)
        
        # 训练时返回3个元素（包含bbox_dist_preds用于loss计算）
        if self.training:
            return tuple(cls_scores), tuple(bbox_preds), tuple(bbox_dist_preds_list)
        else:
            # 推理时：将5D bbox [B, num_cls, 4*reg_max, H, W] 转换为4D [B, 4, H, W]
            # 步骤1: 选择置信度最高的类别对应的bbox预测
            # 步骤2: DFL解码将分布转换为坐标
            bbox_preds_4d = []
            for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
                # cls_score: [B, num_cls, H, W]
                # bbox_pred: [B, num_cls, 4*reg_max, H, W]
                B, num_cls, C_reg, H, W = bbox_pred.shape
                
                # 步骤1: 找到每个位置置信度最高的类别
                max_cls_idx = cls_score.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
                max_cls_idx = max_cls_idx.unsqueeze(2).expand(B, 1, C_reg, H, W)  # [B, 1, 4*reg_max, H, W]
                
                # 选择对应类别的bbox分布预测
                bbox_dist = torch.gather(bbox_pred, 1, max_cls_idx).squeeze(1)  # [B, 4*reg_max, H, W]
                
                # 步骤2: DFL解码 - 将分布转换为坐标
                # [B, 4*reg_max, H, W] -> [B, 4, reg_max, H*W] -> [B, H*W, 4, reg_max]
                bbox_dist = bbox_dist.reshape(B, 4, self.reg_max, H * W).permute(0, 3, 1, 2)
                # Softmax + 加权求和
                bbox_decoded = bbox_dist.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
                # [B, H*W, 4]
                # 转换回 [B, 4, H, W]
                bbox_decoded = bbox_decoded.transpose(1, 2).reshape(B, 4, H, W)
                
                bbox_preds_4d.append(bbox_decoded)
            
            return tuple(cls_scores), tuple(bbox_preds_4d)
    
    def _forward_cls_single(
        self,
        rgb_class_specific: Tensor,
        text_updated: Tensor,
        level_idx: int
    ) -> Tensor:
        """
        单尺度的分类分支
        
        计算Region-Text相似度作为分类分数
        """
        B, num_cls, C, H, W = rgb_class_specific.shape
        
        # 1. 投影到embed_dims
        # Reshape: [B, num_cls, C, H, W] -> [B*num_cls, C, H, W]
        rgb_flat = rgb_class_specific.view(B * num_cls, C, H, W)
        
        rgb_proj = self.cls_projs[level_idx](rgb_flat)
        # [B*num_cls, embed_dims, H, W]
        
        # Reshape back
        rgb_proj = rgb_proj.view(B, num_cls, self.embed_dims, H, W)
        # [B, num_cls, embed_dims, H, W]
        
        # 2. L2 normalize
        rgb_norm = F.normalize(rgb_proj, dim=2)
        # [B, num_cls, embed_dims, H, W]
        
        # 处理text_updated的不同形状
        if text_updated.dim() == 4:
            # [B, 1, num_cls, embed_dims] -> [num_cls, embed_dims]
            text_updated = text_updated[0, 0]
        elif text_updated.dim() == 3:
            # [B, num_cls, embed_dims] -> [num_cls, embed_dims]
            text_updated = text_updated[0]
        elif text_updated.dim() == 2:
            # [num_cls, embed_dims] - 正确形状
            pass
        
        # 确保num_cls匹配
        if text_updated.shape[0] != num_cls:
            # 如果text_updated有更多类别，只取前num_cls个
            text_updated = text_updated[:num_cls]
        
        text_norm = F.normalize(text_updated, dim=-1)
        # [num_cls, embed_dims]
        
        # 3. 扩展Text到空间维度
        text_expanded = text_norm.view(1, num_cls, self.embed_dims, 1, 1)
        # [1, num_cls, embed_dims, 1, 1]
        # 会自动broadcast到 [B, num_cls, embed_dims, H, W]
        
        # 4. 逐位置计算余弦相似度
        cls_score = (rgb_norm * text_expanded).sum(dim=2)
        # [B, num_cls, H, W]
        
        # 5. Temperature scaling
        cls_score = cls_score / self.temperature
        
        return cls_score
    
    def _forward_reg_single(
        self,
        rgb_class_specific: Tensor,
        level_idx: int
    ) -> Tensor:
        """
        单尺度的回归分支
        
        为每个类别独立预测边界框
        """
        B, num_cls, C, H, W = rgb_class_specific.shape
        
        # Reshape: [B, num_cls, C, H, W] -> [B*num_cls, C, H, W]
        rgb_flat = rgb_class_specific.view(B * num_cls, C, H, W)
        
        # 通过回归头
        bbox_pred_flat = self.reg_preds[level_idx](rgb_flat)
        # [B*num_cls, 4*reg_max, H, W]
        
        # Reshape back
        bbox_pred = bbox_pred_flat.view(B, num_cls, -1, H, W)
        # [B, num_cls, 4*reg_max, H, W]
        
        return bbox_pred

