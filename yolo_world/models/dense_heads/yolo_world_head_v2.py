# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World Head V2 with Max Pooling

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule

from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class YOLOWorldHeadModuleV2(YOLOv8HeadModule):
    """
    YOLO-World Head Module V2
    
    改进：使用Max Pooling聚合类别维度，然后使用Region-Text相似度分类
    
    处理类别特定特征 [B, num_cls, C, H, W]，输出:
        - 分类分数: Region-Text相似度 (YOLO-World风格)
        - 边界框: 标准YOLO回归
    
    Args:
        embed_dims (int): Text embedding维度
        num_classes (int): 类别数
        use_bn_head (bool): 是否使用BN
        use_einsum (bool): 是否使用einsum计算相似度
        其他参数继承自YOLOv8HeadModule
    """
    
    def __init__(
        self,
        *args,
        embed_dims: int = 512,
        num_classes: int = 4,
        use_bn_head: bool = True,
        use_einsum: bool = True,
        **kwargs
    ):
        # 确保 num_classes 传递给父类
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = num_classes
        
        super().__init__(*args, **kwargs)
        
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        
        # 重新构建分类和回归头
        self._build_yolo_world_heads()
    
    def _build_yolo_world_heads(self):
        """构建YOLO-World风格的分类和回归头"""
        
        from mmcv.cnn import ConvModule
        
        # 分类分支: 提取embedding + ContrastiveHead
        self.cls_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        
        # 回归分支: 标准YOLO
        self.reg_preds = nn.ModuleList()
        
        for i in range(self.num_levels):
            # 分类embedding提取
            cls_out_channels = max(self.in_channels[i], self.num_classes)
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.embed_dims,
                        kernel_size=1
                    )
                )
            )
            
            # ContrastiveHead
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(
                        self.embed_dims,
                        self.norm_cfg,
                        use_einsum=self.use_einsum
                    )
                )
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(
                        self.embed_dims,
                        use_einsum=self.use_einsum
                    )
                )
            
            # 回归分支
            reg_out_channels = max(self.in_channels[i] // 4, self.reg_max * 4)
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1
                    )
                )
            )
        
        # 注册projection buffer (用于DFL解码)
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)
    
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
            bbox_preds: List of [B, 4, H, W]
            (bbox_dist_preds: List of [B, 4*reg_max, H, W]) - 训练时
        """
        assert len(rgb_class_specific) == self.num_levels
        
        cls_scores = []
        bbox_preds = []
        bbox_dist_preds_list = []
        
        for level_idx, rgb_cs in enumerate(rgb_class_specific):
            # rgb_cs: [B, num_cls, C, H, W]
            
            # Step 1: Max Pooling聚合类别维度
            fused_feat, _ = torch.max(rgb_cs, dim=1)
            # [B, C, H, W]
            
            # Step 2: 分类分支 (Region-Text相似度)
            cls_score = self._forward_cls_single(
                fused_feat, text_updated, txt_masks, level_idx
            )
            # [B, num_cls, H, W]
            
            # Step 3: 回归分支
            bbox_dist_preds = self.reg_preds[level_idx](fused_feat)
            # [B, 4*reg_max, H, W]
            
            # DFL解码
            if self.reg_max > 1:
                B, _, H, W = bbox_dist_preds.shape
                bbox_dist_preds_reshaped = bbox_dist_preds.reshape(
                    B, 4, self.reg_max, H * W
                ).permute(0, 3, 1, 2)
                # [B, H*W, 4, reg_max]
                
                bbox_pred = bbox_dist_preds_reshaped.softmax(3).matmul(
                    self.proj.view([-1, 1])
                ).squeeze(-1)
                # [B, H*W, 4]
                
                bbox_pred = bbox_pred.transpose(1, 2).reshape(B, 4, H, W)
                # [B, 4, H, W]
            else:
                bbox_pred = bbox_dist_preds
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            bbox_dist_preds_list.append(bbox_dist_preds)
        
        if self.training:
            return tuple(cls_scores), tuple(bbox_preds), tuple(bbox_dist_preds_list)
        else:
            return tuple(cls_scores), tuple(bbox_preds)
    
    def _forward_cls_single(
        self,
        fused_feat: Tensor,
        text_updated: Tensor,
        txt_masks: Tensor,
        level_idx: int
    ) -> Tensor:
        """
        单尺度的分类分支 (YOLO-World风格)
        
        计算Region-Text相似度作为分类分数
        """
        B, C, H, W = fused_feat.shape
        
        # 1. 提取分类embedding
        cls_embed = self.cls_preds[level_idx](fused_feat)
        # [B, embed_dims, H, W]
        
        # 2. Region-Text对比 (ContrastiveHead)
        # text_updated: [num_cls, embed_dims] - 没有batch维度
        # 需要扩展到batch维度
        text_expanded = text_updated.unsqueeze(0).expand(B, -1, -1)
        # [B, num_cls, embed_dims]
        
        cls_logit = self.cls_contrasts[level_idx](cls_embed, text_expanded)
        # [B, num_cls, H, W]
        
        # 3. 应用text mask (如果有)
        if txt_masks is not None:
            txt_masks = txt_masks.view(B, -1, 1, 1).expand(-1, -1, H, W)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = -10e6
            else:
                cls_logit[txt_masks == 0] = -10e6
        
        return cls_logit

