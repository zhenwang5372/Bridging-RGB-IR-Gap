# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Scale Text Update Module V4
#
# ============================================================================
# 参考 YOLO-World 的 ImagePoolingAttentionModule 设计
# ============================================================================
#
# 关键改进：
#   1. 使用 Adaptive Pooling (avg/max) 减少视觉 tokens 数量
#   2. 支持跨 Batch 聚合（可选）或每张图片独立更新
#   3. 添加 LayerNorm 稳定训练
#   4. 使用 Multi-Head Attention
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextUpdateV4(BaseModule):
    """
    Multi-Scale Text Update Module V4
    
    参考 YOLO-World 的 ImagePoolingAttentionModule 设计
    
    关键改进：
        1. 使用 Adaptive Pooling (avg/max) 提取视觉特征
        2. 支持跨 Batch 聚合（可选）或每张图片独立更新
        3. 使用 Multi-Head Attention
        4. 添加 LayerNorm 稳定训练
    
    Args:
        in_channels (List[int]): 输入特征的通道数 [P3, P4, P5]
        text_dim (int): Text embedding 维度，默认 512
        embed_channels (int): Attention 的嵌入维度，默认 256
        num_heads (int): Multi-Head Attention 的头数，默认 8
        pool_size (int): Pooling 的输出大小，默认 3 (3x3=9 tokens per level)
        with_scale (bool): 是否使用可学习的 scale 参数，默认 True
        pool_type (str): 'avg' 或 'max'，avg 支持确定性训练
        cross_batch (bool): 是否跨 Batch 聚合，默认 False
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        text_dim: int = 512,
        embed_channels: int = 256,
        num_heads: int = 8,
        pool_size: int = 3,
        with_scale: bool = True,
        pool_type: str = 'avg',  # 'max' 或 'avg'，avg 支持确定性训练
        cross_batch: bool = False,  # ⭐ 是否跨 Batch 聚合
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = len(in_channels)
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.cross_batch = cross_batch  # ⭐ 是否跨 Batch 聚合
        
        # Scale 参数
        if with_scale:
            # ⭐ 初始为 0，让模型学习何时更新
            self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        else:
            self.scale = 1.0
        
        # 投影层：将每个尺度的特征投影到统一维度
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_channels, 1, bias=False),
                nn.BatchNorm2d(embed_channels),
            )
            for ch in in_channels
        ])
        
        # Pooling：提取特征（avg 支持确定性训练，max 不支持）
        if pool_type == 'max':
            self.image_pools = nn.ModuleList([
                nn.AdaptiveMaxPool2d((pool_size, pool_size))
                for _ in range(self.num_feats)
            ])
        else:  # 'avg'
            self.image_pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d((pool_size, pool_size))
                for _ in range(self.num_feats)
            ])
        
        # Q/K/V 投影（带 LayerNorm）
        self.query = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, embed_channels)
        )
        self.key = nn.Sequential(
            nn.LayerNorm(embed_channels),
            nn.Linear(embed_channels, embed_channels)
        )
        self.value = nn.Sequential(
            nn.LayerNorm(embed_channels),
            nn.Linear(embed_channels, embed_channels)
        )
        
        # 输出投影
        self.proj = nn.Linear(embed_channels, text_dim)
        
        # 调试
        self._debug_counter = 0
    
    def forward(
        self,
        fused_feats: Tuple[torch.Tensor, ...],
        text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feats: Tuple of [B, C, H, W] - 多尺度视觉特征
            text_feats: [B, num_cls, text_dim] 或 [num_cls, text_dim]
        
        Returns:
            text_updated: [B, num_cls, text_dim] 更新后的 Text
        """
        B = fused_feats[0].shape[0]
        num_patches = self.pool_size ** 2  # 9
        
        # 处理 text_feats 维度
        if text_feats.dim() == 2:
            # [num_cls, text_dim] -> [B, num_cls, text_dim]
            text_feats = text_feats.unsqueeze(0).expand(B, -1, -1)
        
        # ⭐ Step 1: Max Pooling + 投影
        # 每个尺度: [B, C, H, W] -> [B, embed_channels, pool_size, pool_size] -> [B, embed_channels, 9]
        mlvl_image_features = []
        for feat, proj, pool in zip(fused_feats, self.projections, self.image_pools):
            pooled = pool(proj(feat))  # [B, embed_channels, 3, 3]
            pooled = pooled.view(B, self.embed_channels, num_patches)  # [B, embed_channels, 9]
            mlvl_image_features.append(pooled)
        
        # 拼接多尺度特征: [B, embed_channels, 9*3=27] -> [B, 27, embed_channels]
        image_features = torch.cat(mlvl_image_features, dim=-1).transpose(1, 2)
        # image_features: [B, 27, embed_channels]
        
        # ⭐ Step 2: Multi-Head Cross-Attention
        # Q from text, K/V from image
        q = self.query(text_feats)      # [B, num_cls, embed_channels]
        k = self.key(image_features)    # [B, 27, embed_channels]
        v = self.value(image_features)  # [B, 27, embed_channels]
        
        num_cls = q.shape[1]
        
        # Reshape for multi-head attention
        q = q.reshape(B, num_cls, self.num_heads, self.head_channels)   # [B, num_cls, num_heads, head_ch]
        k = k.reshape(B, -1, self.num_heads, self.head_channels)        # [B, 27, num_heads, head_ch]
        v = v.reshape(B, -1, self.num_heads, self.head_channels)        # [B, 27, num_heads, head_ch]
        
        # Attention: [B, num_cls, num_heads, head_ch] x [B, 27, num_heads, head_ch] -> [B, num_heads, num_cls, 27]
        # 使用 einsum 高效计算
        attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)  # [B, num_heads, num_cls, 27]
        attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)  # [B, num_heads, num_cls, 27]
        
        # Attention output
        x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)  # [B, num_cls, num_heads, head_ch]
        x = x.reshape(B, num_cls, self.embed_channels)  # [B, num_cls, embed_channels]
        
        # ⭐ Step 3: 投影
        x = self.proj(x)  # [B, num_cls, text_dim]
        
        # ⭐ Step 4: 跨 Batch 聚合（可选）
        if self.cross_batch:
            # 跨 Batch 聚合：对所有图片的更新量取平均
            x_avg = x.mean(dim=0)  # [num_cls, text_dim]
            # 取原始 text_feats 的 2D 版本
            text_feats_2d = text_feats[0]  # [num_cls, text_dim]
            # 残差更新
            text_updated_2d = text_feats_2d + self.scale * x_avg  # [num_cls, text_dim]
            # 广播回 Batch 维度
            text_updated = text_updated_2d.unsqueeze(0).expand(B, -1, -1)
        else:
            # 每张图片独立更新（原来的逻辑）
            text_updated = text_feats + self.scale * x
        
        # 调试输出
        if self.training:
            self._debug_counter += 1
            if self._debug_counter % 200 == 0:
                with torch.no_grad():
                    scale_val = self.scale.item() if isinstance(self.scale, nn.Parameter) else self.scale
                    update_norm = x.norm(dim=-1).mean().item()
                    orig_norm = text_feats.norm(dim=-1).mean().item()
                    mode_str = "cross_batch" if self.cross_batch else "per_image"
                    print(f"[TextUpdateV4] iter={self._debug_counter}: "
                          f"mode={mode_str}, "
                          f"scale={scale_val:.4f}, "
                          f"update_norm={update_norm:.4f}, "
                          f"orig_norm={orig_norm:.4f}")
        
        return text_updated
