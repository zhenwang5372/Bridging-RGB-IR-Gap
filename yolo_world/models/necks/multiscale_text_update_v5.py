# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Scale Text Update Module V5
#
# ============================================================================
# 带 Class Mask 的 Text Update（全量 Attention 版本）
# ============================================================================
#
# 核心设计：
#   1. 使用全量 attention（像 V3 一样，不用 Pooling）
#   2. 支持两种 mask 方式（可配置）：
#      - Class Mask: 使用 GT labels（默认，use_confidence_gate=False）
#      - Confidence Gate: 使用 attention 最大值计算置信度
#   3. 支持跨 batch 聚合开关（cross_batch）
#
# 配置示例：
#   cross_batch=True, use_confidence_gate=False  → Class Mask + 跨 Batch（默认）
#   cross_batch=False, use_confidence_gate=False → Class Mask + 每图独立
#   cross_batch=True, use_confidence_gate=True   → 置信度门控 + 跨 Batch
#   cross_batch=False, use_confidence_gate=True  → 置信度门控 + 每图独立
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextUpdateV5(BaseModule):
    """
    Multi-Scale Text Update Module V5 (带 Class Mask，全量 Attention)
    
    核心改进：
        1. 使用全量 attention（全部空间位置，像 V3）
        2. 支持两种 mask 方式：Class Mask（默认）或置信度门控
        3. 支持跨 batch 聚合开关
    
    Args:
        in_channels (List[int]): 输入特征的通道数 [P3, P4, P5]
        text_dim (int): Text embedding 维度，默认 512
        num_classes (int): 类别数，默认 4
        hidden_dim (int): Cross-Attention 的隐藏维度，默认 256
        scale_init (float): 残差缩放初始值，默认 0.0
        fusion_method (str): 多尺度融合方法，'learned_weight' 或 'equal'
        cross_batch (bool): 是否跨 Batch 聚合，默认 True
        use_confidence_gate (bool): 是否使用置信度门控，默认 False（使用 Class Mask）
        confidence_bias (float): 置信度门控的偏置初始值，默认 0.0（可学习参数）
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
        scale_init: float = 0.0,
        fusion_method: str = 'learned_weight',
        cross_batch: bool = True,  # ⭐ 跨 Batch 聚合开关
        use_confidence_gate: bool = False,  # ⭐ 置信度门控开关
        confidence_bias: float = 0.0,  # 置信度偏置
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_levels = len(in_channels)
        self.fusion_method = fusion_method
        self.cross_batch = cross_batch  # ⭐ 跨 batch 聚合开关
        self.use_confidence_gate = use_confidence_gate  # ⭐ 置信度门控开关
        
        # ⭐ 置信度偏置改为可学习参数
        if use_confidence_gate:
            self.confidence_bias = nn.Parameter(torch.tensor(confidence_bias))
        
        # 为每个尺度创建 Cross-Attention 模块（置信度门控需要返回 attention weights）
        self.level_modules = nn.ModuleList([
            SingleLevelTextUpdateV5(
                in_channels=ch,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                return_attn_weights=use_confidence_gate
            )
            for ch in in_channels
        ])
        
        # 多尺度融合权重
        if fusion_method == 'learned_weight':
            self.scale_weights = nn.Parameter(torch.ones(self.num_levels))
        
        # 残差缩放参数
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        # 调试
        self._debug_counter = 0
    
    def forward(
        self,
        fused_feats: Tuple[torch.Tensor, ...],
        text_feats: torch.Tensor,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            fused_feats: Tuple of [B, C, H, W] - 多尺度视觉特征
            text_feats: [B, num_cls, text_dim] 或 [num_cls, text_dim]
            gt_labels: List of [N_i] - 每张图片的 GT 类别索引（训练时使用，仅 Class Mask 模式）
        
        Returns:
            text_updated: [B, num_cls, text_dim] 更新后的 Text
        """
        B = fused_feats[0].shape[0]
        
        # 处理 text_feats 维度
        if text_feats.dim() == 3:
            text_feats_2d = text_feats[0]  # [num_cls, text_dim]
        else:
            text_feats_2d = text_feats
        
        num_cls = text_feats_2d.shape[0]
        
        # Step 1: 每个尺度提取更新量（全量 attention）
        Y_text_list = []
        attn_weights_list = []  # 用于置信度门控
        
        for fused_feat, module in zip(fused_feats, self.level_modules):
            if self.use_confidence_gate:
                # 置信度门控模式：需要返回 attention weights
                Y_text_l, attn_weights_l = module(
                    fused_feat=fused_feat,
                    text_feat=text_feats_2d
                )
                attn_weights_list.append(attn_weights_l)
            else:
                # Class Mask 模式
                Y_text_l = module(
                    fused_feat=fused_feat,
                    text_feat=text_feats_2d
                )
            # Y_text_l: [B, num_cls, text_dim]
            Y_text_list.append(Y_text_l)
        
        # Step 2: 多尺度融合
        if self.fusion_method == 'learned_weight':
            weights = F.softmax(self.scale_weights, dim=0)
            Y_text_fused = sum(w * Y for w, Y in zip(weights, Y_text_list))
        else:
            Y_text_fused = sum(Y_text_list) / self.num_levels
        # Y_text_fused: [B, num_cls, text_dim]
        
        # ⭐ Step 3: 选择 Mask 方式
        if self.use_confidence_gate:
            # 置信度门控模式：用 attention 最大值计算置信度
            # 注意：不同尺度的 attention weights 大小不同（6400, 1600, 400）
            # 所以需要先计算每个尺度的 max_attn，再融合
            
            # 每个尺度独立计算 max_attn
            max_attn_list = []
            for attn_w in attn_weights_list:
                # attn_w: [B, num_cls, H*W]，H*W 每个尺度不同
                max_attn_l, _ = attn_w.max(dim=-1)  # [B, num_cls]
                max_attn_list.append(max_attn_l)
            
            # 融合多尺度的 max_attn
            if self.fusion_method == 'learned_weight':
                max_attn_fused = sum(w * m for w, m in zip(weights, max_attn_list))
            else:
                max_attn_fused = sum(max_attn_list) / self.num_levels
            # max_attn_fused: [B, num_cls]
            
            # 转换为置信度
            confidence = torch.sigmoid(max_attn_fused + self.confidence_bias)  # [B, num_cls]
            mask = confidence.unsqueeze(-1)  # [B, num_cls, 1]
            mask_type = "confidence"
        else:
            # Class Mask 模式：用 GT labels
            if gt_labels is not None and self.training:
                class_mask = self._create_class_mask(gt_labels, num_cls, text_feats_2d.device)
            else:
                class_mask = torch.ones(B, num_cls, device=text_feats_2d.device)
            mask = class_mask.unsqueeze(-1)  # [B, num_cls, 1]
            mask_type = "class_mask"
        
        # Step 4: 应用 mask
        Y_text_masked = Y_text_fused * mask  # [B, num_cls, text_dim]
        
        # ⭐ Step 5: 跨 Batch 聚合（可选）
        if self.cross_batch:
            # 跨 Batch 聚合模式
            if self.use_confidence_gate:
                # 置信度门控：直接平均所有图片
                Y_text_avg = Y_text_masked.mean(dim=0)  # [num_cls, text_dim]
            else:
                # Class Mask：按类别出现次数加权平均
                class_count = mask.sum(dim=0).clamp(min=1)  # [num_cls, 1]
                Y_text_avg = Y_text_masked.sum(dim=0) / class_count  # [num_cls, text_dim]
            
            # 残差更新
            text_updated_2d = text_feats_2d + self.scale * Y_text_avg  # [num_cls, text_dim]
            # 广播回 Batch 维度
            text_updated = text_updated_2d.unsqueeze(0).expand(B, -1, -1)
        else:
            # 每张图片独立更新
            text_feats_3d = text_feats_2d.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, text_dim]
            text_updated = text_feats_3d + self.scale * Y_text_masked
        
        # 调试输出
        if self.training:
            self._debug_counter += 1
            if self._debug_counter % 200 == 0:
                with torch.no_grad():
                    scale_val = self.scale.item()
                    mask_ratio = mask.mean().item()
                    update_norm = Y_text_masked.norm(dim=-1).mean().item()
                    mode_str = f"cross_batch={self.cross_batch}, {mask_type}"
                    print(f"[TextUpdateV5] iter={self._debug_counter}: "
                          f"mode=({mode_str}), "
                          f"scale={scale_val:.4f}, "
                          f"mask_ratio={mask_ratio:.2%}, "
                          f"update_norm={update_norm:.4f}")
        
        return text_updated
    
    def _create_class_mask(
        self,
        gt_labels: List[torch.Tensor],
        num_cls: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        根据 GT labels 创建 class_mask
        
        Args:
            gt_labels: List of [N_i] - 每张图片的 GT 类别索引
            num_cls: 类别总数
            device: 设备
        
        Returns:
            class_mask: [B, num_cls] - 1 表示类别存在，0 表示不存在
        """
        B = len(gt_labels)
        class_mask = torch.zeros(B, num_cls, device=device)
        
        for i, labels in enumerate(gt_labels):
            if labels is not None and len(labels) > 0:
                unique_labels = labels.unique()
                valid_labels = unique_labels[unique_labels < num_cls]
                class_mask[i, valid_labels.long()] = 1.0
        
        return class_mask


class SingleLevelTextUpdateV5(nn.Module):
    """
    单尺度的 Text Update 模块（全量 Attention）
    
    Cross-Attention:
        Q = Text
        K, V = Fused（全部空间位置）
    """
    
    def __init__(
        self,
        in_channels: int,
        text_dim: int = 512,
        hidden_dim: int = 256,
        return_attn_weights: bool = False  # ⭐ 是否返回 attention weights
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.return_attn_weights = return_attn_weights
        
        # Query projection: Text -> hidden_dim
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        
        # Key projection: Fused -> hidden_dim
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Value projection: Fused -> hidden_dim
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Output projection: hidden_dim -> text_dim
        self.out_proj = nn.Linear(hidden_dim, text_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.zeros_(self.key_conv.bias)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.zeros_(self.value_conv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        fused_feat: torch.Tensor,
        text_feat: torch.Tensor
    ):
        """
        全量 Cross-Attention: Text attends to 全部空间位置
        
        Args:
            fused_feat: [B, C, H, W] - Fused RGB-IR features
            text_feat: [num_cls, text_dim] - Text embeddings
        
        Returns:
            Y_text: [B, num_cls, text_dim] - Updated text features
            attn_weights (optional): [B, num_cls, H*W] - Attention weights (if return_attn_weights=True)
        """
        B, C, H, W = fused_feat.shape
        num_cls = text_feat.shape[0]
        
        # Query from Text: [num_cls, text_dim] -> [num_cls, hidden_dim]
        Q = self.query_proj(text_feat)  # [num_cls, hidden_dim]
        
        # Key from Fused: [B, C, H, W] -> [B, hidden_dim, H*W]
        K = self.key_conv(fused_feat)  # [B, hidden_dim, H, W]
        K = K.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Value from Fused: [B, C, H, W] -> [B, hidden_dim, H*W]
        V = self.value_conv(fused_feat)  # [B, hidden_dim, H, W]
        V = V.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Expand Q for batch dimension
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, hidden_dim]
        
        # Attention: Q @ K^T
        # [B, num_cls, hidden_dim] @ [B, hidden_dim, H*W] -> [B, num_cls, H*W]
        attn_scores = torch.bmm(Q_expanded, K) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_cls, H*W]
        
        # Attention output
        # [B, num_cls, H*W] @ [B, H*W, hidden_dim] -> [B, num_cls, hidden_dim]
        V_transposed = V.transpose(1, 2)  # [B, H*W, hidden_dim]
        attn_output = torch.bmm(attn_weights, V_transposed)  # [B, num_cls, hidden_dim]
        
        # Output projection
        Y_text = self.out_proj(attn_output)  # [B, num_cls, text_dim]
        
        if self.return_attn_weights:
            return Y_text, attn_weights
        return Y_text
