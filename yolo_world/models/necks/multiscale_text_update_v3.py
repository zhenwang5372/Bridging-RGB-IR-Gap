# Copyright (c) Tencent Inc. All rights reserved.
# Multi-Scale Text Update Module V3
#
# ============================================================================
# 模块流程:
# ============================================================================
#
#   输入: fused_feats [B, C, H, W] × 3尺度,  text_feats [B, num_cls, 512]
#         │                                       │
#         ▼                                       ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Step 1: 每张图片用自己的视觉特征提取更新信号                          │
#   │         Cross-Attention: Q=Text, K=Fused, V=Fused                   │
#   │         输出: Y_text_l [B, num_cls, 512] × 3尺度                     │
#   └─────────────────────────────────────────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Step 2: 多尺度融合 (加权平均)                                        │
#   │         Y_text_fused [B, num_cls, 512]                              │
#   └─────────────────────────────────────────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Step 3: 跨Batch聚合更新量                                            │
#   │         Y_text_avg = mean(Y_text_fused, dim=0)                      │
#   │         [num_cls, 512]                                              │
#   └─────────────────────────────────────────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Step 4: 残差更新                                                     │
#   │         text_updated_2d = text_feats_2d + scale * Y_text_avg        │
#   │         [num_cls, 512]                                              │
#   └─────────────────────────────────────────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Step 5: 广播回3D (⭐ V3新增)                                         │
#   │         text_updated = text_updated_2d.expand(B, ...)               │
#   │         [B, num_cls, 512]                                           │
#   └─────────────────────────────────────────────────────────────────────┘
#         │
#         ▼
#   输出: text_updated [B, num_cls, 512]
#
# ============================================================================
# 相对V2的修改:
# ============================================================================
#
#   V2:
#     - 输出维度: [num_cls, 512] (2D)
#     - Head需要特殊处理: if txt_feats.dim() == 2: expand to 3D
#
#   V3:
#     - 输出维度: [B, num_cls, 512] (3D) ⭐
#     - Head无需修改，与原始YOLO-World完全兼容
#     - 广播操作在模块内部完成，不侵入其他模块
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextUpdateV3(BaseModule):
    """
    Multi-Scale Text Update Module V3
    
    相对V2的改进：
        - 输出维度为 [B, num_cls, text_dim]，与YOLO-World Head完全兼容
        - 广播操作在模块内部完成，不需要修改Head
    
    设计理念:
        - 每张图片用自己的视觉特征提取更新信号
        - 聚合所有图片的更新信号，得到稳定的全局更新
        - 所有图片共享更新后的text_feats
    
    Args:
        in_channels (List[int]): Fused特征的通道数 [P3, P4, P5]
        text_dim (int): Text embedding维度，默认512
        num_classes (int): 类别数，默认4
        hidden_dim (int): Cross-Attention的隐藏维度，默认256
        scale_init (float): 残差缩放初始值，默认0.0
        fusion_method (str): 多尺度融合方法，'learned_weight'或'equal'
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
        use_confidence_gate: bool = True,  # ⭐ 是否使用置信度门控
        max_update_norm: float = 0.1,      # ⭐ 更新量的最大范数
        debug: bool = False,               # ⭐ 调试模式
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_levels = len(in_channels)
        self.fusion_method = fusion_method
        self.use_confidence_gate = use_confidence_gate
        self.max_update_norm = max_update_norm
        self.debug = debug
        
        # 为每个尺度创建Cross-Attention模块
        self.level_modules = nn.ModuleList([
            SingleLevelTextUpdateV3(
                in_channels=ch,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                use_confidence_gate=use_confidence_gate,  # ⭐ 传递参数
            )
            for ch in in_channels
        ])
        
        # 多尺度融合权重
        if fusion_method == 'learned_weight':
            self.scale_weights = nn.Parameter(
                torch.ones(self.num_levels)
            )
        
        # 残差缩放参数 (YOLO-World风格)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        # 调试计数器
        self._debug_counter = 0
    
    def forward(
        self,
        fused_feats: Tuple[torch.Tensor, ...],
        text_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feats: Tuple of [B, C, H, W] - (P3, P4, P5)
                来自RGB-IR融合模块的特征
            text_feats: [B, num_cls, text_dim] 或 [num_cls, text_dim]
                原始Text embedding
        
        Returns:
            text_updated: [B, num_cls, text_dim] 更新后的Text
        """
        # 获取 batch size
        B = fused_feats[0].shape[0]
        
        # 处理text_feats的不同形状，统一为 [num_cls, text_dim] 用于计算
        if text_feats.dim() == 3:
            # [B, num_cls, text_dim] -> 取第一个batch作为原始text
            # 训练时，batch中所有样本的text是相同的
            text_feats_2d = text_feats[0]  # [num_cls, text_dim]
        elif text_feats.dim() == 2:
            # [num_cls, text_dim] - 已经是正确形状
            text_feats_2d = text_feats
        
        # Step 1: 从每个尺度提取视觉证据
        # 每张图片用自己的视觉特征去更新text
        Y_text_list = []
        
        for fused_feat, module in zip(fused_feats, self.level_modules):
            # fused_feat: [B, C, H, W]
            Y_text_l = module(
                fused_feat=fused_feat,
                text_feat=text_feats_2d  # [num_cls, text_dim]
            )
            # Y_text_l: [B, num_cls, text_dim] - B张图片各自的更新量
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
        
        # Y_text_fused: [B, num_cls, text_dim] - B张图片各自的更新量
        
        # Step 3: 跨Batch聚合更新量
        # 将B张图片的更新量聚合为一个共享的更新量
        Y_text_avg = Y_text_fused.mean(dim=0)
        # Y_text_avg: [num_cls, text_dim]
        
        # ⭐ Step 3.5: 限制更新量的范数，防止破坏原始 text embedding
        if self.max_update_norm > 0:
            update_norm = Y_text_avg.norm(dim=-1, keepdim=True)  # [num_cls, 1]
            # 如果范数超过阈值，进行缩放
            scale_factor = torch.clamp(self.max_update_norm / (update_norm + 1e-8), max=1.0)
            Y_text_avg = Y_text_avg * scale_factor
        
        # Step 4: 残差更新 (YOLO-World风格)
        text_updated_2d = text_feats_2d + self.scale * Y_text_avg
        # text_updated_2d: [num_cls, text_dim]
        
        # ⭐ 调试输出
        if self.debug and self.training:
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                with torch.no_grad():
                    orig_norm = text_feats_2d.norm(dim=-1).mean().item()
                    update_norm = Y_text_avg.norm(dim=-1).mean().item()
                    updated_norm = text_updated_2d.norm(dim=-1).mean().item()
                    scale_val = self.scale.item()
                    print(f"[TextUpdate Debug] iter={self._debug_counter}: "
                          f"scale={scale_val:.4f}, "
                          f"orig_norm={orig_norm:.4f}, "
                          f"update_norm={update_norm:.4f}, "
                          f"updated_norm={updated_norm:.4f}, "
                          f"delta={abs(updated_norm - orig_norm):.4f}")
        
        # Step 5: 广播回 [B, num_cls, text_dim] (⭐ V3新增)
        # 所有图片共享同一套更新后的text_feats
        text_updated = text_updated_2d.unsqueeze(0).expand(B, -1, -1)
        # text_updated: [B, num_cls, text_dim]
        
        return text_updated


class SingleLevelTextUpdateV3(nn.Module):
    """
    单尺度的Text Update模块 (V3)
    
    改进：添加置信度门控
        - 使用 attention score 的最大值作为置信度
        - 当某个类别在图片中不存在时，max score 会很低
        - 用置信度来调节更新量，避免污染
    
    Cross-Attention:
        Q = Text
        K, V = Fused
    """
    
    def __init__(
        self,
        in_channels: int,
        text_dim: int = 512,
        hidden_dim: int = 256,
        confidence_threshold: float = 0.0,  # 置信度阈值
        use_confidence_gate: bool = True,   # 是否使用置信度门控
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.confidence_threshold = confidence_threshold
        self.use_confidence_gate = use_confidence_gate
        
        # Query projection: Text -> hidden_dim
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        
        # Key projection: Fused -> hidden_dim
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Value projection: Fused -> hidden_dim
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # Output projection: hidden_dim -> text_dim
        self.out_proj = nn.Linear(hidden_dim, text_dim)
        
        # ⭐ 置信度门控：学习一个阈值来判断类别是否存在
        if use_confidence_gate:
            # 学习的偏置，用于调整置信度判断
            self.confidence_bias = nn.Parameter(torch.tensor(0.0))
        
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
    ) -> torch.Tensor:
        """
        Cross-Attention: Text attends to Fused features
        
        Args:
            fused_feat: [B, C, H, W] - Fused RGB-IR features
            text_feat: [num_cls, text_dim] - Text embeddings
        
        Returns:
            Y_text: [B, num_cls, text_dim] - Updated text features
        """
        B, C, H, W = fused_feat.shape
        num_cls = text_feat.shape[0]
        
        # Query from Text: [num_cls, text_dim] -> [num_cls, hidden_dim]
        Q = self.query_proj(text_feat)  # [num_cls, hidden_dim]
        
        # Key from Fused: [B, C, H, W] -> [B, hidden_dim, H, W]
        K = self.key_conv(fused_feat)  # [B, hidden_dim, H, W]
        K = K.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Value from Fused: [B, C, H, W] -> [B, hidden_dim, H, W]
        V = self.value_conv(fused_feat)  # [B, hidden_dim, H, W]
        V = V.view(B, self.hidden_dim, -1)  # [B, hidden_dim, H*W]
        
        # Expand Q for batch dimension
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, hidden_dim]
        
        # Attention: Q @ K^T
        # [B, num_cls, hidden_dim] @ [B, hidden_dim, H*W] -> [B, num_cls, H*W]
        attn_scores = torch.bmm(Q_expanded, K) / (self.hidden_dim ** 0.5)
        
        # ⭐ 计算置信度：使用 softmax 之前的 max score
        # 如果类别存在，max score 应该较高；如果不存在，max score 较低
        if self.use_confidence_gate:
            max_scores, _ = attn_scores.max(dim=-1)  # [B, num_cls]
            # 用 sigmoid 转换为 0-1 的置信度
            confidence = torch.sigmoid(max_scores + self.confidence_bias)  # [B, num_cls]
            confidence = confidence.unsqueeze(-1)  # [B, num_cls, 1]
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_cls, H*W]
        
        # Attention: attn_weights @ V
        # [B, num_cls, H*W] @ [B, H*W, hidden_dim] -> [B, num_cls, hidden_dim]
        V_transposed = V.transpose(1, 2)  # [B, H*W, hidden_dim]
        attn_output = torch.bmm(attn_weights, V_transposed)  # [B, num_cls, hidden_dim]
        
        # Output projection: [B, num_cls, hidden_dim] -> [B, num_cls, text_dim]
        Y_text = self.out_proj(attn_output)  # [B, num_cls, text_dim]
        
        # ⭐ 用置信度门控更新量
        # 如果类别不存在（置信度低），减少更新量
        if self.use_confidence_gate:
            Y_text = Y_text * confidence  # [B, num_cls, text_dim]
        
        return Y_text
