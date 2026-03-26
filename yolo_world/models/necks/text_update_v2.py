# Copyright (c) Tencent Inc. All rights reserved.
# Text Update Module V2 for Trimodal Neck (Multi-scale I-Pooling style)
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from .trimodal_utils import IRGuidedCBAM


class TextUpdateModuleV2(BaseModule):
    """Text更新模块V2：使用多尺度I-Pooling风格的注意力机制
    
    完整流程：
        Step A: IR语义锚点 → 类别权重w（使用P4尺度）
        Step B: IR-Guided CBAM处理三个尺度的RGB
        Step C: 三个尺度特征池化并拼接
        Step D: Text作为Query，多尺度特征作为K/V
        Step E: 逐类别残差更新Text
    
    参考YOLO-World的ImagePoolingAttentionModule设计
    """
    
    def __init__(self,
                 in_channels: list = [128, 256, 512],  # P3, P4, P5的通道数
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.07,
                 gamma: float = 0.1,
                 cbam_reduction: int = 16,
                 pool_size: int = 3,
                 num_heads: int = 8,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.pool_size = pool_size
        self.num_heads = num_heads
        self.head_channels = hidden_dim // num_heads
        
        # IR语义锚点（使用P4尺度）
        self.ir_to_text = nn.Linear(in_channels[1], text_dim)
        
        # 每个尺度的IR-Guided CBAM
        self.ir_guided_cbam = nn.ModuleList([
            IRGuidedCBAM(
                channels=ch,
                reduction=cbam_reduction,
                kernel_size=7
            ) for ch in in_channels
        ])
        
        # 每个尺度的特征投影到统一维度
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, 1) for ch in in_channels
        ])
        
        # 每个尺度的池化层
        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(self.num_levels)
        ])
        
        # Multi-head Attention的Q/K/V投影
        self.query = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim)
        )
        self.key = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.value = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 输出投影
        self.proj = nn.Linear(hidden_dim, text_dim)
        
        self.scale = hidden_dim ** -0.5
        
    def forward(self,
                x_rgb_list: list,
                x_ir_list: list,
                text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb_list: RGB特征列表 [P3, P4, P5]，每个 [B, C, H, W]
            x_ir_list: IR特征列表 [P3, P4, P5]，每个 [B, C, H, W]
            text: 文本原型 [num_cls, text_dim]
        Returns:
            text_new: 更新后的文本原型 [B, num_cls, text_dim]
        """
        B = x_rgb_list[0].shape[0]
        num_cls = text.shape[0]
        num_patches = self.pool_size ** 2
        
        # Step A: IR语义锚点（使用P4）
        ir_pool = x_ir_list[1].mean(dim=[2, 3])  # P4
        u_ir = self.ir_to_text(ir_pool)
        u_ir = F.normalize(u_ir, dim=-1)
        
        logits = u_ir @ text.T / self.temperature
        w = F.softmax(logits, dim=-1)  # [B, num_cls]
        
        # Step B: IR-Guided CBAM处理所有尺度的RGB
        x_rgb_enhanced = [
            cbam(rgb, ir) 
            for cbam, rgb, ir in zip(self.ir_guided_cbam, x_rgb_list, x_ir_list)
        ]
        
        # Step C: 多尺度特征池化并拼接
        mlvl_features = [
            pool(proj(x)).view(B, -1, num_patches)  # [B, hidden_dim, 9]
            for x, proj, pool in zip(x_rgb_enhanced, self.projections, self.image_pools)
        ]
        # 拼接: [B, hidden_dim, 27] -> [B, 27, hidden_dim]
        mlvl_features = torch.cat(mlvl_features, dim=-1).transpose(1, 2)
        
        # Step D: Multi-head Attention
        Q = self.query(text)  # [num_cls, hidden_dim]
        K = self.key(mlvl_features)  # [B, 27, hidden_dim]
        V = self.value(mlvl_features)  # [B, 27, hidden_dim]
        
        # Expand text for batch
        Q = Q.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, hidden_dim]
        
        # Reshape for multi-head
        Q = Q.reshape(B, num_cls, self.num_heads, self.head_channels)  # [B, num_cls, 8, 32]
        K = K.reshape(B, -1, self.num_heads, self.head_channels)       # [B, 27, 8, 32]
        V = V.reshape(B, -1, self.num_heads, self.head_channels)       # [B, 27, 8, 32]
        
        # Attention: Q @ K^T
        attn_weight = torch.einsum('bnmc,bkmc->bmnk', Q, K)  # [B, num_cls, 8, 27]
        attn_weight = attn_weight * self.scale
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        # Attention @ V
        x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, V)  # [B, num_cls, 8, 32]
        x = x.reshape(B, num_cls, self.hidden_dim)  # [B, num_cls, hidden_dim]
        
        # 输出投影
        Y_aligned = self.proj(x)  # [B, num_cls, text_dim]
        
        # Step E: 加权残差更新
        w_expanded = w.unsqueeze(-1)  # [B, num_cls, 1]
        delta = self.gamma * w_expanded * Y_aligned
        
        text_expanded = text.unsqueeze(0).expand(B, -1, -1)
        text_new = text_expanded + delta
        
        text_new = F.normalize(text_new, dim=-1)
        
        return text_new

