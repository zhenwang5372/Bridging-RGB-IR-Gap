# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB-IR Fusion Module V3 (Scheme 2 进阶版)
#
# ==================== V3 版本核心改进 ====================
#
# 相比 V2 版本的新增功能：
# 1. S_map 计算顺序可配置（先乘后加 vs 先加后乘）
# 2. Mask 生成方式可配置（4种方案）
#
# ==================== 接口说明 ====================
#
# 【接口1】smap_order - S_map 计算顺序
# ================================================
#
# 'multiply_first' (先乘后加):
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: A_rgb^c ⊙ A_ir^c → Σ_c(...)                   │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 对每个类别 c: hadamard_c = (w_c×A_rgb^c) ⊙ (w_c×A_ir^c)  │
#   │   2. 类别求和: S_map = Σ_c hadamard_c               │
#   │                                                     │
#   │ 特点: 保留类别级别的细粒度交互                      │
#   │ 输出: S_map [B, 1, H, W]                            │
#   └─────────────────────────────────────────────────────┘
#
# 'sum_first' (先加后乘) ⭐ 默认:
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: Σ_c(A_rgb^c) → A_rgb_agg ⊙ A_ir_agg          │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 类别聚合: A_rgb_agg = Σ_c(w_c × A_rgb^c)      │
#   │   2. 类别聚合: A_ir_agg = Σ_c(w_c × A_ir^c)        │
#   │   3. Hadamard: S_map = A_rgb_agg ⊙ A_ir_agg        │
#   │                                                     │
#   │ 特点: 先聚合类别信息，再计算一致性                  │
#   │ 输出: S_map [B, 1, H, W]                            │
#   └─────────────────────────────────────────────────────┘
#
# 【接口2】mask_method - Mask 生成方式
# ================================================
#
# 'conv_gen' (方案A: 轻量级卷积生成器) ⭐ 默认:
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: mask_raw = σ(β·X_ir + γ·S_map) → Conv细化    │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 原始公式: mask_raw = σ(β·X_ir + γ·S_map)      │
#   │      - S_map [B,1,H,W] 广播到 [B,C,H,W]            │
#   │      - 输出 mask_raw [B, C, H, W]                   │
#   │   2. 卷积细化: mask = conv_refine(mask_raw)        │
#   │      - Conv2d(C, mid, k=3) → BN → ReLU             │
#   │      - Conv2d(mid, C, k=3) → Sigmoid               │
#   │                                                     │
#   │ 特点: 先用原始公式，再用卷积学习空间特征           │
#   └─────────────────────────────────────────────────────┘
#
# 'residual' (方案B: 残差 Mask 细化):
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: mask_raw → Conv细化 → mask_raw + delta       │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 粗糙mask: mask_raw = σ(β·X_ir + γ·S_map)      │
#   │   2. 残差学习: delta = refine_conv(mask_raw)        │
#   │   3. 细化mask: mask = clamp(mask_raw + delta, 0, 1) │
#   │                                                     │
#   │ 特点: 保留原有逻辑，用残差学习增量                  │
#   └─────────────────────────────────────────────────────┘
#
# 'dual_branch' (方案C: 双分支 Mask 生成):
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: X_ir → 通道分支, S_map → 空间分支 → 融合     │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 空间分支: mask_spatial = spatial_branch(S_map)  │
#   │      → [B, 1, H, W]                                 │
#   │   2. 通道分支: mask_channel = channel_branch(X_ir)   │
#   │      → [B, C, H, W]                                 │
#   │   3. 融合: mask = mask_spatial × mask_channel       │
#   │                                                     │
#   │ 特点: 解耦空间和通道信息                            │
#   └─────────────────────────────────────────────────────┘
#
# 'se_spatial' (方案D: SE通道注意力 + 空间卷积):
#   ┌─────────────────────────────────────────────────────┐
#   │ 流程: X_ir → SE块, S_map → 空间卷积 → 组合         │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 通道注意力: se_weight = SE(X_ir) → [B, C, 1, 1] │
#   │   2. 空间细化: S_refined = spatial_conv(S_map)      │
#   │   3. 组合: mask = σ(se_weight × X_ir + S_refined)   │
#   │                                                     │
#   │ 特点: 通道注意力增强重要通道，空间卷积细化S_map    │
#   └─────────────────────────────────────────────────────┘
#
# ==================== 数据流程图 ====================
#
#                    ┌──────────────┐
#                    │   txt_feats  │ [B, N, 512]
#                    └──────┬───────┘
#                           │
#                    ┌──────▼───────┐
#                    │ text_query   │
#                    │   _proj      │
#                    └──────┬───────┘
#                           │ Q [B, N, d_k]
#          ┌────────────────┴────────────────┐
#          │                                  │
#   ┌──────▼───────┐                  ┌──────▼───────┐
#   │    x_rgb     │                  │    x_ir      │
#   │ [B,C_rgb,H,W]│                  │ [B,C_ir,H,W] │
#   └──────┬───────┘                  └──────┬───────┘
#          │                                  │
#   ┌──────▼───────┐                  ┌──────▼───────┐
#   │ rgb_key_proj │                  │ ir_key_proj  │
#   └──────┬───────┘                  └──────┬───────┘
#          │ K_rgb                           │ K_ir
#          │ [B,d_k,HW]                      │ [B,d_k,HW]
#          │                                  │
#   ┌──────▼───────┐                  ┌──────▼───────┐
#   │   Q @ K_rgb  │                  │   Q @ K_ir   │
#   │   Attention  │                  │   Attention  │
#   └──────┬───────┘                  └──────┬───────┘
#          │ attn_logits_rgb                 │ attn_logits_ir
#          │ [B, N, HW]                      │ [B, N, HW]
#          │                                  │
#   ┌──────▼───────┐                  ┌──────▼───────┐
#   │   _compute   │                  │   _compute   │
#   │     _gap     │                  │     _gap     │
#   └──────┬───────┘                  └──────┬───────┘
#          │ gap_rgb [B, N]                  │ gap_ir [B, N]
#          │                                  │
#          └────────────┬────────────────────┘
#                       │
#                ┌──────▼───────┐
#                │ class_weight │
#                │     _mlp     │
#                └──────┬───────┘
#                       │ weights [B, N, 1]
#                       │
#          ┌────────────┴────────────────────┐
#          │                                  │
#          ▼                                  ▼
#   ┌─────────────────────────────────────────────────┐
#   │              _compute_smap                       │
#   │                                                  │
#   │  smap_order='sum_first' (默认):                 │
#   │    A_rgb_agg = Σ(w × A_rgb)                     │
#   │    A_ir_agg = Σ(w × A_ir)                       │
#   │    S_map = A_rgb_agg ⊙ A_ir_agg                 │
#   │                                                  │
#   │  smap_order='multiply_first':                   │
#   │    S_map = Σ((w×A_rgb) ⊙ (w×A_ir))             │
#   └──────────────────┬──────────────────────────────┘
#                      │ S_map [B, 1, H, W]
#                      │
#   ┌──────────────────┼──────────────────────────────┐
#   │                  │                               │
#   │                  ▼                               │
#   │   ┌─────────────────────────────────────────┐   │
#   │   │           _generate_mask                 │   │
#   │   │                                          │   │
#   │   │  mask_method='conv_gen' (默认):         │   │
#   │   │    σ(β·X_ir + γ·S_map) → Conv细化       │   │
#   │   │                                          │   │
#   │   │  mask_method='residual':                │   │
#   │   │    mask_raw + refine_conv(mask_raw)     │   │
#   │   │                                          │   │
#   │   │  mask_method='dual_branch':             │   │
#   │   │    spatial(S_map) × channel(X_ir)       │   │
#   │   │                                          │   │
#   │   │  mask_method='se_spatial':              │   │
#   │   │    SE(X_ir) + spatial_conv(S_map)       │   │
#   │   └──────────────────┬──────────────────────┘   │
#   │                      │ mask [B, C_rgb, H, W]    │
#   │                      │                          │
#   └──────────────────────┼──────────────────────────┘
#                          │
#                   ┌──────▼───────┐
#                   │   x_fused =  │
#                   │ x_rgb × mask │
#                   │   + x_rgb    │
#                   └──────┬───────┘
#                          │
#                   ┌──────▼───────┐
#                   │   x_fused    │
#                   │[B,C_rgb,H,W] │
#                   └──────────────┘
#
# ==================== 使用方法 ====================
#
# 在配置文件中设置：
# text_guided_fusion=dict(
#     type='TextGuidedRGBIRFusionV3',
#     gap_method='logits',          # 'logits' | 'max' | 'entropy'
#     smap_method='sigmoid',        # 'sigmoid' | 'sigmoid_temp' | 'normalized'
#     smap_order='sum_first',       # 'sum_first' | 'multiply_first'  ⭐ V3新增
#     mask_method='conv_gen',       # 'conv_gen' | 'residual' | 'dual_branch' | 'se_spatial'  ⭐ V3新增
#     mask_reduction=8,             # mask_method='conv_gen' 的通道缩减比例  ⭐ V3新增
#     ...
# )

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SingleLevelTextGuidedFusionV3(nn.Module):
    """
    单尺度的文本引导 RGB-IR 融合模块 V3（方案二进阶版）
    
    相比 V2 的新增功能：
    1. smap_order: S_map 计算顺序（先乘后加 vs 先加后乘）
    2. mask_method: Mask 生成方式（4种方案）
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        beta (float): X_ir 的权重系数，默认 1.0
        gamma (float): S_map 的权重系数，默认 0.5
        gap_method (str): GAP 计算方式 ('logits' | 'max' | 'entropy')
        smap_method (str): S_map 归一化方式 ('sigmoid' | 'sigmoid_temp' | 'normalized')
        smap_order (str): S_map 计算顺序 ('sum_first' | 'multiply_first')
        mask_method (str): Mask 生成方式 ('conv_gen' | 'residual' | 'dual_branch' | 'se_spatial')
        mask_reduction (int): mask_method='conv_gen' 的通道缩减比例
        temperature (float): 温度参数
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
        gap_method: Literal['logits', 'max', 'entropy'] = 'logits',
        smap_method: Literal['sigmoid', 'sigmoid_temp', 'normalized'] = 'sigmoid',
        smap_order: Literal['sum_first', 'multiply_first'] = 'sum_first',
        mask_method: Literal['conv_gen', 'residual', 'dual_branch', 'se_spatial'] = 'conv_gen',
        mask_reduction: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.gap_method = gap_method
        self.smap_method = smap_method
        self.smap_order = smap_order
        self.mask_method = mask_method
        
        # ===== Step 1: Query/Key 投影 =====
        d_k = 128
        self.d_k = d_k
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1, bias=False)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1, bias=False)
        
        # ===== Step 2: 类别权重计算 MLP =====
        self.class_weight_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # ===== 温度参数 =====
        if smap_method == 'sigmoid_temp':
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # ===== Step C: 门控参数（用于部分 mask_method）=====
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
        # ===== IR 通道对齐 =====
        if ir_channels != rgb_channels:
            self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False)
        else:
            self.ir_align = nn.Identity()
        
        # ===== Mask 生成模块（根据 mask_method 初始化）=====
        self._init_mask_modules(rgb_channels, mask_reduction)
        
        # 打印配置信息
        print(f"[SingleLevelTextGuidedFusionV3] rgb_ch={rgb_channels}, ir_ch={ir_channels}")
        print(f"  gap_method={gap_method}, smap_method={smap_method}")
        print(f"  smap_order={smap_order}, mask_method={mask_method}")
    
    def _init_mask_modules(self, rgb_channels: int, reduction: int):
        """根据 mask_method 初始化相应的模块"""
        
        if self.mask_method == 'conv_gen':
            # 方案A: 轻量级卷积生成器
            # 流程: mask_raw = σ(β·X_ir + γ·S_map) → Conv网络 → mask
            # 先用原始公式得到 mask_raw，再过卷积细化
            mid_channels = max(rgb_channels // reduction, 8)
            self.mask_refine_conv = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()  # 输出范围 [0, 1]
            )
            
        elif self.mask_method == 'residual':
            # 方案B: 残差 Mask 细化
            mid_channels = max(rgb_channels // reduction, 8)
            self.mask_refine = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
                nn.Tanh()  # 输出范围 [-1, 1] 作为残差
            )
            
        elif self.mask_method == 'dual_branch':
            # 方案C: 双分支 Mask 生成
            # 空间分支：处理 S_map
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )
            # 通道分支：处理 X_ir
            mid_channels = max(rgb_channels // reduction, 8)
            self.channel_branch = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            
        elif self.mask_method == 'se_spatial':
            # 方案D: SE通道注意力 + 空间卷积
            self.se_block = SEBlock(rgb_channels, reduction=max(reduction, 4))
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
            )
    
    def _compute_gap(
        self, 
        attn_logits: torch.Tensor, 
        attn_probs: torch.Tensor,
        H: int, 
        W: int
    ) -> torch.Tensor:
        """计算类别重要性的 GAP 值（与 V2 相同）"""
        if self.gap_method == 'logits':
            gap = attn_logits.mean(dim=-1)
        elif self.gap_method == 'max':
            gap = attn_probs.max(dim=-1)[0]
        elif self.gap_method == 'entropy':
            entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(H * W, dtype=torch.float32, device=attn_probs.device))
            gap = 1 - entropy / max_entropy
        else:
            raise ValueError(f"Unknown gap_method: {self.gap_method}")
        return gap
    
    def _compute_smap(
        self, 
        attn_logits_rgb: torch.Tensor,
        attn_logits_ir: torch.Tensor,
        weights: torch.Tensor,
        B: int, N: int, H: int, W: int
    ) -> torch.Tensor:
        """
        计算 S_map（语义一致性图）
        
        支持两种计算顺序：
        - 'multiply_first': 先乘后加（V2 的默认逻辑）
        - 'sum_first': 先加后乘（V3 新增，默认）
        """
        d_k_sqrt = self.d_k ** 0.5
        
        # 根据 smap_method 计算归一化的 attention
        if self.smap_method == 'sigmoid_temp':
            temp = torch.abs(self.temperature) + 1e-6
            A_rgb = torch.sigmoid(attn_logits_rgb / temp)
            A_ir = torch.sigmoid(attn_logits_ir / temp)
        else:
            A_rgb = torch.sigmoid(attn_logits_rgb / d_k_sqrt)
            A_ir = torch.sigmoid(attn_logits_ir / d_k_sqrt)
        
        if self.smap_order == 'multiply_first':
            # ========== 先乘后加（V2 逻辑）==========
            # 公式: S_map = Σ_c ((w_c × A_rgb^c) ⊙ (w_c × A_ir^c))
            A_rgb_weighted = weights * A_rgb  # [B, N, HW]
            A_ir_weighted = weights * A_ir    # [B, N, HW]
            hadamard = A_rgb_weighted * A_ir_weighted  # [B, N, HW]
            S_map_flat = hadamard.sum(dim=1, keepdim=True)  # [B, 1, HW]
            
        elif self.smap_order == 'sum_first':
            # ========== 先加后乘（V3 新增，默认）==========
            # 公式: 
            #   A_rgb_agg = Σ_c (w_c × A_rgb^c)
            #   A_ir_agg = Σ_c (w_c × A_ir^c)
            #   S_map = A_rgb_agg ⊙ A_ir_agg
            A_rgb_weighted = weights * A_rgb  # [B, N, HW]
            A_ir_weighted = weights * A_ir    # [B, N, HW]
            
            # 先按类别求和
            A_rgb_agg = A_rgb_weighted.sum(dim=1, keepdim=True)  # [B, 1, HW]
            A_ir_agg = A_ir_weighted.sum(dim=1, keepdim=True)    # [B, 1, HW]
            
            # 再做 Hadamard 积
            S_map_flat = A_rgb_agg * A_ir_agg  # [B, 1, HW]
        
        else:
            raise ValueError(f"Unknown smap_order: {self.smap_order}")
        
        S_map = S_map_flat.view(B, 1, H, W)
        
        # 根据 smap_method 进行后处理
        if self.smap_method == 'normalized':
            S_map_mean = S_map.mean(dim=[2, 3], keepdim=True)
            S_map = torch.sigmoid(S_map - S_map_mean)
        else:
            S_map = S_map / N  # 归一化到 [0, 1]
        
        return S_map
    
    def _generate_mask(
        self,
        x_ir_aligned: torch.Tensor,
        S_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成 Mask
        
        根据 mask_method 使用不同的生成方式：
        - 'conv_gen': 轻量级卷积生成器
        - 'residual': 残差细化
        - 'dual_branch': 双分支
        - 'se_spatial': SE + 空间卷积
        """
        B, C, H, W = x_ir_aligned.shape
        
        if self.mask_method == 'conv_gen':
            # ========== 方案A: 轻量级卷积生成器 ==========
            # 流程: 
            #   1. mask_raw = σ(β·X_ir + γ·S_map)  (原始公式)
            #   2. mask = conv_refine(mask_raw)    (卷积细化)
            
            # Step 1: 原始公式计算 mask_raw
            # S_map [B, 1, H, W] 会自动广播到 [B, C, H, W]
            mask_raw = torch.sigmoid(self.beta * x_ir_aligned + self.gamma * S_map)  # [B, C, H, W]
            
            # Step 2: 卷积细化
            mask = self.mask_refine_conv(mask_raw)  # [B, C, H, W]
            
        elif self.mask_method == 'residual':
            # ========== 方案B: 残差 Mask 细化 ==========
            # 1. 粗糙 mask
            mask_raw = torch.sigmoid(self.beta * x_ir_aligned + self.gamma * S_map)
            # 2. 残差学习
            delta = self.mask_refine(mask_raw)  # [B, C, H, W], 范围 [-1, 1]
            # 3. 细化 mask
            mask = torch.clamp(mask_raw + 0.1 * delta, 0, 1)  # 0.1 是残差缩放因子
            
        elif self.mask_method == 'dual_branch':
            # ========== 方案C: 双分支 Mask 生成 ==========
            # 1. 空间分支处理 S_map
            mask_spatial = self.spatial_branch(S_map)  # [B, 1, H, W]
            # 2. 通道分支处理 X_ir
            mask_channel = self.channel_branch(x_ir_aligned)  # [B, C, H, W]
            # 3. 融合
            mask = mask_spatial * mask_channel  # [B, C, H, W]
            
        elif self.mask_method == 'se_spatial':
            # ========== 方案D: SE通道注意力 + 空间卷积 ==========
            # 1. SE 通道注意力
            se_weight = self.se_block(x_ir_aligned)  # [B, C, 1, 1]
            x_ir_se = se_weight * x_ir_aligned  # [B, C, H, W]
            # 2. 空间卷积细化 S_map
            S_refined = self.spatial_conv(S_map)  # [B, 1, H, W]
            # 3. 组合生成 mask
            mask = torch.sigmoid(self.beta * x_ir_se + self.gamma * S_refined)
            
        else:
            raise ValueError(f"Unknown mask_method: {self.mask_method}")
        
        return mask
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_rgb: RGB 特征 [B, C_rgb, H, W]
            x_ir: IR 特征 [B, C_ir, H, W]
            txt_feats: 文本特征 [B, N, d]
        
        Returns:
            x_fused: 融合后的特征 [B, C_rgb, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        N = txt_feats.size(1)
        
        # ===== Step 1: 计算 Attention Logits =====
        Q = self.text_query_proj(txt_feats)
        
        K_rgb = self.rgb_key_proj(x_rgb)
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)
        
        x_ir_resized = x_ir
        if x_ir.shape[-2:] != (H, W):
            x_ir_resized = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        K_ir = self.ir_key_proj(x_ir_resized)
        K_ir_flat = K_ir.view(B, self.d_k, H * W)
        
        d_k_sqrt = self.d_k ** 0.5
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat) / d_k_sqrt
        attn_logits_ir = torch.bmm(Q, K_ir_flat) / d_k_sqrt
        
        attn_probs_rgb = F.softmax(attn_logits_rgb, dim=-1)
        attn_probs_ir = F.softmax(attn_logits_ir, dim=-1)
        
        # ===== Step 2: 计算类别权重 w_c =====
        gap_rgb = self._compute_gap(attn_logits_rgb, attn_probs_rgb, H, W)
        gap_ir = self._compute_gap(attn_logits_ir, attn_probs_ir, H, W)
        
        weights = []
        for c in range(N):
            class_input = torch.stack([gap_rgb[:, c], gap_ir[:, c]], dim=-1)
            w_c = self.class_weight_mlp(class_input)
            weights.append(w_c)
        weights = torch.stack(weights, dim=1)
        
        # ===== Step 3: 计算 S_map =====
        S_map = self._compute_smap(attn_logits_rgb, attn_logits_ir, weights, B, N, H, W)
        
        # ===== Step C: 生成 Mask =====
        x_ir_aligned = self.ir_align(x_ir_resized)
        mask = self._generate_mask(x_ir_aligned, S_map)
        
        # ===== Step D: 最终融合 =====
        x_fused = x_rgb * mask + x_rgb
        
        return x_fused


@MODELS.register_module()
class TextGuidedRGBIRFusionV3(BaseModule):
    """
    Text-guided RGB-IR Fusion Module V3 (方案二进阶版)
    
    多尺度文本引导融合模块，为每个金字塔层级独立应用融合。
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表
        ir_channels (List[int]): IR 特征通道数列表
        text_dim (int): 文本特征维度
        num_classes (int): 类别数
        beta (float): X_ir 的权重系数
        gamma (float): S_map 的权重系数
        gap_method (str): GAP 计算方式
        smap_method (str): S_map 归一化方式
        smap_order (str): S_map 计算顺序 ('sum_first' | 'multiply_first')
        mask_method (str): Mask 生成方式
        mask_reduction (int): Mask 生成器的通道缩减比例
        temperature (float): 温度参数
    """
    
    def __init__(
        self,
        rgb_channels: List[int] = [128, 256, 512],
        ir_channels: List[int] = [64, 128, 256],
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
        gap_method: str = 'logits',
        smap_method: str = 'sigmoid',
        smap_order: str = 'sum_first',
        mask_method: str = 'conv_gen',
        mask_reduction: int = 8,
        temperature: float = 1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.num_levels = len(rgb_channels)
        
        print(f"\n{'='*60}")
        print(f"[TextGuidedRGBIRFusionV3] 初始化配置:")
        print(f"  - RGB channels: {rgb_channels}")
        print(f"  - IR channels: {ir_channels}")
        print(f"  - gap_method: {gap_method}")
        print(f"  - smap_method: {smap_method}")
        print(f"  - smap_order: {smap_order}  ⭐ V3新增")
        print(f"  - mask_method: {mask_method}  ⭐ V3新增")
        print(f"  - mask_reduction: {mask_reduction}")
        print(f"{'='*60}\n")
        
        self.fusion_modules = nn.ModuleList([
            SingleLevelTextGuidedFusionV3(
                rgb_channels=rgb_channels[i],
                ir_channels=ir_channels[i],
                text_dim=text_dim,
                num_classes=num_classes,
                beta=beta,
                gamma=gamma,
                gap_method=gap_method,
                smap_method=smap_method,
                smap_order=smap_order,
                mask_method=mask_method,
                mask_reduction=mask_reduction,
                temperature=temperature,
            )
            for i in range(self.num_levels)
        ])
    
    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: RGB 特征元组 (P3, P4, P5)
            ir_feats: IR 特征元组 (P3, P4, P5)
            txt_feats: 文本特征 [B, N, d]
        
        Returns:
            fused_feats: 融合后的特征元组
        """
        fused_feats = []
        
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](
                x_rgb=rgb_feats[i],
                x_ir=ir_feats[i],
                txt_feats=txt_feats,
            )
            fused_feats.append(fused)
        
        return tuple(fused_feats)
