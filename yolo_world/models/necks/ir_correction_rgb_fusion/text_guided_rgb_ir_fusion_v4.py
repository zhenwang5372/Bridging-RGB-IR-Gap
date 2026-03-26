# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB-IR Fusion Module V4
#
# ==================== V4 版本核心改进 ====================
#
# 相比 V3 版本的新增功能：
# 1. param_constraint: β和γ参数约束方式（解决学习到负值的问题）
# 2. mask_center: mask 零中心化方式（解决全正值问题，增加抑制能力）
#
# ==================== V4 新增接口说明 ====================
#
# 【接口1】param_constraint - β和γ参数约束方式
# ================================================
#
# 问题背景：V3 中 β 和 γ 是无约束的可学习参数，训练时可能学习到负值。
# 当 β < 0 时，mask_raw = σ(β·X_ir + γ·S_map) 会导致目标区域 mask 值低，
# 背景区域 mask 值高，与预期相反。
#
# 'softplus' ⭐ 默认:
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: β_pos = softplus(β), γ_pos = softplus(γ)     │
#   │                                                     │
#   │ softplus(x) = log(1 + exp(x))                      │
#   │   - 当 x → -∞ 时，softplus(x) → 0+                 │
#   │   - 当 x → +∞ 时，softplus(x) ≈ x                  │
#   │   - 平滑、可导、确保 > 0                            │
#   │                                                     │
#   │ 公式: mask_raw = σ(β_pos·X_ir + γ_pos·S_map)       │
#   │ 效果: 目标区域 mask 高，背景区域 mask 低            │
#   └─────────────────────────────────────────────────────┘
#
# 'abs':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: β_pos = |β|, γ_pos = |γ|                     │
#   │                                                     │
#   │ 特点: 简单直接，但在 0 附近不可导                   │
#   │ 公式: mask_raw = σ(|β|·X_ir + |γ|·S_map)           │
#   └─────────────────────────────────────────────────────┘
#
# 'residual_alpha':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: 不约束 β/γ，改用带符号的残差融合              │
#   │                                                     │
#   │ 原始: x_fused = x_rgb * mask + x_rgb               │
#   │ 改为: x_fused = x_rgb + α * (x_rgb * mask)         │
#   │                                                     │
#   │ α 是可学习参数，可以学习补偿 mask 的方向            │
#   │ 即使 mask 学反了，α 也可以学习到负值来补偿          │
#   └─────────────────────────────────────────────────────┘
#
# 'none':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: 不约束（与 V3 行为一致）                      │
#   │                                                     │
#   │ β 和 γ 可以是任意值（包括负值）                     │
#   └─────────────────────────────────────────────────────┘
#
# 【接口2】mask_center - Mask 零中心化方式
# ================================================
#
# 问题背景：V3 中由于使用 sigmoid，mask 值全部 > 0，
# 导致所有区域都被增强，只是程度不同，缺乏"抑制背景"的能力。
#
# 'spatial_mean' ⭐ 默认:
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: mask_centered = mask - mean(mask)            │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 计算 mask 的空间均值: μ = mean(mask, [H, W])  │
#   │   2. 减去均值: mask_centered = mask - μ            │
#   │                                                     │
#   │ 效果:                                               │
#   │   - 高于均值的区域 → 正值 → 增强                    │
#   │   - 低于均值的区域 → 负值 → 抑制                    │
#   │   - 零和性质: sum(mask_centered) ≈ 0               │
#   │                                                     │
#   │ 融合: x_fused = x_rgb + x_rgb * mask_centered      │
#   └─────────────────────────────────────────────────────┘
#
# 'tanh':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: 用 tanh 代替 sigmoid 作为最后的激活函数       │
#   │                                                     │
#   │ sigmoid 输出: (0, 1) → 全正                         │
#   │ tanh 输出: (-1, 1) → 有正有负                       │
#   │                                                     │
#   │ 修改 conv_refine 最后一层为 Tanh()                  │
#   │ 效果: mask 天然具有正负值                           │
#   └─────────────────────────────────────────────────────┘
#
# 'smap_center':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: 在 S_map 阶段就进行零中心化                   │
#   │                                                     │
#   │ 步骤:                                               │
#   │   1. 计算 S_map                                     │
#   │   2. S_map_centered = S_map - mean(S_map)          │
#   │   3. mask_raw = σ(β·X_ir + γ·S_map_centered)       │
#   │                                                     │
#   │ 效果: S_map 有正有负，影响后续 mask 生成            │
#   └─────────────────────────────────────────────────────┘
#
# 'none':
#   ┌─────────────────────────────────────────────────────┐
#   │ 方法: 不零中心化（与 V3 行为一致）                  │
#   │                                                     │
#   │ mask 值全部 > 0                                     │
#   └─────────────────────────────────────────────────────┘
#
# ==================== 继承自 V3 的接口 ====================
#
# 【接口3】gap_method - GAP 计算方式
# 【接口4】smap_method - S_map 归一化方式
# 【接口5】smap_order - S_map 计算顺序
# 【接口6】mask_method - Mask 生成方式 (V4 扩展)
#
# 详见 V3 文档。
#
# ==================== V4 对 mask_method 的扩展 ====================
#
# V3 原有: 'conv_gen', 'residual', 'dual_branch', 'se_spatial'
#
# V4 扩展:
# 'conv_gen' → 分为 'conv_gen_1x1' 和 'conv_gen_3x3'
#
# 'conv_gen_3x3' (原 'conv_gen'):
#   ┌─────────────────────────────────────────────────────┐
#   │ 使用 3x3 卷积进行 mask 细化                         │
#   │                                                     │
#   │ mask_raw = σ(β·X_ir + γ·S_map)                     │
#   │ mask = Conv3x3(C→mid) → BN → ReLU                  │
#   │      → Conv3x3(mid→C) → Sigmoid/Tanh               │
#   │                                                     │
#   │ 优点: 3x3 感受野更大，能捕获局部空间特征            │
#   │ 缺点: 参数量相对较多                                │
#   └─────────────────────────────────────────────────────┘
#
# 'conv_gen_1x1' (V4 新增):
#   ┌─────────────────────────────────────────────────────┐
#   │ 使用 1x1 卷积进行 mask 细化                         │
#   │                                                     │
#   │ mask_raw = σ(β·X_ir + γ·S_map)                     │
#   │ mask = Conv1x1(C→mid) → BN → ReLU                  │
#   │      → Conv1x1(mid→C) → Sigmoid/Tanh               │
#   │                                                     │
#   │ 优点: 参数量更少，纯通道变换                        │
#   │ 缺点: 无空间感受野，依赖输入质量                    │
#   └─────────────────────────────────────────────────────┘
#
# 'conv_gen': 向后兼容，等同于 'conv_gen_3x3'
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (默认推荐):
#   param_constraint='softplus', mask_center='spatial_mean'
#   适用: 大多数场景，解决 V3 的两个核心问题
#
# 配置2 (简单约束):
#   param_constraint='abs', mask_center='spatial_mean'
#   适用: 快速测试
#
# 配置3 (tanh 方案):
#   param_constraint='softplus', mask_center='tanh'
#   适用: 需要更大的负值范围
#
# 配置4 (V3 兼容):
#   param_constraint='none', mask_center='none'
#   适用: 与 V3 完全一致的行为
#

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


class SingleLevelTextGuidedFusionV4(nn.Module):
    """
    单尺度的文本引导 RGB-IR 融合模块 V4
    
    相比 V3 的新增功能：
    1. param_constraint: β和γ参数约束方式
    2. mask_center: mask 零中心化方式
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        beta (float): X_ir 的权重系数初始值，默认 1.0
        gamma (float): S_map 的权重系数初始值，默认 0.5
        alpha (float): 残差融合系数（仅 param_constraint='residual_alpha' 时使用）
        gap_method (str): GAP 计算方式
        smap_method (str): S_map 归一化方式
        smap_order (str): S_map 计算顺序
        mask_method (str): Mask 生成方式
        mask_reduction (int): Mask 生成器的通道缩减比例
        temperature (float): 温度参数
        param_constraint (str): β和γ参数约束方式 ⭐ V4新增
        mask_center (str): mask 零中心化方式 ⭐ V4新增
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
        alpha: float = 0.1,
        gap_method: Literal['logits', 'max', 'entropy'] = 'logits',
        smap_method: Literal['sigmoid', 'sigmoid_temp', 'normalized'] = 'sigmoid',
        smap_order: Literal['sum_first', 'multiply_first'] = 'sum_first',
        mask_method: Literal['conv_gen', 'residual', 'dual_branch', 'se_spatial'] = 'conv_gen',
        mask_reduction: int = 8,
        temperature: float = 1.0,
        param_constraint: Literal['softplus', 'abs', 'residual_alpha', 'none'] = 'softplus',
        mask_center: Literal['spatial_mean', 'tanh', 'smap_center', 'none'] = 'spatial_mean',
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
        self.param_constraint = param_constraint
        self.mask_center = mask_center
        
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
        
        # ===== 门控参数 β, γ =====
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
        # ===== 残差融合系数 α（仅 residual_alpha 模式使用）=====
        if param_constraint == 'residual_alpha':
            self.alpha = nn.Parameter(torch.tensor(alpha))
        
        # ===== IR 通道对齐 =====
        if ir_channels != rgb_channels:
            self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False)
        else:
            self.ir_align = nn.Identity()
        
        # ===== Mask 生成模块 =====
        self._init_mask_modules(rgb_channels, mask_reduction)
        
        # 打印配置信息
        print(f"[SingleLevelTextGuidedFusionV4] rgb_ch={rgb_channels}, ir_ch={ir_channels}")
        print(f"  gap_method={gap_method}, smap_method={smap_method}")
        print(f"  smap_order={smap_order}, mask_method={mask_method}")
        print(f"  param_constraint={param_constraint}  ⭐ V4新增")
        print(f"  mask_center={mask_center}  ⭐ V4新增")
    
    def _init_mask_modules(self, rgb_channels: int, reduction: int):
        """根据 mask_method 和 mask_center 初始化相应的模块"""
        
        # 确定最后一层激活函数
        if self.mask_center == 'tanh':
            final_activation = nn.Tanh()  # 输出 (-1, 1)
        else:
            final_activation = nn.Sigmoid()  # 输出 (0, 1)
        
        if self.mask_method in ['conv_gen', 'conv_gen_3x3']:
            # 使用 3x3 卷积 (原 conv_gen，向后兼容)
            mid_channels = max(rgb_channels // reduction, 8)
            self.mask_refine_conv = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=3, padding=1, bias=False),
                final_activation
            )
            
        elif self.mask_method == 'conv_gen_1x1':
            # 使用 1x1 卷积 (V4 新增)
            mid_channels = max(rgb_channels // reduction, 8)
            self.mask_refine_conv = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
                final_activation
            )
            
        elif self.mask_method == 'residual':
            mid_channels = max(rgb_channels // reduction, 8)
            self.mask_refine = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),  # 1x1 不需要 padding
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),  # 1x1 不需要 padding
                nn.Tanh()  # 残差始终用 Tanh
            )
            
        elif self.mask_method == 'dual_branch':
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
                final_activation
            )
            mid_channels = max(rgb_channels // reduction, 8)
            self.channel_branch = nn.Sequential(
                nn.Conv2d(rgb_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, rgb_channels, kernel_size=1, bias=False),
                final_activation
            )
            
        elif self.mask_method == 'se_spatial':
            self.se_block = SEBlock(rgb_channels, reduction=max(reduction, 4))
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
            )
    
    def _get_constrained_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 param_constraint 获取约束后的 β 和 γ"""
        if self.param_constraint == 'softplus':
            beta_pos = F.softplus(self.beta)
            gamma_pos = F.softplus(self.gamma)
        elif self.param_constraint == 'abs':
            beta_pos = torch.abs(self.beta)
            gamma_pos = torch.abs(self.gamma)
        else:  # 'none' 或 'residual_alpha'
            beta_pos = self.beta
            gamma_pos = self.gamma
        return beta_pos, gamma_pos
    
    def _compute_gap(
        self, 
        attn_logits: torch.Tensor, 
        attn_probs: torch.Tensor,
        H: int, 
        W: int
    ) -> torch.Tensor:
        """计算类别重要性的 GAP 值"""
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
        """计算 S_map（语义一致性图）"""
        d_k_sqrt = self.d_k ** 0.5
        
        if self.smap_method == 'sigmoid_temp':
            temp = torch.abs(self.temperature) + 1e-6
            A_rgb = torch.sigmoid(attn_logits_rgb / temp)
            A_ir = torch.sigmoid(attn_logits_ir / temp)
        else:
            A_rgb = torch.sigmoid(attn_logits_rgb / d_k_sqrt)
            A_ir = torch.sigmoid(attn_logits_ir / d_k_sqrt)
        
        if self.smap_order == 'multiply_first':
            A_rgb_weighted = weights * A_rgb
            A_ir_weighted = weights * A_ir
            hadamard = A_rgb_weighted * A_ir_weighted
            S_map_flat = hadamard.sum(dim=1, keepdim=True)
        elif self.smap_order == 'sum_first':
            A_rgb_weighted = weights * A_rgb
            A_ir_weighted = weights * A_ir
            A_rgb_agg = A_rgb_weighted.sum(dim=1, keepdim=True)
            A_ir_agg = A_ir_weighted.sum(dim=1, keepdim=True)
            S_map_flat = A_rgb_agg * A_ir_agg
        else:
            raise ValueError(f"Unknown smap_order: {self.smap_order}")
        
        S_map = S_map_flat.view(B, 1, H, W)
        
        if self.smap_method == 'normalized':
            S_map_mean = S_map.mean(dim=[2, 3], keepdim=True)
            S_map = torch.sigmoid(S_map - S_map_mean)
        else:
            S_map = S_map / N
        
        # ⭐ V4 新增：S_map 零中心化
        if self.mask_center == 'smap_center':
            S_map = S_map - S_map.mean(dim=[2, 3], keepdim=True)
        
        return S_map
    
    def _generate_mask(
        self,
        x_ir_aligned: torch.Tensor,
        S_map: torch.Tensor,
    ) -> torch.Tensor:
        """生成 Mask"""
        B, C, H, W = x_ir_aligned.shape
        
        # 获取约束后的参数
        beta_pos, gamma_pos = self._get_constrained_params()
        
        if self.mask_method in ['conv_gen', 'conv_gen_3x3', 'conv_gen_1x1']:
            # conv_gen / conv_gen_3x3 / conv_gen_1x1 共用相同逻辑
            mask_raw = torch.sigmoid(beta_pos * x_ir_aligned + gamma_pos * S_map)
            mask = self.mask_refine_conv(mask_raw)
            
        elif self.mask_method == 'residual':
            mask_raw = torch.sigmoid(beta_pos * x_ir_aligned + gamma_pos * S_map)
            delta = self.mask_refine(mask_raw)
            mask = torch.clamp(mask_raw + 0.1 * delta, 0, 1)
            
        elif self.mask_method == 'dual_branch':
            mask_spatial = self.spatial_branch(S_map)
            mask_channel = self.channel_branch(x_ir_aligned)
            mask = mask_spatial * mask_channel
            
        elif self.mask_method == 'se_spatial':
            se_weight = self.se_block(x_ir_aligned)
            x_ir_se = se_weight * x_ir_aligned
            S_refined = self.spatial_conv(S_map)
            mask = torch.sigmoid(beta_pos * x_ir_se + gamma_pos * S_refined)
        else:
            raise ValueError(f"Unknown mask_method: {self.mask_method}")
        
        # ⭐ V4 新增：Mask 零中心化（spatial_mean 模式）
        if self.mask_center == 'spatial_mean':
            mask = mask - mask.mean(dim=[2, 3], keepdim=True)
        
        return mask
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
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
        
        # ===== Step 4: 生成 Mask =====
        x_ir_aligned = self.ir_align(x_ir_resized)
        mask = self._generate_mask(x_ir_aligned, S_map)
        
        # ===== Step 5: 最终融合 =====
        if self.param_constraint == 'residual_alpha':
            # 残差形式: x_fused = x_rgb + α * (x_rgb * mask)
            x_fused = x_rgb + self.alpha * (x_rgb * mask)
        else:
            # 标准形式: x_fused = x_rgb * mask + x_rgb = x_rgb * (1 + mask)
            # 注意：当 mask_center='spatial_mean' 时，mask 有正有负
            x_fused = x_rgb + x_rgb * mask
        
        return x_fused


@MODELS.register_module()
class TextGuidedRGBIRFusionV4(BaseModule):
    """
    Text-guided RGB-IR Fusion Module V4
    
    多尺度文本引导融合模块，为每个金字塔层级独立应用融合。
    
    相比 V3 的新增功能：
    1. param_constraint: β和γ参数约束方式（解决学习到负值的问题）
    2. mask_center: mask 零中心化方式（解决全正值问题）
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表
        ir_channels (List[int]): IR 特征通道数列表
        text_dim (int): 文本特征维度
        num_classes (int): 类别数
        beta (float): X_ir 的权重系数初始值
        gamma (float): S_map 的权重系数初始值
        alpha (float): 残差融合系数
        gap_method (str): GAP 计算方式
        smap_method (str): S_map 归一化方式
        smap_order (str): S_map 计算顺序
        mask_method (str): Mask 生成方式
        mask_reduction (int): Mask 生成器的通道缩减比例
        temperature (float): 温度参数
        param_constraint (str): β和γ参数约束方式 ⭐ V4新增
        mask_center (str): mask 零中心化方式 ⭐ V4新增
    """
    
    def __init__(
        self,
        rgb_channels: List[int] = [128, 256, 512],
        ir_channels: List[int] = [64, 128, 256],
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
        alpha: float = 0.1,
        gap_method: str = 'logits',
        smap_method: str = 'sigmoid',
        smap_order: str = 'sum_first',
        mask_method: str = 'conv_gen',
        mask_reduction: int = 8,
        temperature: float = 1.0,
        param_constraint: str = 'softplus',
        mask_center: str = 'spatial_mean',
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.num_levels = len(rgb_channels)
        
        print(f"\n{'='*60}")
        print(f"[TextGuidedRGBIRFusionV4] 初始化配置:")
        print(f"  - RGB channels: {rgb_channels}")
        print(f"  - IR channels: {ir_channels}")
        print(f"  - gap_method: {gap_method}")
        print(f"  - smap_method: {smap_method}")
        print(f"  - smap_order: {smap_order}")
        print(f"  - mask_method: {mask_method}")
        print(f"  - param_constraint: {param_constraint}  ⭐ V4新增")
        print(f"  - mask_center: {mask_center}  ⭐ V4新增")
        print(f"{'='*60}\n")
        
        self.fusion_modules = nn.ModuleList([
            SingleLevelTextGuidedFusionV4(
                rgb_channels=rgb_channels[i],
                ir_channels=ir_channels[i],
                text_dim=text_dim,
                num_classes=num_classes,
                beta=beta,
                gamma=gamma,
                alpha=alpha,
                gap_method=gap_method,
                smap_method=smap_method,
                smap_order=smap_order,
                mask_method=mask_method,
                mask_reduction=mask_reduction,
                temperature=temperature,
                param_constraint=param_constraint,
                mask_center=mask_center,
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
