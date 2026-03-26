# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB-IR Fusion Module V5
#
# ==================== V5 版本核心改进 ====================
#
# V5 相比 V4 的主要改进：
# 1. 去除 x_ir_aligned 对 S_map 的干扰，直接使用 S_map 引导融合
# 2. 使用归一化 sigmoid 替代 softmax/普通 sigmoid，解决值过小/分布偏移问题
# 3. 去除类别权重 w（V4 中 w 全退化为 1.0），直接求和
# 4. α 使用 softplus 约束，保证 > 0
# 5. 支持三种融合模式，可通过配置切换
#
# ==================== 三种融合模式 ====================
#
# 【模式A】smap_direct - S_map 直接引导（默认推荐）
# ─────────────────────────────────────────────────────────
#   mask = S_map - mean(S_map)
#   x_fused = x_rgb + α * x_rgb * mask
#
#   特点: 最简单直接，S_map 信息无损传递
#
# 【模式B】channel_spatial - 通道-空间分离注意力
# ─────────────────────────────────────────────────────────
#   spatial_mask = S_map
#   channel_attn = sigmoid(pool(ir_align(x_ir)))
#   mask = channel_attn * spatial_mask - mean(...)
#   x_fused = x_rgb + α * x_rgb * mask
#
#   特点: 空间和通道分离，IR 只提供通道注意力
#
# 【模式C】dual_gate - 双向门控
# ─────────────────────────────────────────────────────────
#   enhance_gate = sigmoid(γ_e * S_map)
#   suppress_gate = sigmoid(-γ_s * S_map)
#   x_fused = x_rgb + α * x_rgb * enhance_gate - β * x_rgb * suppress_gate
#
#   特点: 显式的增强/抑制机制
#
# ==================== 参数说明 ====================
#
# fusion_mode: 融合模式 ('smap_direct' | 'channel_spatial' | 'dual_gate')
# alpha_init: α 初始值
# alpha_constraint: α 约束方式 ('softplus' | 'abs' | 'sigmoid' | 'none')
# smap_normalize: S_map 归一化方式 ('sigmoid_centered' | 'sigmoid' | 'none')
# use_class_weight: 是否使用类别权重 w
# use_refine_conv: 是否使用细化卷积
# mask_center: mask 零中心化方式 ('spatial_mean' | 'none')
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal, Dict, Any, Optional
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


class SingleLevelTextGuidedFusionV5(nn.Module):
    """
    单尺度的文本引导 RGB-IR 融合模块 V5
    
    V5 核心改进：
    1. S_map 直接引导，不被 x_ir_aligned 干扰
    2. 归一化 sigmoid 解决数值问题
    3. 去除退化的类别权重 w
    4. α 使用 softplus 约束
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        fusion_mode (str): 融合模式 ('smap_direct' | 'channel_spatial' | 'dual_gate')
        alpha_init (float): α 初始值
        alpha_constraint (str): α 约束方式
        smap_normalize (str): S_map 归一化方式
        use_class_weight (bool): 是否使用类别权重
        use_refine_conv (bool): 是否使用细化卷积
        refine_conv_kernel (int): 细化卷积核大小
        mask_center (str): mask 零中心化方式
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        fusion_mode: Literal['smap_direct', 'channel_spatial', 'dual_gate'] = 'smap_direct',
        alpha_init: float = 0.1,
        alpha_constraint: Literal['softplus', 'abs', 'sigmoid', 'none'] = 'softplus',
        smap_normalize: Literal['sigmoid_centered', 'sigmoid', 'none'] = 'sigmoid_centered',
        use_class_weight: bool = False,
        use_refine_conv: bool = False,
        refine_conv_kernel: int = 1,
        mask_center: Literal['spatial_mean', 'none'] = 'spatial_mean',
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.alpha_constraint = alpha_constraint
        self.smap_normalize = smap_normalize
        self.use_class_weight = use_class_weight
        self.use_refine_conv = use_refine_conv
        self.mask_center = mask_center
        
        # ===== Step 1: Query/Key 投影 =====
        d_k = 128
        self.d_k = d_k
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1, bias=False)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1, bias=False)
        
        # ===== Step 2: 可学习参数 α =====
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # ===== Step 3: 类别权重 MLP（可选）=====
        if use_class_weight:
            self.class_weight_mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # ===== Step 4: 模式相关参数 =====
        if fusion_mode == 'channel_spatial':
            # 通道注意力需要 IR 对齐
            if ir_channels != rgb_channels:
                self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False)
            else:
                self.ir_align = nn.Identity()
            # 通道投影
            self.channel_proj = nn.Sequential(
                nn.Linear(rgb_channels, rgb_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(rgb_channels // 4, rgb_channels),
                nn.Sigmoid()
            )
        elif fusion_mode == 'dual_gate':
            # 双门控参数
            self.gamma_enhance = nn.Parameter(torch.tensor(1.0))
            self.gamma_suppress = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(alpha_init))  # 抑制系数
        
        # ===== Step 5: 可选的细化卷积 =====
        if use_refine_conv:
            padding = refine_conv_kernel // 2
            mid_channels = max(rgb_channels // 8, 8)
            self.mask_refine_conv = nn.Sequential(
                nn.Conv2d(1, mid_channels, kernel_size=refine_conv_kernel, padding=padding, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 1, kernel_size=refine_conv_kernel, padding=padding, bias=False),
                nn.Sigmoid()
            )
        
        # 打印配置信息
        print(f"[SingleLevelTextGuidedFusionV5] rgb_ch={rgb_channels}, ir_ch={ir_channels}")
        print(f"  fusion_mode={fusion_mode}")
        print(f"  alpha_constraint={alpha_constraint}, alpha_init={alpha_init}")
        print(f"  smap_normalize={smap_normalize}")
        print(f"  use_class_weight={use_class_weight}")
        print(f"  use_refine_conv={use_refine_conv}")
        print(f"  mask_center={mask_center}")
    
    def _get_alpha(self) -> torch.Tensor:
        """获取约束后的 α"""
        if self.alpha_constraint == 'softplus':
            return F.softplus(self.alpha)
        elif self.alpha_constraint == 'abs':
            return torch.abs(self.alpha)
        elif self.alpha_constraint == 'sigmoid':
            return torch.sigmoid(self.alpha)
        else:  # 'none'
            return self.alpha
    
    def _compute_smap(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 S_map（语义一致性图）
        
        Returns:
            S_map: [B, 1, H, W]
            intermediates: 中间结果字典（用于可视化）
        """
        B, C_rgb, H, W = x_rgb.shape
        N = txt_feats.size(1)
        
        # Query/Key 投影
        Q = self.text_query_proj(txt_feats)  # [B, N, d_k]
        K_rgb = self.rgb_key_proj(x_rgb)      # [B, d_k, H, W]
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)  # [B, d_k, H*W]
        
        # IR resize if needed
        x_ir_resized = x_ir
        if x_ir.shape[-2:] != (H, W):
            x_ir_resized = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        K_ir = self.ir_key_proj(x_ir_resized)  # [B, d_k, H, W]
        K_ir_flat = K_ir.view(B, self.d_k, H * W)  # [B, d_k, H*W]
        
        # Attention logits
        d_k_sqrt = self.d_k ** 0.5
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat) / d_k_sqrt  # [B, N, H*W]
        attn_logits_ir = torch.bmm(Q, K_ir_flat) / d_k_sqrt    # [B, N, H*W]
        
        # 归一化 sigmoid（方案b）
        # A = sigmoid(logits) / sum(sigmoid(logits))
        A_rgb_raw = torch.sigmoid(attn_logits_rgb)  # [B, N, H*W]
        A_ir_raw = torch.sigmoid(attn_logits_ir)    # [B, N, H*W]
        
        # 归一化（避免值太小）
        A_rgb = A_rgb_raw / (A_rgb_raw.sum(dim=-1, keepdim=True) + 1e-6)
        A_ir = A_ir_raw / (A_ir_raw.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 计算 S_map
        if self.use_class_weight:
            # 计算类别权重
            gap_rgb = attn_logits_rgb.mean(dim=-1)  # [B, N]
            gap_ir = attn_logits_ir.mean(dim=-1)    # [B, N]
            
            weights = []
            for c in range(N):
                class_input = torch.stack([gap_rgb[:, c], gap_ir[:, c]], dim=-1)
                w_c = self.class_weight_mlp(class_input)
                weights.append(w_c)
            weights = torch.stack(weights, dim=1)  # [B, N, 1]
            
            # 加权 Hadamard 积
            A_rgb_weighted = weights * A_rgb  # [B, N, H*W]
            A_ir_weighted = weights * A_ir
            S_map_flat = (A_rgb_weighted * A_ir_weighted).sum(dim=1, keepdim=True)  # [B, 1, H*W]
        else:
            # 不加权，直接 Hadamard 积求和（方案d）
            S_map_flat = (A_rgb * A_ir).sum(dim=1, keepdim=True)  # [B, 1, H*W]
            weights = None
        
        S_map_raw = S_map_flat.view(B, 1, H, W)  # [B, 1, H, W]
        
        # S_map 归一化
        if self.smap_normalize == 'sigmoid_centered':
            # 先减均值再 sigmoid，确保有高有低
            S_map = torch.sigmoid(S_map_raw - S_map_raw.mean(dim=[2, 3], keepdim=True))
        elif self.smap_normalize == 'sigmoid':
            S_map = torch.sigmoid(S_map_raw)
        else:  # 'none'
            S_map = S_map_raw
        
        # 收集中间结果
        intermediates = {
            'A_rgb': A_rgb.view(B, N, H, W),
            'A_ir': A_ir.view(B, N, H, W),
            'S_map_raw': S_map_raw,
            'S_map': S_map,
            'weights': weights,
            'x_ir_resized': x_ir_resized,
        }
        
        return S_map, intermediates
    
    def _fusion_smap_direct(
        self,
        x_rgb: torch.Tensor,
        S_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        模式A: S_map 直接引导
        """
        alpha_pos = self._get_alpha()
        
        # 可选的细化卷积
        if self.use_refine_conv:
            mask = self.mask_refine_conv(S_map)
        else:
            mask = S_map
        
        # mask 零中心化
        if self.mask_center == 'spatial_mean':
            mask_centered = mask - mask.mean(dim=[2, 3], keepdim=True)
        else:
            mask_centered = mask
        
        # 融合
        x_rgb_masked = alpha_pos * x_rgb * mask_centered
        x_fused = x_rgb + x_rgb_masked
        
        intermediates = {
            'mask': mask,
            'mask_centered': mask_centered,
            'x_rgb_masked': x_rgb_masked,
            'alpha': alpha_pos,
        }
        
        return x_fused, intermediates
    
    def _fusion_channel_spatial(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        S_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        模式B: 通道-空间分离注意力
        """
        alpha_pos = self._get_alpha()
        B, C, H, W = x_rgb.shape
        
        # 计算通道注意力（从 IR 特征）
        x_ir_aligned = self.ir_align(x_ir)
        channel_pool = F.adaptive_avg_pool2d(x_ir_aligned, 1).view(B, C)  # [B, C]
        channel_attn = self.channel_proj(channel_pool).view(B, C, 1, 1)   # [B, C, 1, 1]
        
        # 空间注意力（来自 S_map）
        spatial_mask = S_map  # [B, 1, H, W]
        
        # 可选的细化卷积
        if self.use_refine_conv:
            spatial_mask = self.mask_refine_conv(spatial_mask)
        
        # 组合通道和空间注意力
        mask = channel_attn * spatial_mask  # [B, C, H, W]
        
        # mask 零中心化
        if self.mask_center == 'spatial_mean':
            mask_centered = mask - mask.mean(dim=[2, 3], keepdim=True)
        else:
            mask_centered = mask
        
        # 融合
        x_rgb_masked = alpha_pos * x_rgb * mask_centered
        x_fused = x_rgb + x_rgb_masked
        
        intermediates = {
            'x_ir_aligned': x_ir_aligned,
            'channel_attn': channel_attn,
            'spatial_mask': spatial_mask,
            'mask': mask,
            'mask_centered': mask_centered,
            'x_rgb_masked': x_rgb_masked,
            'alpha': alpha_pos,
        }
        
        return x_fused, intermediates
    
    def _fusion_dual_gate(
        self,
        x_rgb: torch.Tensor,
        S_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        模式C: 双向门控
        """
        alpha_pos = self._get_alpha()
        beta_pos = F.softplus(self.beta)
        gamma_e = F.softplus(self.gamma_enhance)
        gamma_s = F.softplus(self.gamma_suppress)
        
        # 增强门和抑制门
        enhance_gate = torch.sigmoid(gamma_e * S_map)      # 目标区域高
        suppress_gate = torch.sigmoid(-gamma_s * S_map)    # 背景区域高
        
        # 可选的细化卷积
        if self.use_refine_conv:
            enhance_gate = self.mask_refine_conv(enhance_gate)
            # suppress_gate 不做细化，保持简单
        
        # 双向调制
        x_enhanced = alpha_pos * x_rgb * enhance_gate
        x_suppressed = beta_pos * x_rgb * suppress_gate
        x_fused = x_rgb + x_enhanced - x_suppressed
        
        intermediates = {
            'enhance_gate': enhance_gate,
            'suppress_gate': suppress_gate,
            'x_enhanced': x_enhanced,
            'x_suppressed': x_suppressed,
            'alpha': alpha_pos,
            'beta': beta_pos,
            'gamma_enhance': gamma_e,
            'gamma_suppress': gamma_s,
        }
        
        return x_fused, intermediates
    
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
            txt_feats: 文本特征 [B, N, D]
        
        Returns:
            x_fused: 融合后的特征 [B, C_rgb, H, W]
        """
        # Step 1: 计算 S_map
        S_map, _ = self._compute_smap(x_rgb, x_ir, txt_feats)
        
        # Step 2: 根据融合模式进行融合
        if self.fusion_mode == 'smap_direct':
            x_fused, _ = self._fusion_smap_direct(x_rgb, S_map)
        elif self.fusion_mode == 'channel_spatial':
            x_fused, _ = self._fusion_channel_spatial(x_rgb, x_ir, S_map)
        elif self.fusion_mode == 'dual_gate':
            x_fused, _ = self._fusion_dual_gate(x_rgb, S_map)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        return x_fused
    
    def forward_with_intermediates(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        带中间特征输出的前向传播（用于可视化）
        
        Returns:
            x_fused: 融合后的特征
            intermediates: 所有中间特征的字典
        """
        B, C_rgb, H, W = x_rgb.shape
        
        # Step 1: 计算 S_map
        S_map, smap_intermediates = self._compute_smap(x_rgb, x_ir, txt_feats)
        
        # Step 2: 根据融合模式进行融合
        if self.fusion_mode == 'smap_direct':
            x_fused, fusion_intermediates = self._fusion_smap_direct(x_rgb, S_map)
        elif self.fusion_mode == 'channel_spatial':
            x_fused, fusion_intermediates = self._fusion_channel_spatial(x_rgb, x_ir, S_map)
        elif self.fusion_mode == 'dual_gate':
            x_fused, fusion_intermediates = self._fusion_dual_gate(x_rgb, S_map)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        # 合并所有中间结果
        intermediates = {
            'x_rgb': x_rgb,
            'x_ir': x_ir,
            'x_fused': x_fused,
            'H': H, 'W': W,
            'fusion_mode': self.fusion_mode,
            'alpha_constraint': self.alpha_constraint,
            'smap_normalize': self.smap_normalize,
            'mask_center': self.mask_center,
            **smap_intermediates,
            **fusion_intermediates,
        }
        
        return x_fused, intermediates


@MODELS.register_module()
class TextGuidedRGBIRFusionV5(BaseModule):
    """
    Text-guided RGB-IR Fusion Module V5
    
    多尺度文本引导融合模块，为每个金字塔层级独立应用融合。
    
    V5 核心改进：
    1. S_map 直接引导，不被 x_ir_aligned 干扰
    2. 归一化 sigmoid 解决数值问题
    3. 去除退化的类别权重 w
    4. α 使用 softplus 约束
    5. 支持三种融合模式，可通过配置切换
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表
        ir_channels (List[int]): IR 特征通道数列表
        text_dim (int): 文本特征维度
        num_classes (int): 类别数
        fusion_mode (str): 融合模式
        alpha_init (float): α 初始值
        alpha_constraint (str): α 约束方式
        smap_normalize (str): S_map 归一化方式
        use_class_weight (bool): 是否使用类别权重
        use_refine_conv (bool): 是否使用细化卷积
        refine_conv_kernel (int): 细化卷积核大小
        mask_center (str): mask 零中心化方式
    """
    
    def __init__(
        self,
        rgb_channels: List[int] = [128, 256, 512],
        ir_channels: List[int] = [64, 128, 256],
        text_dim: int = 512,
        num_classes: int = 4,
        fusion_mode: str = 'smap_direct',
        alpha_init: float = 0.1,
        alpha_constraint: str = 'softplus',
        smap_normalize: str = 'sigmoid_centered',
        use_class_weight: bool = False,
        use_refine_conv: bool = False,
        refine_conv_kernel: int = 1,
        mask_center: str = 'spatial_mean',
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.num_levels = len(rgb_channels)
        self._s_maps = None  # 存储 S_map 用于损失计算
        
        print(f"\n{'='*60}")
        print(f"[TextGuidedRGBIRFusionV5] 初始化配置:")
        print(f"  - RGB channels: {rgb_channels}")
        print(f"  - IR channels: {ir_channels}")
        print(f"  - fusion_mode: {fusion_mode}")
        print(f"  - alpha_init: {alpha_init}, constraint: {alpha_constraint}")
        print(f"  - smap_normalize: {smap_normalize}")
        print(f"  - use_class_weight: {use_class_weight}")
        print(f"  - use_refine_conv: {use_refine_conv}")
        print(f"  - mask_center: {mask_center}")
        print(f"{'='*60}\n")
        
        self.fusion_modules = nn.ModuleList([
            SingleLevelTextGuidedFusionV5(
                rgb_channels=rgb_channels[i],
                ir_channels=ir_channels[i],
                text_dim=text_dim,
                num_classes=num_classes,
                fusion_mode=fusion_mode,
                alpha_init=alpha_init,
                alpha_constraint=alpha_constraint,
                smap_normalize=smap_normalize,
                use_class_weight=use_class_weight,
                use_refine_conv=use_refine_conv,
                refine_conv_kernel=refine_conv_kernel,
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
        s_maps = []
        
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](
                x_rgb=rgb_feats[i],
                x_ir=ir_feats[i],
                txt_feats=txt_feats,
            )
            fused_feats.append(fused)
            
            # 存储 S_map 用于损失计算
            S_map, _ = self.fusion_modules[i]._compute_smap(
                rgb_feats[i], ir_feats[i], txt_feats)
            s_maps.append(S_map)
        
        self._s_maps = s_maps
        return tuple(fused_feats)
    
    def get_s_maps(self) -> Optional[List[torch.Tensor]]:
        """获取各尺度的 S_map（用于辅助损失）"""
        return self._s_maps
    
    def forward_with_intermediates(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], List[Dict[str, Any]]]:
        """
        带中间特征输出的前向传播（用于可视化）
        
        Returns:
            fused_feats: 融合后的特征元组
            all_intermediates: 各尺度的中间特征列表
        """
        fused_feats = []
        all_intermediates = []
        
        for i in range(self.num_levels):
            fused, intermediates = self.fusion_modules[i].forward_with_intermediates(
                x_rgb=rgb_feats[i],
                x_ir=ir_feats[i],
                txt_feats=txt_feats,
            )
            fused_feats.append(fused)
            all_intermediates.append(intermediates)
        
        return tuple(fused_feats), all_intermediates
