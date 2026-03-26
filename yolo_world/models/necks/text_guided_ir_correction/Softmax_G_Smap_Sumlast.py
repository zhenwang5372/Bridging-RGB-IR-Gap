# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module - Softmax-G-Smap-SumLast
# 
# 方案核心思想：G求积归一化加卷积 (G-Product-Norm-Conv)
# 
# 与 V4 的主要区别：
# - V4: 使用差值 |A_rgb - A_ir| 和 disagreement = 1 - G
# - 本方案: 使用求积 A_rgb ⊙ A_ir 和 G 作为权重（共识区域增强）
# 
# - V4: x_ir - alpha * Error_map（减法纠错）
# - 本方案: x_ir * (1 + alpha * M)（乘法增强）
#
# 公式:
#   Step 1: A_rgb, A_ir = Softmax(Q @ K / √d_k)
#   Step 2: G_c = cos(A_rgb^c, A_ir^c)  -- 类别一致性
#   Step 3: S_raw = Σ_c (G_c × (A_rgb^c ⊙ A_ir^c))  -- 加权求积
#   Step 4: S_map = MinMax_Normalize(S_raw)
#   Step 5: M = Conv_Refine(S_map)
#   Step 6: X_ir_out = X_ir × (1 + α × M)  -- 增强

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


@MODELS.register_module()
class SoftmaxGSmapSumlast(BaseModule):
    """
    Text-guided IR Correction Module - Softmax-G-Smap-SumLast
    
    方案核心思想：
    - Softmax: 使用 Softmax Attention 生成概率分布形式的语义图
    - G (Global): 利用文本计算类别一致性 G_c，作为可信度权重
    - Smap (Product): 利用哈达姆积 (A ⊙ B) 挖掘空间共鸣
    - SumLast: 在计算完所有类别的加权共识后，最后进行求和聚合 (N → 1)
    - Norm & Conv: 引入归一化和卷积层，将共识图转化为高质量的特征门控
    
    物理含义：
    - G 高: RGB 和 IR 在该类别上语义对齐好
    - A_rgb ⊙ A_ir 高: 两模态在该空间位置都有高响应
    - 最终增强的是"语义和空间都高度一致的可信区域"
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认512
        num_classes (int): 类别数，默认4
        enhancement_alpha (float): 增强强度初始值，默认 0.1
        d_k (int): Attention 的 key 维度，默认 128
        log_alpha (bool): 是否打印 Alpha 值到日志，默认 True
        log_interval (int): 训练时打印 Alpha 的间隔（iteration 数），默认 50
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        enhancement_alpha: float = 0.1,  # 增强强度初始值
        d_k: int = 128,
        log_alpha: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        self.d_k = d_k
        self.log_alpha = log_alpha
        self.log_interval = log_interval
        
        # 训练时的 iteration 计数器
        self._train_iter_count = 0
        # 用于控制验证时每个 epoch 只打印一次
        self._alpha_logged_this_epoch = False
        
        # 每个尺度独立的 alpha
        self.enhancement_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(enhancement_alpha))
            for _ in range(self.num_levels)
        ])
        
        # 为每个尺度创建处理模块
        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelSoftmaxGSmapSumlast(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    d_k=d_k,
                )
            )
    
    def get_alpha_values(self):
        """获取当前的 Alpha 值"""
        alphas = {
            'P3': self.enhancement_alphas[0].item(),
            'P4': self.enhancement_alphas[1].item(),
            'P5': self.enhancement_alphas[2].item(),
        }
        alphas['mean'] = sum(alphas.values()) / 3
        return alphas
    
    def log_alpha_values(self, stage: str = 'val', iter_num: int = None):
        """
        打印 Alpha 值到日志
        
        Args:
            stage: 'train' 或 'val'
            iter_num: 当前 iteration 数（仅 train 时使用）
        """
        if not self.log_alpha:
            return
        
        # 验证阶段：每个 epoch 只打印一次
        if stage == 'val' and self._alpha_logged_this_epoch:
            return
        
        alphas = self.get_alpha_values()
        logger = MMLogger.get_current_instance()
        
        if stage == 'train' and iter_num is not None:
            logger.info(
                f"[Softmax-G-Smap-SumLast] Iter [{iter_num}] Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}"
            )
        else:
            logger.info(
                f"[Softmax-G-Smap-SumLast] Val Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}"
            )
        
        if stage == 'val':
            self._alpha_logged_this_epoch = True
    
    def reset_alpha_log_flag(self):
        """重置 Alpha 打印标志（在新 epoch 开始时调用）"""
        self._alpha_logged_this_epoch = False
        self._train_iter_count = 0
    
    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Args:
            rgb_feats: RGB特征 (P3, P4, P5)
            ir_feats: IR特征 (P3, P4, P5)
            txt_feats: 文本特征
        
        Returns:
            rgb_feats: 原始RGB特征（不变）
            ir_enhanced_feats: 增强后的IR特征 (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Feature levels mismatch: RGB={len(rgb_feats)}, IR={len(ir_feats)}, expected={self.num_levels}"
        
        # 打印 Alpha 值
        if self.training:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self.log_alpha_values(stage='train', iter_num=self._train_iter_count)
        else:
            self.log_alpha_values(stage='val')
        
        # 统一处理txt_feats
        if isinstance(txt_feats, tuple):
            txt_feats, text_mask = txt_feats
        else:
            text_mask = None
        
        # 处理维度
        B = rgb_feats[0].size(0)
        
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)
        elif txt_feats.dim() == 3:
            B_txt = txt_feats.size(0)
            if B_txt != B:
                if B_txt == 1:
                    txt_feats = txt_feats.expand(B, -1, -1)
                else:
                    txt_feats = txt_feats[:B]
        
        ir_enhanced_feats = []
        
        for i in range(self.num_levels):
            ir_enhanced = self.correction_modules[i](
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats,
                self.enhancement_alphas[i]
            )
            ir_enhanced_feats.append(ir_enhanced)
        
        return rgb_feats, tuple(ir_enhanced_feats)


class SingleLevelSoftmaxGSmapSumlast(nn.Module):
    """
    单尺度的 Softmax-G-Smap-SumLast 模块
    
    处理流程:
    1. 计算 Softmax Attention: A_rgb, A_ir
    2. 计算一致性分数 G: 余弦相似度
    3. 加权求积: S_raw = Σ_c (G_c × (A_rgb^c ⊙ A_ir^c))
    4. 归一化: S_map = MinMax_Normalize(S_raw)
    5. 卷积细化: M = Conv_Refine(S_map)
    6. 特征增强: X_ir_out = X_ir × (1 + α × M)
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        d_k: int = 128,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.d_k = d_k
        
        # Query/Key 投影
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        
        # 卷积细化模块: Conv1x1 -> ReLU -> Conv3x3 -> Sigmoid
        mid_channels = max(ir_channels // 4, 16)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB特征 [B, C_rgb, H, W]
            x_ir: IR特征 [B, C_ir, H, W]
            txt_feats: 文本特征 [B, N, text_dim]
            alpha: 增强强度参数
        
        Returns:
            ir_enhanced: 增强后的IR特征 [B, C_ir, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N = txt_feats.size(1)  # 类别数
        
        # 尺寸对齐
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        # ===== Step 1: Semantic Activation (Softmax Attention) =====
        # Q: [B, N, d_k]
        Q = self.text_query_proj(txt_feats)
        
        # K_rgb: [B, d_k, H*W]
        K_rgb = self.rgb_key_proj(x_rgb)
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)
        
        # K_ir: [B, d_k, H*W]
        K_ir = self.ir_key_proj(x_ir)
        K_ir_flat = K_ir.view(B, self.d_k, H * W)
        
        # Attention: Q @ K^T / √d_k -> Softmax
        d_k_sqrt = self.d_k ** 0.5
        
        # A_rgb: [B, N, H*W]
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat) / d_k_sqrt
        A_rgb = F.softmax(attn_logits_rgb, dim=-1)
        
        # A_ir: [B, N, H*W]
        attn_logits_ir = torch.bmm(Q, K_ir_flat) / d_k_sqrt
        A_ir = F.softmax(attn_logits_ir, dim=-1)
        
        # ===== Step 2: Global Consistency G (余弦相似度) =====
        # Normalize for cosine similarity
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)  # [B, N, H*W]
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)    # [B, N, H*W]
        
        # G: [B, N] - 每个类别的一致性分数
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # [B, N]
        G = torch.clamp(G, 0.0, 1.0)
        
        # ===== Step 3: Weighted Product & SumLast =====
        # Spatial Product: A_rgb ⊙ A_ir, [B, N, H*W]
        spatial_product = A_rgb * A_ir
        
        # Weighted sum: G * Product -> sum(dim=1)
        # G: [B, N] -> [B, N, 1] for broadcasting
        G_expanded = G.unsqueeze(-1)  # [B, N, 1]
        
        # S_raw: [B, H*W]
        S_raw = (G_expanded * spatial_product).sum(dim=1)  # [B, H*W]
        
        # ===== Step 4: Normalization (Min-Max per sample) =====
        S_min = S_raw.min(dim=-1, keepdim=True)[0]  # [B, 1]
        S_max = S_raw.max(dim=-1, keepdim=True)[0]  # [B, 1]
        S_range = torch.clamp(S_max - S_min, min=1e-6)
        S_norm = (S_raw - S_min) / S_range
        
        # Reshape to [B, 1, H, W]
        S_map = S_norm.view(B, 1, H, W)
        
        # ===== Step 5: Conv Refinement =====
        # M_final: [B, 1, H, W], 值在 [0, 1]
        M_final = self.conv_refine(S_map)
        
        # ===== Step 6: Feature Enhancement =====
        # X_ir_out = X_ir × (1 + α × M)
        # 物理含义: 在确信正确的区域（共识区），放大红外特征的表达能力
        ir_enhanced = x_ir * (1 + alpha * M_final)
        
        return ir_enhanced
