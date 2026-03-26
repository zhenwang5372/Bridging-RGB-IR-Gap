# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB-IR Fusion Module (Scheme 2)
# 
# 实现方案二：基于文本加权的逐类哈达姆门控 (Text-Weighted Class-wise Hadamard Gating)
# 
# 核心思想：
# 1. 利用文本特征生成 Query，从 RGB/IR 特征中提取 N 个类别的注意力图
# 2. 在 HW 维度上做 GAP，计算每个类别的重要性权重 w_c
# 3. 用 w_c 对 A_rgb 和 A_ir 进行加权，计算哈达姆积得到 S_map
# 4. 利用 S_map 和 X_ir 生成门控 Mask，对 RGB 特征进行调制

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


class SingleLevelTextGuidedFusion(nn.Module):
    """
    单尺度的文本引导 RGB-IR 融合模块（方案二）
    
    流程：
        Step 1: 文本引导的语义激活 - 生成 A_rgb 和 A_ir
        Step 2: 类别重要性计算 - 通过 GAP + MLP 得到 w_c
        Step 3: 加权哈达姆对齐 - 计算 S_map
        Step C: 门控生成 - Mask = σ(β·X_ir + γ·S_map)
        Step D: 特征融合 - X_fused = X_rgb · Mask + X_rgb
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        beta (float): X_ir 的权重系数，默认 1.0
        gamma (float): S_map 的权重系数，默认 0.5
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        
        # ===== Step 1: Query/Key 投影 =====
        d_k = 128
        self.d_k = d_k
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1, bias=False)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1, bias=False)
        
        # ===== Step 2: 类别权重计算 MLP =====
        # 输入: concat(GAP_rgb, GAP_ir) = 2 维
        # 输出: w_c ∈ [0, 1]
        # 
        # 物理含义：
        # - 如果某个类别在 RGB 和 IR 上都有强激活 → w_c 较大（该类别重要）
        # - 如果某个类别激活都很弱 → w_c 较小（该类别不重要/不存在）
        # - MLP 学习这种模式，为每个类别分配权重
        self.class_weight_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # ===== Step C: 门控生成的权重参数（可学习）=====
        self.beta = nn.Parameter(torch.tensor(beta))   # X_ir 的权重
        self.gamma = nn.Parameter(torch.tensor(gamma)) # S_map 的权重
        
        # ===== IR 通道对齐 =====
        if ir_channels != rgb_channels:
            self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False)
            print(f"[SingleLevelTextGuidedFusion] IR Channel Align: {ir_channels} -> {rgb_channels}")
        else:
            self.ir_align = nn.Identity()
            print(f"[SingleLevelTextGuidedFusion] IR Channel Already Aligned: {ir_channels} == {rgb_channels}")
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB 特征 [B, C_rgb, H, W]
            x_ir: IR 特征 [B, C_ir, H, W]
            txt_feats: 文本特征 [B, N, d]
        
        Returns:
            x_fused: 融合后的特征 [B, C_rgb, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N = txt_feats.size(1)  # num_classes
        
        # ===== Step 1: 语义激活（与方案一相同）=====
        # ⭐ 重要：先对原始 IR 特征提取 Key，再进行通道对齐
        
        # Query: [B, N, d_k]
        Q = self.text_query_proj(txt_feats)
        
        # Key RGB: [B, d_k, H, W] -> [B, d_k, HW]
        K_rgb = self.rgb_key_proj(x_rgb)
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)
        
        # Key IR: 先对原始 x_ir（未对齐）提取 Key
        # 如果空间尺寸不匹配，先插值
        x_ir_resized = x_ir
        if x_ir.shape[-2:] != (H, W):
            x_ir_resized = F.interpolate(
                x_ir, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        K_ir = self.ir_key_proj(x_ir_resized)  # ⭐ 使用原始通道数的 IR
        K_ir_flat = K_ir.view(B, self.d_k, H * W)
        
        # 计算注意力: Q @ K^T / sqrt(d_k)
        d_k_sqrt = self.d_k ** 0.5
        
        # A_rgb: [B, N, HW]
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat) / d_k_sqrt
        A_rgb = F.softmax(attn_logits_rgb, dim=-1)
        
        # A_ir: [B, N, HW]
        attn_logits_ir = torch.bmm(Q, K_ir_flat) / d_k_sqrt
        A_ir = F.softmax(attn_logits_ir, dim=-1)
        
        # ===== Step 2: 类别重要性权重计算 =====
        # 对每个类别在 HW 维度上做全局平均池化（GAP）
        # gap_rgb[b, c] = 类别 c 在 RGB 特征图上的平均激活强度
        gap_rgb = A_rgb.mean(dim=-1)  # [B, N, HW] -> [B, N]
        gap_ir = A_ir.mean(dim=-1)    # [B, N, HW] -> [B, N]
        
        # 对每个类别计算权重 w_c
        # 方式: 对每个类别 c，将 [gap_rgb[c], gap_ir[c]] 输入 MLP 得到 w_c
        weights = []
        for c in range(N):
            # class_input: [B, 2]
            # class_input[:, 0] = 类别 c 在 RGB 上的平均激活
            # class_input[:, 1] = 类别 c 在 IR 上的平均激活
            class_input = torch.stack([gap_rgb[:, c], gap_ir[:, c]], dim=-1)
            
            # w_c: [B, 1]，值域 [0, 1]
            w_c = self.class_weight_mlp(class_input)
            weights.append(w_c)
        
        # weights: [B, N, 1]
        weights = torch.stack(weights, dim=1)
        
        # ===== Step 3: 加权哈达姆对齐 =====
        # Ã_rgb = w_c * A_rgb
        A_rgb_weighted = weights * A_rgb  # [B, N, 1] * [B, N, HW] -> [B, N, HW]
        A_ir_weighted = weights * A_ir    # [B, N, 1] * [B, N, HW] -> [B, N, HW]
        
        # 哈达姆积（逐元素相乘）并沿类别维度聚合
        # S_map = Σ(Ã_rgb ⊙ Ã_ir)
        hadamard = A_rgb_weighted * A_ir_weighted  # [B, N, HW]
        S_map_flat = hadamard.sum(dim=1, keepdim=True)  # [B, 1, HW]
        S_map = S_map_flat.view(B, 1, H, W)  # [B, 1, H, W]
        
        # ===== 对齐 IR 通道（用于门控生成）=====
        # 在 Step C 之前进行通道对齐
        x_ir_aligned = self.ir_align(x_ir_resized)
        
        # ===== Step C: 门控生成 =====
        # Mask = σ(β · X_ir + γ · S_map)
        
        # 方式1: 简单广播（S_map 会自动广播到 C 维度）
        # mask = torch.sigmoid(self.beta * x_ir_aligned + self.gamma * S_map)
        
        # 方式2: 更合理的做法 - 先归一化 S_map，然后广播（当前使用）
        # 归一化到 [0, 1] 范围，确保与 X_ir（经过BN）在相近尺度
        S_map_norm = (S_map - S_map.min()) / (S_map.max() - S_map.min() + 1e-6)
        
        # β 和 γ 控制两个信号的相对强度
        # β = 1.0: X_ir 提供基础的红外结构信息（主要）
        # γ = 0.5: S_map 提供一致性校验（辅助）
        mask = torch.sigmoid(self.beta * x_ir_aligned + self.gamma * S_map_norm)
        
        # ===== Step D: 最终融合 =====
        # X_fused = X_rgb · Mask + X_rgb = X_rgb · (1 + Mask)
        x_fused = x_rgb * mask + x_rgb
        
        return x_fused


@MODELS.register_module()
class TextGuidedRGBIRFusion(BaseModule):
    """
    Text-guided RGB-IR Fusion Module (方案二)
    
    多尺度文本引导融合模块，为每个金字塔层级（P3, P4, P5）独立应用融合。
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表 [P3, P4, P5]
        ir_channels (List[int]): IR 特征通道数列表 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        beta (float): X_ir 的权重系数初始值，默认 1.0
        gamma (float): S_map 的权重系数初始值，默认 0.5
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        beta: float = 1.0,
        gamma: float = 0.5,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        
        assert len(rgb_channels) == len(ir_channels), \
            f"rgb_channels and ir_channels must have same length, " \
            f"got {len(rgb_channels)} vs {len(ir_channels)}"
        
        # 为每个尺度创建独立的融合模块
        self.fusion_modules = nn.ModuleList()
        for i, (rgb_ch, ir_ch) in enumerate(zip(rgb_channels, ir_channels)):
            level_name = f"P{i+3}"  # P3, P4, P5
            print(f"[TextGuidedRGBIRFusion] Building {level_name}: "
                  f"RGB={rgb_ch}, IR={ir_ch}, Text={text_dim}, Classes={num_classes}")
            
            self.fusion_modules.append(
                SingleLevelTextGuidedFusion(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    beta=beta,
                    gamma=gamma,
                )
            )
        
        logger = MMLogger.get_current_instance()
        logger.info(
            f"[TextGuidedRGBIRFusion] Initialized with {self.num_levels} levels. "
            f"Beta={beta}, Gamma={gamma}"
        )
    
    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            rgb_feats: RGB 特征 (P3, P4, P5)
            ir_feats: IR 特征 (P3, P4, P5)
            txt_feats: 文本特征 [B, N, d] 或 (txt_feats, text_mask)
        
        Returns:
            fused_feats: 融合后的特征 (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Feature levels mismatch: RGB={len(rgb_feats)}, IR={len(ir_feats)}, " \
            f"expected={self.num_levels}"
        
        # 统一处理 txt_feats（可能是 tuple）
        if isinstance(txt_feats, tuple):
            txt_feats, text_mask = txt_feats
        else:
            text_mask = None
        
        # 处理维度：确保 txt_feats 是 [B, N, d]
        B = rgb_feats[0].size(0)
        
        if txt_feats.dim() == 2:
            # [N, d] -> [B, N, d]
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)
        elif txt_feats.dim() == 3:
            B_txt = txt_feats.size(0)
            if B_txt != B:
                if B_txt == 1:
                    txt_feats = txt_feats.expand(B, -1, -1)
                else:
                    txt_feats = txt_feats[:B]
        
        # 对每个尺度进行融合
        fused_feats = []
        for i in range(self.num_levels):
            fused = self.fusion_modules[i](
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats
            )
            fused_feats.append(fused)
        
        return tuple(fused_feats)
