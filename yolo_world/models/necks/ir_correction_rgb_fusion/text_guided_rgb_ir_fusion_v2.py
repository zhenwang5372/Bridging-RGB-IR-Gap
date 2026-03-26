# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided RGB-IR Fusion Module V2 (Scheme 2 改进版)
# 
# 相比 V1 版本的改进：
# 1. 修复了 w_c 全部相同的问题（softmax 后 GAP 恒为 1/HW）
# 2. 修复了 S_map 值过小的问题（softmax 后 Hadamard 积约为 0）
# 3. 提供多种可配置的计算方案
#
# ==================== 核心改进 ====================
#
# 【问题1】w_c 计算问题：
#   原始方案：gap = mean(softmax(logits))
#   问题：softmax 保证 sum=1，所以 mean = 1/HW ≈ 0.0002，所有类别相同
#   
#   解决方案（3种可选）：
#   - gap_method='logits': 使用 logits 的平均值（推荐）
#   - gap_method='max': 使用 softmax 后的最大值
#   - gap_method='entropy': 使用熵来度量确定性
#
# 【问题2】S_map 计算问题：
#   原始方案：S_map = Σ(softmax(A_rgb) × softmax(A_ir))
#   问题：0.0002 × 0.0002 ≈ 0，S_map 几乎为 0
#   
#   解决方案（3种可选）：
#   - smap_method='sigmoid': 使用 sigmoid 归一化（推荐）
#   - smap_method='sigmoid_temp': 带温度参数的 sigmoid
#   - smap_method='normalized': 完整归一化流程
#
# ==================== 使用方法 ====================
#
# 在配置文件中设置：
# text_guided_fusion=dict(
#     type='TextGuidedRGBIRFusionV2',
#     gap_method='logits',      # 'logits' | 'max' | 'entropy'
#     smap_method='sigmoid',    # 'sigmoid' | 'sigmoid_temp' | 'normalized'
#     temperature=1.0,          # 仅当 smap_method='sigmoid_temp' 时生效
#     ...
# )

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


class SingleLevelTextGuidedFusionV2(nn.Module):
    """
    单尺度的文本引导 RGB-IR 融合模块 V2（方案二改进版）
    
    相比 V1 的改进：
    1. 提供多种 GAP 计算方式，解决 w_c 全部相同的问题
    2. 提供多种 S_map 计算方式，解决值过小的问题
    
    Args:
        rgb_channels (int): RGB 特征通道数
        ir_channels (int): IR 特征通道数
        text_dim (int): 文本特征维度，默认 512
        num_classes (int): 类别数，默认 4
        beta (float): X_ir 的权重系数，默认 1.0
        gamma (float): S_map 的权重系数，默认 0.5
        gap_method (str): GAP 计算方式
            - 'logits': 使用 attention logits 的平均值（推荐）
            - 'max': 使用 softmax 后的最大值
            - 'entropy': 使用熵来度量确定性
        smap_method (str): S_map 计算方式
            - 'sigmoid': 使用 sigmoid 归一化 logits（推荐）
            - 'sigmoid_temp': 带温度的 sigmoid
            - 'normalized': 完整归一化流程
        temperature (float): 温度参数，仅当 smap_method='sigmoid_temp' 时生效
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
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.gap_method = gap_method
        self.smap_method = smap_method
        
        # ===== Step 1: Query/Key 投影 =====
        d_k = 128
        self.d_k = d_k
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1, bias=False)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1, bias=False)
        
        # ===== Step 2: 类别权重计算 MLP =====
        # 输入维度根据 gap_method 不同而不同
        # - logits/max: 输入 2 维 [gap_rgb, gap_ir]
        # - entropy: 输入 2 维 [certainty_rgb, certainty_ir]
        self.class_weight_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # ===== 温度参数（用于 smap_method='sigmoid_temp'）=====
        if smap_method == 'sigmoid_temp':
            # 可学习的温度参数
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            # 固定温度
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # ===== Step C: 门控生成的权重参数（可学习）=====
        self.beta = nn.Parameter(torch.tensor(beta))   # X_ir 的权重
        self.gamma = nn.Parameter(torch.tensor(gamma)) # S_map 的权重
        
        # ===== IR 通道对齐 =====
        if ir_channels != rgb_channels:
            self.ir_align = nn.Conv2d(ir_channels, rgb_channels, kernel_size=1, bias=False)
            print(f"[SingleLevelTextGuidedFusionV2] IR Channel Align: {ir_channels} -> {rgb_channels}")
        else:
            self.ir_align = nn.Identity()
            print(f"[SingleLevelTextGuidedFusionV2] IR Channel Already Aligned: {ir_channels} == {rgb_channels}")
        
        # 打印配置信息
        print(f"[SingleLevelTextGuidedFusionV2] gap_method={gap_method}, smap_method={smap_method}, temperature={temperature}")
    
    def _compute_gap(
        self, 
        attn_logits: torch.Tensor, 
        attn_probs: torch.Tensor,
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        计算类别重要性的 GAP 值
        
        ==================== 方案详解 ====================
        
        【方案A: logits】gap_method='logits'
        ------------------------------------------------
        原理：使用 softmax 之前的 logits 值的平均值
        
        公式：gap[b, c] = mean(logits[b, c, :])
        
        优点：
        - logits 保留了原始的激活强度差异
        - 不同类别会有不同的平均 logits 值
        - 避免了 softmax 导致的 1/HW 问题
        
        值域：(-∞, +∞)，通常在 [-10, 10] 范围内
        
        【方案B: max】gap_method='max'
        ------------------------------------------------
        原理：使用 softmax 后的最大激活值
        
        公式：gap[b, c] = max(softmax(logits[b, c, :]))
        
        优点：
        - 捕捉最强激活位置的强度
        - 如果某类别存在，会有一个位置强激活
        - 如果某类别不存在，所有位置激活都很弱
        
        值域：[1/HW, 1]，约 [0.0002, 1]
        
        【方案C: entropy】gap_method='entropy'
        ------------------------------------------------
        原理：使用熵来度量 attention 分布的确定性
        
        公式：
        entropy = -Σ p(x) × log(p(x))
        certainty = 1 - entropy / max_entropy
        
        物理含义：
        - 熵越低 → attention 越集中（聚焦在少数位置）→ 类别可能存在
        - 熵越高 → attention 越均匀（分散在所有位置）→ 类别可能不存在
        
        值域：[0, 1]
        - certainty=1: 完全集中在一个位置（类别明确存在）
        - certainty=0: 完全均匀分布（类别不存在）
        
        Args:
            attn_logits: [B, N, HW] attention logits（softmax 之前）
            attn_probs: [B, N, HW] attention probabilities（softmax 之后）
            H, W: 空间尺寸
        
        Returns:
            gap: [B, N] 每个类别的 GAP 值
        """
        if self.gap_method == 'logits':
            # ==================== 方案A: 使用 logits 的平均值 ====================
            # 
            # 计算：gap = mean(logits)
            # 
            # 示例：
            #   类别0 (person): logits 在人的位置较高 → gap ≈ 2.5
            #   类别1 (bicycle): logits 在自行车位置较高 → gap ≈ 1.8
            #   类别2 (car): 图中无车 → gap ≈ -1.0
            #   类别3 (dog): 图中无狗 → gap ≈ -0.5
            # 
            # 不同类别会有不同的 gap 值！
            gap = attn_logits.mean(dim=-1)  # [B, N, HW] -> [B, N]
            
        elif self.gap_method == 'max':
            # ==================== 方案B: 使用 softmax 后的最大值 ====================
            # 
            # 计算：gap = max(softmax(logits))
            # 
            # 示例：
            #   类别0 (person): attention 集中在人的位置 → max ≈ 0.15
            #   类别1 (bicycle): attention 集中在自行车位置 → max ≈ 0.08
            #   类别2 (car): 图中无车，attention 分散 → max ≈ 0.002
            #   类别3 (dog): 图中无狗，attention 分散 → max ≈ 0.001
            # 
            # 存在的类别 max 值较高，不存在的类别 max 值接近 1/HW
            gap = attn_probs.max(dim=-1)[0]  # [B, N, HW] -> [B, N]
            
        elif self.gap_method == 'entropy':
            # ==================== 方案C: 使用熵计算确定性 ====================
            # 
            # 熵公式：H(p) = -Σ p(x) × log(p(x))
            # 确定性：certainty = 1 - H(p) / H_max
            # 
            # 其中 H_max = log(HW) 是均匀分布的熵（最大熵）
            # 
            # 示例：
            #   类别0 (person): attention 集中 → entropy 低 → certainty ≈ 0.7
            #   类别1 (bicycle): attention 较集中 → entropy 中 → certainty ≈ 0.5
            #   类别2 (car): attention 分散 → entropy 高 → certainty ≈ 0.1
            #   类别3 (dog): attention 分散 → entropy 高 → certainty ≈ 0.05
            # 
            # certainty 越高，表示该类别越可能存在
            
            # 计算熵（加 1e-10 避免 log(0)）
            entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1)  # [B, N]
            
            # 最大熵（均匀分布）
            max_entropy = torch.log(torch.tensor(H * W, dtype=torch.float32, device=attn_probs.device))
            
            # 转换为确定性分数（熵越低，确定性越高）
            gap = 1 - entropy / max_entropy  # [B, N]，范围 [0, 1]
        
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
        
        ==================== 方案详解 ====================
        
        【方案1-A: sigmoid】smap_method='sigmoid'
        ------------------------------------------------
        原理：使用 sigmoid 将 logits 归一化到 [0, 1]
        
        与 softmax 的区别：
        - softmax: 所有位置竞争，和为1 → 每个位置约 1/HW
        - sigmoid: 每个位置独立，范围 [0, 1] → 可以多个位置同时高
        
        公式：
        A_rgb = sigmoid(logits_rgb / sqrt(d_k))
        A_ir = sigmoid(logits_ir / sqrt(d_k))
        S_map = Σ(w_c × A_rgb × w_c × A_ir) / N
        
        值域：[0, 1]
        
        优点：
        - 简单直接
        - 值域可控
        - 允许多个位置同时有强激活
        
        【方案1-B: sigmoid_temp】smap_method='sigmoid_temp'
        ------------------------------------------------
        原理：带可学习温度参数的 sigmoid
        
        公式：
        A_rgb = sigmoid(logits_rgb / τ)
        A_ir = sigmoid(logits_ir / τ)
        
        温度 τ 的作用：
        - τ < 1: 分布更尖锐（接近二值化）
        - τ = 1: 标准 sigmoid
        - τ > 1: 分布更平滑
        
        优点：
        - 温度可学习，模型自适应调节
        - 更灵活
        
        【方案1-C: normalized】smap_method='normalized'
        ------------------------------------------------
        原理：完整的归一化流程，确保数值稳定
        
        流程：
        1. sigmoid(logits) → 范围 [0, 1]
        2. 加权 Hadamard 积 → 范围 [0, 1]
        3. 类别求和 → 范围 [0, N]
        4. 中心化 + sigmoid → 范围 [0, 1]
        
        最后一步的中心化确保：
        - 高于平均的位置 → 接近 1
        - 低于平均的位置 → 接近 0
        
        优点：
        - 数值最稳定
        - 语义最清晰
        
        Args:
            attn_logits_rgb: [B, N, HW] RGB attention logits
            attn_logits_ir: [B, N, HW] IR attention logits
            weights: [B, N, 1] 类别权重
            B, N, H, W: batch size, num_classes, height, width
        
        Returns:
            S_map: [B, 1, H, W] 语义一致性图，范围 [0, 1]
        """
        d_k_sqrt = self.d_k ** 0.5
        
        if self.smap_method == 'sigmoid':
            # ==================== 方案1-A: Sigmoid 归一化 ====================
            # 
            # 步骤1: 使用 sigmoid 将 logits 转换为 [0, 1]
            #        注意：sigmoid 是独立的，不像 softmax 要求和为1
            # 
            # 步骤2: 加权 Hadamard 积
            #        每个位置的值 = w_c² × A_rgb × A_ir
            #        由于 A 在 [0,1]，乘积也在 [0,1]
            # 
            # 步骤3: 归一化到 [0, 1]
            #        除以类别数 N
            
            # sigmoid 归一化（除以 sqrt(d_k) 使输入在合理范围）
            A_rgb = torch.sigmoid(attn_logits_rgb / d_k_sqrt)  # [B, N, HW]
            A_ir = torch.sigmoid(attn_logits_ir / d_k_sqrt)    # [B, N, HW]
            
            # 加权 Hadamard 积
            A_rgb_weighted = weights * A_rgb  # [B, N, HW]
            A_ir_weighted = weights * A_ir    # [B, N, HW]
            hadamard = A_rgb_weighted * A_ir_weighted  # 范围 [0, 1]
            
            # 聚合并归一化
            S_map_flat = hadamard.sum(dim=1, keepdim=True)  # [B, 1, HW]
            S_map = S_map_flat.view(B, 1, H, W)
            S_map = S_map / N  # 归一化到 [0, 1]
            
        elif self.smap_method == 'sigmoid_temp':
            # ==================== 方案1-B: 带温度的 Sigmoid ====================
            # 
            # 与方案1-A的区别：
            # - 使用可学习的温度参数 τ
            # - sigmoid(x/τ) 控制分布的尖锐程度
            # 
            # 温度 τ 的效果：
            # - τ → 0: sigmoid 接近阶跃函数（0或1）
            # - τ = 1: 标准 sigmoid
            # - τ → ∞: sigmoid 接近常数 0.5
            
            # 确保温度为正
            temp = torch.abs(self.temperature) + 1e-6
            
            # 带温度的 sigmoid
            A_rgb = torch.sigmoid(attn_logits_rgb / temp)  # [B, N, HW]
            A_ir = torch.sigmoid(attn_logits_ir / temp)    # [B, N, HW]
            
            # 加权 Hadamard 积
            A_rgb_weighted = weights * A_rgb
            A_ir_weighted = weights * A_ir
            hadamard = A_rgb_weighted * A_ir_weighted
            
            # 聚合并归一化
            S_map_flat = hadamard.sum(dim=1, keepdim=True)
            S_map = S_map_flat.view(B, 1, H, W)
            S_map = S_map / N
            
        elif self.smap_method == 'normalized':
            # ==================== 方案1-C: 完整归一化流程 ====================
            # 
            # 步骤1: sigmoid 归一化
            # 步骤2: 加权 Hadamard 积
            # 步骤3: 类别求和
            # 步骤4: 中心化 + sigmoid（确保最终在 [0,1] 且有对比度）
            # 
            # 最后一步的意义：
            # - 减去均值使分布中心化
            # - sigmoid 将结果映射到 [0, 1]
            # - 高于平均的区域 → 接近 1（一致性高）
            # - 低于平均的区域 → 接近 0（一致性低）
            
            # sigmoid 归一化
            A_rgb = torch.sigmoid(attn_logits_rgb / d_k_sqrt)
            A_ir = torch.sigmoid(attn_logits_ir / d_k_sqrt)
            
            # 加权 Hadamard 积
            A_rgb_weighted = weights * A_rgb
            A_ir_weighted = weights * A_ir
            hadamard = A_rgb_weighted * A_ir_weighted
            
            # 聚合
            S_map_flat = hadamard.sum(dim=1, keepdim=True)  # [B, 1, HW]
            S_map = S_map_flat.view(B, 1, H, W)
            
            # 中心化 + sigmoid（增强对比度）
            S_map_mean = S_map.mean(dim=[2, 3], keepdim=True)
            S_map = torch.sigmoid(S_map - S_map_mean)
            
        else:
            raise ValueError(f"Unknown smap_method: {self.smap_method}")
        
        return S_map
    
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
        N = txt_feats.size(1)  # num_classes
        
        # ===== Step 1: 计算 Attention Logits =====
        # Query: [B, N, d_k]
        Q = self.text_query_proj(txt_feats)
        
        # Key RGB: [B, d_k, HW]
        K_rgb = self.rgb_key_proj(x_rgb)
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)
        
        # Key IR（先调整空间尺寸）
        x_ir_resized = x_ir
        if x_ir.shape[-2:] != (H, W):
            x_ir_resized = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        K_ir = self.ir_key_proj(x_ir_resized)
        K_ir_flat = K_ir.view(B, self.d_k, H * W)
        
        # 计算 attention logits: Q @ K^T / sqrt(d_k)
        d_k_sqrt = self.d_k ** 0.5
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat) / d_k_sqrt  # [B, N, HW]
        attn_logits_ir = torch.bmm(Q, K_ir_flat) / d_k_sqrt    # [B, N, HW]
        
        # 计算 softmax attention（仅用于 gap_method='max' 或 'entropy'）
        attn_probs_rgb = F.softmax(attn_logits_rgb, dim=-1)  # [B, N, HW]
        attn_probs_ir = F.softmax(attn_logits_ir, dim=-1)    # [B, N, HW]
        
        # ===== Step 2: 计算类别权重 w_c =====
        gap_rgb = self._compute_gap(attn_logits_rgb, attn_probs_rgb, H, W)  # [B, N]
        gap_ir = self._compute_gap(attn_logits_ir, attn_probs_ir, H, W)      # [B, N]
        
        # 对每个类别计算权重
        weights = []
        for c in range(N):
            class_input = torch.stack([gap_rgb[:, c], gap_ir[:, c]], dim=-1)  # [B, 2]
            w_c = self.class_weight_mlp(class_input)  # [B, 1]
            weights.append(w_c)
        
        weights = torch.stack(weights, dim=1)  # [B, N, 1]
        
        # ===== Step 3: 计算 S_map =====
        S_map = self._compute_smap(attn_logits_rgb, attn_logits_ir, weights, B, N, H, W)
        
        # ===== Step C: 门控生成 =====
        # 对齐 IR 通道
        x_ir_aligned = self.ir_align(x_ir_resized)
        
        # Mask = σ(β · X_ir + γ · S_map)
        mask = torch.sigmoid(self.beta * x_ir_aligned + self.gamma * S_map)
        
        # ===== Step D: 最终融合 =====
        x_fused = x_rgb * mask + x_rgb
        
        return x_fused


@MODELS.register_module()
class TextGuidedRGBIRFusionV2(BaseModule):
    """
    Text-guided RGB-IR Fusion Module V2 (方案二改进版)
    
    多尺度文本引导融合模块，为每个金字塔层级独立应用融合。
    
    ==================== 配置示例 ====================
    
    text_guided_fusion=dict(
        type='TextGuidedRGBIRFusionV2',
        rgb_channels=[128, 256, 512],    # P3, P4, P5 的 RGB 通道数
        ir_channels=[64, 128, 256],      # P3, P4, P5 的 IR 通道数
        text_dim=512,                    # 文本特征维度
        num_classes=4,                   # 类别数
        beta=1.0,                        # X_ir 权重
        gamma=0.5,                       # S_map 权重
        
        # ===== V2 新增参数 =====
        gap_method='logits',             # 'logits' | 'max' | 'entropy'
        smap_method='sigmoid',           # 'sigmoid' | 'sigmoid_temp' | 'normalized'
        temperature=1.0,                 # 仅当 smap_method='sigmoid_temp' 时生效
    ),
    
    Args:
        rgb_channels (List[int]): RGB 特征通道数列表
        ir_channels (List[int]): IR 特征通道数列表
        text_dim (int): 文本特征维度
        num_classes (int): 类别数
        beta (float): X_ir 的权重系数
        gamma (float): S_map 的权重系数
        gap_method (str): GAP 计算方式
        smap_method (str): S_map 计算方式
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
        temperature: float = 1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.num_levels = len(rgb_channels)
        
        # 打印配置信息
        print(f"\n{'='*60}")
        print(f"[TextGuidedRGBIRFusionV2] 初始化配置:")
        print(f"  - RGB channels: {rgb_channels}")
        print(f"  - IR channels: {ir_channels}")
        print(f"  - Text dim: {text_dim}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Beta (X_ir weight): {beta}")
        print(f"  - Gamma (S_map weight): {gamma}")
        print(f"  - GAP method: {gap_method}")
        print(f"  - S_map method: {smap_method}")
        print(f"  - Temperature: {temperature}")
        print(f"{'='*60}\n")
        
        # 为每个层级创建独立的融合模块
        self.fusion_modules = nn.ModuleList([
            SingleLevelTextGuidedFusionV2(
                rgb_channels=rgb_channels[i],
                ir_channels=ir_channels[i],
                text_dim=text_dim,
                num_classes=num_classes,
                beta=beta,
                gamma=gamma,
                gap_method=gap_method,
                smap_method=smap_method,
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
