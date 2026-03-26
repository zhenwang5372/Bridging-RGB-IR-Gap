# Copyright (c) Tencent Inc. All rights reserved.
# Lightweight DCT-based IR Backbone V2 with Enhanced Features
#
# ============================================================================
# 相对于 LiteDCTGhostIRBackbone (V1) 的改进:
#   1. 【改进1】自适应频率分离 - 可学习的低频/高频边界 (替代固定 low_freq_ratio)
#   2. 【改进2】增强 Ghost Module - 多尺度(3x3+5x5) + SE + Skip Connection
#   3. 【改进3】轻量级 FPN - 多尺度特征融合 (可选)
#   4. 【改进4】热力学先验模块 - 利用红外物理特性 (可选)
# ============================================================================

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS


# ============================================================================
# DCT/IDCT 实现 (与 V1 相同)
# ============================================================================

def dct_2d_simple(x: torch.Tensor) -> torch.Tensor:
    """
    简化版 2D DCT (带正交归一化，防止数值爆炸)
    
    归一化因子: sqrt(2/N) 用于保持能量守恒
    """
    B, C, H, W = x.shape
    
    # 水平方向
    x_ext_h = torch.cat([x, x.flip(-1)], dim=-1)
    X_h = torch.fft.rfft(x_ext_h, dim=-1, norm='ortho')  # 添加正交归一化
    x_dct_h = X_h[..., :W].real
    
    # 垂直方向
    x_ext_v = torch.cat([x_dct_h, x_dct_h.flip(-2)], dim=-2)
    X_v = torch.fft.rfft(x_ext_v, dim=-2, norm='ortho')  # 添加正交归一化
    dct_coeff = X_v[..., :H, :].real
    
    return dct_coeff


def idct_2d_simple(dct_coeff: torch.Tensor) -> torch.Tensor:
    """
    简化版 2D IDCT (带正交归一化，与 DCT 保持一致)
    """
    B, C, H, W = dct_coeff.shape
    
    # 垂直方向 IDCT
    x_ext_v = torch.cat([dct_coeff, dct_coeff[..., 1:, :].flip(-2)], dim=-2)
    x_v = torch.fft.irfft(x_ext_v.to(torch.complex64), n=2*H-1, dim=-2, norm='ortho')  # 添加正交归一化
    x_idct_v = x_v[..., :H, :]
    
    # 水平方向 IDCT
    x_ext_h = torch.cat([x_idct_v, x_idct_v[..., 1:].flip(-1)], dim=-1)
    x_h = torch.fft.irfft(x_ext_h.to(torch.complex64), n=2*W-1, dim=-1, norm='ortho')  # 添加正交归一化
    output = x_h[..., :W]
    
    return output


# ============================================================================
# 【改进1】自适应频率分离
# ============================================================================

class AdaptiveFrequencySeparation(nn.Module):
    """
    【V2 新增】自适应频率分离模块
    
    相对于 V1 的改进:
        V1: 固定 low_freq_ratio = 0.25
        V2: 可学习的频率分离边界，使用软掩码
    
    Args:
        channels (int): 通道数
        init_ratio (float): 初始的低频区域比例
    """
    
    def __init__(self, channels: int, init_ratio: float = 0.25):
        super().__init__()
        
        # 【改进】可学习的频率分离边界
        self.boundary_h = nn.Parameter(torch.tensor([init_ratio]))
        self.boundary_w = nn.Parameter(torch.tensor([init_ratio]))
        
        # 【改进】可学习的软边界锐度
        self.sigma = nn.Parameter(torch.tensor([0.1]))
    
    def forward(self, dct_coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            low_freq_mask: 低频区域掩码 [B, C, H, W]
            high_freq_mask: 高频区域掩码 [B, C, H, W]
        """
        B, C, H, W = dct_coeff.shape
        device = dct_coeff.device
        
        # 生成坐标
        h_idx = torch.arange(H, device=device, dtype=torch.float32) / H
        w_idx = torch.arange(W, device=device, dtype=torch.float32) / W
        
        # 【改进】使用 Sigmoid 生成软边界掩码
        # boundary 控制分界位置，sigma 控制过渡锐度
        h_mask = torch.sigmoid((self.boundary_h - h_idx) / (self.sigma + 1e-6))
        w_mask = torch.sigmoid((self.boundary_w - w_idx) / (self.sigma + 1e-6))
        
        # 2D 低频掩码 (左上角为低频)
        low_freq_mask = h_mask.view(1, 1, H, 1) * w_mask.view(1, 1, 1, W)
        high_freq_mask = 1 - low_freq_mask
        
        return low_freq_mask, high_freq_mask


# ============================================================================
# 【改进2】增强 Ghost Module
# ============================================================================

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedGhostModule(nn.Module):
    """
    【V2 新增】增强版 Ghost Module
    
    相对于 V1 的改进:
        V1: 单尺度 3x3 Depthwise Conv 生成影子特征
        V2: 多尺度 (3x3 + 5x5) + SE通道注意力 + Skip Connection
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        ratio (int): 主特征与影子特征的比例
        use_se (bool): 是否使用 SE 注意力
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int = 2,
        use_se: bool = True
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.use_se = use_se
        
        # 主特征通道数
        init_channels = out_channels // ratio
        # 影子特征通道数
        new_channels = out_channels - init_channels
        new_channels_3x3 = new_channels // 2
        new_channels_5x5 = new_channels - new_channels_3x3
        
        # Primary Conv: 生成主特征
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True),
        )
        
        # 【改进】多尺度 Cheap Operation
        # 使用 Depthwise + Pointwise 设计（先 DW 保持通道，再 1x1 调整）
        self.cheap_op_3x3 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, 3,
                     padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Conv2d(init_channels, new_channels_3x3, 1, bias=False),
            nn.BatchNorm2d(new_channels_3x3),
        )
        self.cheap_op_5x5 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, 5,
                     padding=2, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Conv2d(init_channels, new_channels_5x5, 1, bias=False),
            nn.BatchNorm2d(new_channels_5x5),
        )
        
        self.act = nn.SiLU(inplace=True)
        
        # 【改进】SE 通道注意力
        if use_se:
            self.se = SELayer(out_channels, reduction=16)
        else:
            self.se = nn.Identity()
        
        # 【改进】Skip Connection
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        # 主特征
        x1 = self.primary_conv(x)
        
        # 【改进】多尺度影子特征
        x2_3x3 = self.cheap_op_3x3(x1)
        x2_5x5 = self.cheap_op_5x5(x1)
        x2 = self.act(torch.cat([x2_3x3, x2_5x5], dim=1))
        
        # 拼接
        out = torch.cat([x1, x2], dim=1)
        
        # 【改进】SE 注意力
        out = self.se(out)
        
        # 【改进】残差连接
        out = out + identity
        
        return out


# ============================================================================
# 【改进3】轻量级 FPN
# ============================================================================

class LightweightFPN(nn.Module):
    """
    【V2 新增】轻量级特征金字塔网络
    
    相对于 V1 的改进:
        V1: 三个 Stage 完全独立
        V2: 添加 Top-down 路径，多尺度特征融合
    
    Args:
        in_channels (List[int]): 输入通道数 [P3, P4, P5]
    """
    
    def __init__(self, in_channels: Sequence[int] = [64, 128, 256]):
        super().__init__()
        
        # 统一的中间维度
        mid_channels = 64
        
        # Lateral connections (1x1 conv 降维)
        self.lateral_p5 = nn.Conv2d(in_channels[2], mid_channels, 1, bias=False)
        self.lateral_p4 = nn.Conv2d(in_channels[1], mid_channels, 1, bias=False)
        self.lateral_p3 = nn.Conv2d(in_channels[0], mid_channels, 1, bias=False)
        
        # Smooth convs (3x3 conv 平滑)
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.SiLU(inplace=True)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, feats: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            feats: (P3, P4, P5) 原始多尺度特征
        
        Returns:
            (P3_out, P4_out, P5_out) 融合后的多尺度特征
        """
        p3, p4, p5 = feats
        
        # Top-down pathway
        p5_lateral = self.lateral_p5(p5)
        
        p4_lateral = self.lateral_p4(p4)
        p4_lateral = p4_lateral + F.interpolate(
            p5_lateral, size=p4.shape[-2:], mode='nearest'
        )
        
        p3_lateral = self.lateral_p3(p3)
        p3_lateral = p3_lateral + F.interpolate(
            p4_lateral, size=p3.shape[-2:], mode='nearest'
        )
        
        # Smooth
        p4_out = self.smooth_p4(p4_lateral)
        p3_out = self.smooth_p3(p3_lateral)
        
        return (p3_out, p4_out, p5)


# ============================================================================
# 【改进4】热力学先验模块
# ============================================================================

class ThermalPriorModule(nn.Module):
    """
    【V2 新增】热力学先验模块
    
    利用红外图像的物理特性:
        - 温度高的区域（亮）通常是目标
        - 温度梯度大的区域是边缘
    
    Args:
        channels (int): 输入通道数
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 热强度通道 (全局特征)
        self.intensity_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        # 热梯度通道 (边缘特征)
        self.gradient_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 热强度 (全局加权)
        intensity_weight = self.intensity_branch(x).view(B, C, 1, 1)
        x_intensity = x * intensity_weight
        
        # 热梯度 (边缘增强)
        x_gradient = torch.abs(self.gradient_conv(x))
        
        # 融合
        out = self.fusion(torch.cat([x_intensity, x_gradient], dim=1))
        
        # 残差
        return out + x


# ============================================================================
# SpectralBlock V2
# ============================================================================

class SpectralBlockDCTGhostV2(nn.Module):
    """
    【V2 改进】频谱处理模块
    
    相对于 V1 的改进:
        1. 使用 AdaptiveFrequencySeparation 替代固定比例
        2. 使用 EnhancedGhostModule 替代普通 Ghost
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        freq_ratio (float): 频域分支占输出通道的比例
        ghost_ratio (int): Ghost Module 的比例
        init_low_freq_ratio (float): 初始低频区域比例
        use_se (bool): Ghost Module 是否使用 SE
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        freq_ratio: float = 0.5,
        ghost_ratio: int = 2,
        init_low_freq_ratio: float = 0.25,
        use_se: bool = True
    ):
        super().__init__()
        
        self.freq_ratio = freq_ratio
        
        # 通道分配
        self.freq_channels = int(out_channels * freq_ratio)
        self.spatial_channels = out_channels - self.freq_channels
        
        # ========================================
        # Spatial Branch: 【改进】增强 Ghost Module
        # ========================================
        self.spatial_conv = EnhancedGhostModule(
            in_channels=in_channels,
            out_channels=self.spatial_channels,
            ratio=ghost_ratio,
            use_se=use_se
        )
        
        # ========================================
        # Frequency Branch: 【改进】自适应频率分离
        # ========================================
        self.freq_proj = nn.Conv2d(in_channels, self.freq_channels, 1, bias=False)
        
        # 【改进】自适应频率分离
        self.adaptive_sep = AdaptiveFrequencySeparation(
            channels=self.freq_channels,
            init_ratio=init_low_freq_ratio
        )
        
        # 可学习的频率调制权重
        self.low_freq_weight = nn.Parameter(torch.ones(1, self.freq_channels, 1, 1))
        self.high_freq_weight = nn.Parameter(torch.ones(1, self.freq_channels, 1, 1) * 0.5)
        
        # ========================================
        # Fusion
        # ========================================
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Spatial Branch ===
        spatial_feat = self.spatial_conv(x)
        
        # === Frequency Branch ===
        freq_input = self.freq_proj(x)
        
        # DCT 正向变换
        dct_coeff = dct_2d_simple(freq_input)
        
        # 【改进】自适应频率分离
        low_mask, high_mask = self.adaptive_sep(dct_coeff)
        
        # 频率调制
        low_freq = dct_coeff * low_mask * self.low_freq_weight
        high_freq = dct_coeff * high_mask * self.high_freq_weight
        dct_modulated = low_freq + high_freq
        
        # IDCT 逆向变换
        freq_feat = idct_2d_simple(dct_modulated)
        
        # === Fusion ===
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        output = self.fusion(combined)
        
        return output


# ============================================================================
# LiteDCTGhostIRBackboneV2
# ============================================================================

@MODELS.register_module()
class LiteDCTGhostIRBackboneV2(BaseModule):
    """
    轻量级 IR Backbone V2
    
    相对于 V1 (LiteDCTGhostIRBackbone) 的改进:
        1. 【改进1】自适应频率分离 - SpectralBlockDCTGhostV2 中使用
        2. 【改进2】增强 Ghost Module - 多尺度 + SE + Skip
        3. 【改进3】轻量级 FPN - 可选的多尺度特征融合
        4. 【改进4】热力学先验 - 可选的物理先验模块
    
    Architecture:
        Stem → SpectralBlockV2 × 3 → [FPN] → [ThermalPrior] → Multi-scale outputs
    
    Args:
        in_channels (int): 输入通道数，默认 3
        base_channels (int): 基础通道数，默认 32
        out_indices (Sequence[int]): 输出的 stage 索引，默认 (0, 1, 2)
        frozen_stages (int): 冻结的 stage 数，-1 表示不冻结
        freq_ratio (float): 频域分支通道比例，默认 0.5
        ghost_ratio (int): Ghost Module 比例，默认 2
        init_low_freq_ratio (float): 初始低频区域比例，默认 0.25
        use_fpn (bool): 【V2新增】是否使用 FPN，默认 True
        use_thermal_prior (bool): 【V2新增】是否使用热力先验，默认 False
        use_se (bool): 【V2新增】Ghost 是否使用 SE，默认 True
        norm_eval (bool): 是否在 eval 时设置 BN 为 eval 模式
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_indices: Sequence[int] = (0, 1, 2),
        frozen_stages: int = -1,
        freq_ratio: float = 0.5,
        ghost_ratio: int = 2,
        init_low_freq_ratio: float = 0.25,
        use_fpn: bool = True,
        use_thermal_prior: bool = False,
        use_se: bool = True,
        norm_eval: bool = False,
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.freq_ratio = freq_ratio
        self.ghost_ratio = ghost_ratio
        self.init_low_freq_ratio = init_low_freq_ratio
        self.use_fpn = use_fpn
        self.use_thermal_prior = use_thermal_prior
        self.use_se = use_se
        self.norm_eval = norm_eval
        
        # 输出通道数 (与 V1 相同)
        self.out_channels = [
            base_channels * 2,   # P3: 64
            base_channels * 4,   # P4: 128
            base_channels * 8,   # P5: 256
        ]
        
        # Stem (与 V1 相同)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1 → P3
        self.stage1 = nn.Sequential(
            SpectralBlockDCTGhostV2(
                base_channels, self.out_channels[0],
                freq_ratio=freq_ratio,
                ghost_ratio=ghost_ratio,
                init_low_freq_ratio=init_low_freq_ratio,
                use_se=use_se
            ),
            nn.Conv2d(self.out_channels[0], self.out_channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.SiLU(inplace=True),
        )
        
        # Stage 2 → P4
        self.stage2 = nn.Sequential(
            SpectralBlockDCTGhostV2(
                self.out_channels[0], self.out_channels[1],
                freq_ratio=freq_ratio,
                ghost_ratio=ghost_ratio,
                init_low_freq_ratio=init_low_freq_ratio,
                use_se=use_se
            ),
            nn.Conv2d(self.out_channels[1], self.out_channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels[1]),
            nn.SiLU(inplace=True),
        )
        
        # Stage 3 → P5
        self.stage3 = nn.Sequential(
            SpectralBlockDCTGhostV2(
                self.out_channels[1], self.out_channels[2],
                freq_ratio=freq_ratio,
                ghost_ratio=ghost_ratio,
                init_low_freq_ratio=init_low_freq_ratio,
                use_se=use_se
            ),
            nn.Conv2d(self.out_channels[2], self.out_channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels[2]),
            nn.SiLU(inplace=True),
        )
        
        # 【改进3】轻量级 FPN (可选)
        if use_fpn:
            self.fpn = LightweightFPN(self.out_channels)
        else:
            self.fpn = None
        
        # 【改进4】热力学先验 (可选)
        if use_thermal_prior:
            self.thermal_priors = nn.ModuleList([
                ThermalPriorModule(ch) for ch in self.out_channels
            ])
        else:
            self.thermal_priors = None
        
        self.layers = ['stem', 'stage1', 'stage2', 'stage3']
        self._freeze_stages()
    
    def _freeze_stages(self):
        """冻结指定 stage 的参数"""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        
        for i in range(min(self.frozen_stages, 3)):
            stage = getattr(self, f'stage{i+1}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: 输入 [B, C, H, W]
        
        Returns:
            (P3, P4, P5) 多尺度特征
        """
        outs = []
        
        x = self.stem(x)           # 1/2
        
        x = self.stage1(x)         # 1/8 → P3
        outs.append(x)
        
        x = self.stage2(x)         # 1/16 → P4
        outs.append(x)
        
        x = self.stage3(x)         # 1/32 → P5
        outs.append(x)
        
        # 【改进3】FPN 融合
        if self.fpn is not None:
            outs = list(self.fpn(tuple(outs)))
        
        # 【改进4】热力学先验
        if self.thermal_priors is not None:
            outs = [self.thermal_priors[i](outs[i]) for i in range(len(outs))]
        
        # 按 out_indices 返回
        return tuple(outs[i] for i in self.out_indices if i < len(outs))
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
