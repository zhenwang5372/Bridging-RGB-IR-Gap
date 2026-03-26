# Copyright (c) Tencent Inc. All rights reserved.
# Lightweight FFT-based IR Backbone for RGB-IR Fusion Detection
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for channel attention.
    
    Reference: "Squeeze-and-Excitation Networks" (CVPR 2018)
    
    Architecture:
        Input (B, C, H, W)
          ↓
        Global Average Pooling → (B, C, 1, 1)
          ↓
        FC1 (C → C//reduction) + ReLU
          ↓
        FC2 (C//reduction → C) + Sigmoid → Channel Weights
          ↓
        Input × Channel Weights → Output (B, C, H, W)
    
    即插即用说明：
        在 SpectralBlock 中通过 use_se 参数控制：
        - use_se=True:  self.se = SELayer(...)  # 使用 SE 加权
        - use_se=False: self.se = nn.Identity() # 无参数，直接返回输入
    
    Args:
        channels (int): Number of input channels.
        reduction (int): Channel reduction ratio. Defaults to 16.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)
        
        # SE module components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Channel-weighted tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)
    
    def extra_repr(self) -> str:
        """Extra representation for print."""
        return f'channels={self.channels}, reduction={self.reduction}'


class SpectralBlockPreSE(nn.Module):
    """
    Lightweight block combining spatial convolution and frequency domain processing.
    
    Architecture (with SE):
        Input 
          ↓
        ┌─────────────────┬──────────────────────────┐
        │  Spatial Branch │     Frequency Branch     │
        │  Depthwise Conv │  FFT → Modulation → IFFT │
        └────────┬────────┴─────────────┬────────────┘
                 └──────── Concat ──────┘
                           ↓
                     SE-Block (动态通道加权) ← 新增！
                           ↓
                     1x1 Conv (特征融合)
                           ↓
                        Output
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for depthwise conv (default: 3)
        freq_ratio (float): Ratio of channels to process in frequency domain (0-1)
        use_se (bool): Whether to use SE attention after concat. Defaults to True.
            - True: 创建 SELayer，增加通道注意力
            - False: 使用 nn.Identity()，无额外参数
        se_reduction (int): SE reduction ratio. Defaults to 16.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, freq_ratio: float = 0.5,
                 use_se: bool = True, se_reduction: int = 16):
        super(SpectralBlock, self).__init__()
        
        self.freq_ratio = freq_ratio
        self.freq_channels = int(out_channels * freq_ratio)
        self.spatial_channels = out_channels - self.freq_channels
        self.use_se = use_se
        
        # Spatial branch: Depthwise + Pointwise Conv (MobileNet style)
        self.spatial_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, self.spatial_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.spatial_channels),
        )
        
        # Frequency branch: Learnable amplitude and phase modulation
        self.freq_conv = nn.Conv2d(in_channels, self.freq_channels, kernel_size=1, bias=False)
        
        # Learnable modulation weights (applied in frequency domain)
        self.amp_weight = nn.Parameter(torch.ones(1, self.freq_channels, 1, 1))
        self.phase_weight = nn.Parameter(torch.zeros(1, self.freq_channels, 1, 1))
        
        # SE-Block for channel attention after concat (即插即用)
        # use_se=True: 创建 SELayer
        # use_se=False: 使用 Identity，无额外参数开销
        if use_se:
            self.se = SELayer(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()  # 无参数，直接返回输入
        
        # Fusion: 1x1 Conv for feature mixing
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        spatial_feat = self.spatial_conv(x)
        
        # Frequency branch
        freq_input = self.freq_conv(x)
        
        # FFT: Transform to frequency domain
        freq_domain = torch.fft.rfft2(freq_input, norm='ortho')
        
        # Separate amplitude and phase
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)
        
        # Apply learnable modulation
        amplitude_mod = amplitude * self.amp_weight
        phase_mod = phase + self.phase_weight
        
        # Reconstruct complex tensor
        freq_modulated = amplitude_mod * torch.exp(1j * phase_mod)
        
        # IFFT: Transform back to spatial domain
        freq_feat = torch.fft.irfft2(freq_modulated, s=freq_input.shape[-2:], norm='ortho')
        
        # Step 1: Concatenate spatial and frequency features
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        
        # Step 2: SE-Block - 动态通道加权 (即插即用，可通过 use_se 开关)
        combined = self.se(combined)
        
        # Step 3: 1x1 Conv - 特征融合
        output = self.fusion(combined)
        
        return output


@MODELS.register_module()
class LiteFFTIRBackbonePreSE(BaseModule):
    """
    Lightweight IR backbone with frequency-enhanced feature extraction.
    
    Architecture:
        Stem → SpectralBlock × 3 → Multi-scale outputs [P3, P4, P5]
    
    SpectralBlock with SE:
        Spatial Branch + Frequency Branch → Concat → SE-Block → 1x1 Conv
    
    Parameter count: ~1/8 of YOLOv8s backbone
    
    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Base channel count for the network. Defaults to 32.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (0, 1, 2) which corresponds to P3, P4, P5.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        freq_ratio (float): Ratio of channels for frequency domain processing.
            Defaults to 0.5.
        use_se (bool): Whether to use SE attention in SpectralBlocks.
            Defaults to True. Set to False for ablation study.
        se_reduction (int): SE channel reduction ratio. Defaults to 16.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 base_channels: int = 32,
                 out_indices: Sequence[int] = (0, 1, 2),
                 frozen_stages: int = -1,
                 freq_ratio: float = 0.5,
                 use_se: bool = True,
                 se_reduction: int = 16,
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = None):
        super(LiteFFTIRBackbone, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.freq_ratio = freq_ratio
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.norm_eval = norm_eval
        
        # Output channels for each stage (matches YOLOv8s P3, P4, P5 with widen_factor=0.5)
        # YOLOv8s outputs: [128, 256, 512]
        # IR backbone outputs: [64, 128, 256] (half for lightweight)
        self.out_channels = [
            base_channels * 2,   # P3: 64
            base_channels * 4,   # P4: 128
            base_channels * 8,   # P5: 256
        ]
        
        # Stem: Initial downsampling (1/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1 → P3 (1/8 resolution)
        self.stage1 = nn.Sequential(
            SpectralBlock(base_channels, base_channels * 2, 
                         freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        
        # Stage 2 → P4 (1/16 resolution)
        self.stage2 = nn.Sequential(
            SpectralBlock(base_channels * 2, base_channels * 4, 
                         freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
        )
        
        # Stage 3 → P5 (1/32 resolution)
        self.stage3 = nn.Sequential(
            SpectralBlock(base_channels * 4, base_channels * 8, 
                         freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU(inplace=True),
        )
        
        self.layers = ['stem', 'stage1', 'stage2', 'stage3']
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
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
        """Forward pass through the IR backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of feature tensors (P3, P4, P5)
        """
        outs = []
        
        x = self.stem(x)           # 1/2
        
        x = self.stage1(x)         # 1/8 → P3
        if 0 in self.out_indices:
            outs.append(x)
        
        x = self.stage2(x)         # 1/16 → P4
        if 1 in self.out_indices:
            outs.append(x)
        
        x = self.stage3(x)         # 1/32 → P5
        if 2 in self.out_indices:
            outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen stages."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


# ============================================================================
# Ablation Study: Post-SE Variants
# ============================================================================

class SpectralBlockPostSE(nn.Module):
    """
    SpectralBlock variant with Post-SE structure for ablation study.
    
    Architecture Comparison:
    
    SpectralBlock (Pre-SE, 先加权后融合):
        Concat → SE-Block → 1x1 Conv → Output
        
    SpectralBlockPostSE (Post-SE, 先融合后加权):
        Concat → 1x1 Conv → SE-Block → Output
    
    Theory:
        Post-SE: 先通过 1x1 Conv 混合 Spatial/Frequency 特征，
        再使用 SE-Block 对混合后的特征进行通道加权。
        
        对比 Pre-SE 的理论：
        - Pre-SE: SE 对解耦的 Spatial/Freq 通道分别加权（Domain Reweighting）
        - Post-SE: SE 对混合后的特征加权（Feature Reweighting）
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for depthwise conv (default: 3)
        freq_ratio (float): Ratio of channels to process in frequency domain (0-1)
        use_se (bool): Whether to use SE attention after conv. Defaults to True.
        se_reduction (int): SE reduction ratio. Defaults to 16.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, freq_ratio: float = 0.5,
                 use_se: bool = True, se_reduction: int = 16):
        super(SpectralBlockPostSE, self).__init__()
        
        self.freq_ratio = freq_ratio
        self.freq_channels = int(out_channels * freq_ratio)
        self.spatial_channels = out_channels - self.freq_channels
        self.use_se = use_se
        
        # Spatial branch: Depthwise + Pointwise Conv (MobileNet style)
        self.spatial_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, self.spatial_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.spatial_channels),
        )
        
        # Frequency branch: Learnable amplitude and phase modulation
        self.freq_conv = nn.Conv2d(in_channels, self.freq_channels, kernel_size=1, bias=False)
        
        # Learnable modulation weights (applied in frequency domain)
        self.amp_weight = nn.Parameter(torch.ones(1, self.freq_channels, 1, 1))
        self.phase_weight = nn.Parameter(torch.zeros(1, self.freq_channels, 1, 1))
        
        # Fusion: 1x1 Conv for feature mixing (BEFORE SE in Post-SE variant)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # SE-Block for channel attention AFTER fusion (Post-SE)
        if use_se:
            self.se = SELayer(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        spatial_feat = self.spatial_conv(x)
        
        # Frequency branch
        freq_input = self.freq_conv(x)
        
        # FFT: Transform to frequency domain
        freq_domain = torch.fft.rfft2(freq_input, norm='ortho')
        
        # Separate amplitude and phase
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)
        
        # Apply learnable modulation
        amplitude_mod = amplitude * self.amp_weight
        phase_mod = phase + self.phase_weight
        
        # Reconstruct complex tensor
        freq_modulated = amplitude_mod * torch.exp(1j * phase_mod)
        
        # IFFT: Transform back to spatial domain
        freq_feat = torch.fft.irfft2(freq_modulated, s=freq_input.shape[-2:], norm='ortho')
        
        # Step 1: Concatenate spatial and frequency features
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        
        # Step 2: 1x1 Conv - 特征融合 (先融合)
        fused = self.fusion(combined)
        
        # Step 3: SE-Block - 动态通道加权 (后加权)
        output = self.se(fused)
        
        return output


@MODELS.register_module()
class LiteFFTIRBackbonePostSE(BaseModule):
    """
    Lightweight IR backbone with Post-SE structure for ablation study.
    
    Architecture:
        Stem → SpectralBlockPostSE × 3 → Multi-scale outputs [P3, P4, P5]
    
    SpectralBlockPostSE (Post-SE, 先融合后加权):
        Spatial Branch + Frequency Branch → Concat → 1x1 Conv → SE-Block
    
    Comparison with LiteFFTIRBackbone (Pre-SE):
        - LiteFFTIRBackbone: Concat → SE → Conv (先加权后融合)
        - LiteFFTIRBackbonePostSE: Concat → Conv → SE (先融合后加权)
    
    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Base channel count for the network. Defaults to 32.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (0, 1, 2) which corresponds to P3, P4, P5.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        freq_ratio (float): Ratio of channels for frequency domain processing.
            Defaults to 0.5.
        use_se (bool): Whether to use SE attention in SpectralBlocks.
            Defaults to True. Set to False for ablation study.
        se_reduction (int): SE channel reduction ratio. Defaults to 16.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 base_channels: int = 32,
                 out_indices: Sequence[int] = (0, 1, 2),
                 frozen_stages: int = -1,
                 freq_ratio: float = 0.5,
                 use_se: bool = True,
                 se_reduction: int = 16,
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = None):
        super(LiteFFTIRBackbonePostSE, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.freq_ratio = freq_ratio
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.norm_eval = norm_eval
        
        # Output channels for each stage
        self.out_channels = [
            base_channels * 2,   # P3: 64
            base_channels * 4,   # P4: 128
            base_channels * 8,   # P5: 256
        ]
        
        # Stem: Initial downsampling (1/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1 → P3 (1/8 resolution) - Using SpectralBlockPostSE
        self.stage1 = nn.Sequential(
            SpectralBlockPostSE(base_channels, base_channels * 2, 
                               freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        
        # Stage 2 → P4 (1/16 resolution) - Using SpectralBlockPostSE
        self.stage2 = nn.Sequential(
            SpectralBlockPostSE(base_channels * 2, base_channels * 4, 
                               freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
        )
        
        # Stage 3 → P5 (1/32 resolution) - Using SpectralBlockPostSE
        self.stage3 = nn.Sequential(
            SpectralBlockPostSE(base_channels * 4, base_channels * 8, 
                               freq_ratio=freq_ratio, use_se=use_se, se_reduction=se_reduction),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU(inplace=True),       
        )
        
        self.layers = ['stem', 'stage1', 'stage2', 'stage3']
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
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
        """Forward pass through the IR backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of feature tensors (P3, P4, P5)
        """
        outs = []
        
        x = self.stem(x)           # 1/2
        
        x = self.stage1(x)         # 1/8 → P3
        if 0 in self.out_indices:
            outs.append(x)
        
        x = self.stage2(x)         # 1/16 → P4
        if 1 in self.out_indices:
            outs.append(x)
        
        x = self.stage3(x)         # 1/32 → P5
        if 2 in self.out_indices:
            outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen stages."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

class SpectralBlock(nn.Module):
    """
    Lightweight block combining spatial convolution and frequency domain processing.
    
    Architecture (without SE):
        Input 
          ↓
        ┌─────────────────┬──────────────────────────┐
        │  Spatial Branch │     Frequency Branch     │
        │  Depthwise Conv │  FFT → Modulation → IFFT │
        └────────┬────────┴─────────────┬────────────┘
                 └──────── Concat ──────┘
                           ↓
                     1x1 Conv (特征融合)
                           ↓
                        Output
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for depthwise conv (default: 3)
        freq_ratio (float): Ratio of channels to process in frequency domain (0-1)
        use_se (bool): Whether to use SE attention after concat. Defaults to True.
            - True: 创建 SELayer，增加通道注意力
            - False: 使用 nn.Identity()，无额外参数
        se_reduction (int): SE reduction ratio. Defaults to 16.
    """

    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, freq_ratio: float = 0.5):
               
        super(SpectralBlock, self).__init__()
        
        self.freq_ratio = freq_ratio
        self.freq_channels = int(out_channels * freq_ratio)
        self.spatial_channels = out_channels - self.freq_channels

        
        # Spatial branch: Depthwise + Pointwise Conv (MobileNet style)
        self.spatial_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, self.spatial_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.spatial_channels),
        )
        
        # Frequency branch: Learnable amplitude and phase modulation
        self.freq_conv = nn.Conv2d(in_channels, self.freq_channels, kernel_size=1, bias=False)
        
        # Learnable modulation weights (applied in frequency domain)
        self.amp_weight = nn.Parameter(torch.ones(1, self.freq_channels, 1, 1))
        self.phase_weight = nn.Parameter(torch.zeros(1, self.freq_channels, 1, 1))
        

  
        # Fusion: 1x1 Conv for feature mixing
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        spatial_feat = self.spatial_conv(x)
        
        # Frequency branch
        freq_input = self.freq_conv(x)
        
        # FFT: Transform to frequency domain
        freq_domain = torch.fft.rfft2(freq_input, norm='ortho')
        
        # Separate amplitude and phase
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)
        
        # Apply learnable modulation
        amplitude_mod = amplitude * self.amp_weight
        phase_mod = phase + self.phase_weight
        
        # Reconstruct complex tensor
        freq_modulated = amplitude_mod * torch.exp(1j * phase_mod)
        
        # IFFT: Transform back to spatial domain
        freq_feat = torch.fft.irfft2(freq_modulated, s=freq_input.shape[-2:], norm='ortho')
        
        # Step 1: Concatenate spatial and frequency features
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        
        # Step 2: SE-Block - 动态通道加权 (即插即用，可通过 use_se 开关)
        
        # Step 3: 1x1 Conv - 特征融合
        output = self.fusion(combined)
        
        return output


@MODELS.register_module()
class LiteFFTIRBackbone(BaseModule):
    """
    Lightweight IR backbone with frequency-enhanced feature extraction.
    
    Architecture:
        Stem → SpectralBlock × 3 → Multi-scale outputs [P3, P4, P5]
    
    SpectralBlock with SE:
        Spatial Branch + Frequency Branch → Concat → SE-Block → 1x1 Conv
    
    Parameter count: ~1/8 of YOLOv8s backbone
    
    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Base channel count for the network. Defaults to 32.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (0, 1, 2) which corresponds to P3, P4, P5.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        freq_ratio (float): Ratio of channels for frequency domain processing.
            Defaults to 0.5.
        use_se (bool): Whether to use SE attention in SpectralBlocks.
            Defaults to True. Set to False for ablation study.
        se_reduction (int): SE channel reduction ratio. Defaults to 16.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 base_channels: int = 32,
                 out_indices: Sequence[int] = (0, 1, 2),
                 frozen_stages: int = -1,
                 freq_ratio: float = 0.5,
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = None):
        super(LiteFFTIRBackbone, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.freq_ratio = freq_ratio
        self.norm_eval = norm_eval
        
        # Output channels for each stage (matches YOLOv8s P3, P4, P5 with widen_factor=0.5)
        # YOLOv8s outputs: [128, 256, 512]
        # IR backbone outputs: [64, 128, 256] (half for lightweight)
        self.out_channels = [
            base_channels * 2,   # P3: 64
            base_channels * 4,   # P4: 128
            base_channels * 8,   # P5: 256
        ]
        
        # Stem: Initial downsampling (1/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1 → P3 (1/8 resolution)
        self.stage1 = nn.Sequential(
            SpectralBlock(base_channels, base_channels * 2, 
                         freq_ratio=freq_ratio,),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        
        # Stage 2 → P4 (1/16 resolution)
        self.stage2 = nn.Sequential(
            SpectralBlock(base_channels * 2, base_channels * 4, 
                         freq_ratio=freq_ratio),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
        )
        
        # Stage 3 → P5 (1/32 resolution)
        self.stage3 = nn.Sequential(
            SpectralBlock(base_channels * 4, base_channels * 8, 
                         freq_ratio=freq_ratio),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU(inplace=True),
        )
        
        self.layers = ['stem', 'stage1', 'stage2', 'stage3']
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
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
        """Forward pass through the IR backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of feature tensors (P3, P4, P5)
        """
        outs = []
        
        x = self.stem(x)           # 1/2
        
        x = self.stage1(x)         # 1/8 → P3
        if 0 in self.out_indices:
            outs.append(x)
        
        x = self.stage2(x)         # 1/16 → P4
        if 1 in self.out_indices:
            outs.append(x)
        
        x = self.stage3(x)         # 1/32 → P5
        if 2 in self.out_indices:
            outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen stages."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
