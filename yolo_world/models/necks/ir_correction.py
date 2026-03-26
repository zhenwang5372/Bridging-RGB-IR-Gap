# Copyright (c) Tencent Inc. All rights reserved.
# IR Correction Module for Trimodal Neck
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from .trimodal_utils import CrossAttention, SmoothConv, compute_ir_statistics


class IRCorrectionModule(BaseModule):
    """IR纠错模块：使用Text+RGB生成物理参数纠错IR特征
    
    物理参数（前景/背景各6个，共12个）：
        g: 传感器增益
        o: 传感器偏置
        κ: 反射污染系数
        b: 路径辐射/环境背景项
        τ: 大气透过率
        ε: 等效发射率
    """
    
    def __init__(self,
                 channels: int,
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.07,
                 eps: float = 1e-6,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.channels = channels
        self.text_dim = text_dim
        self.temperature = temperature
        self.eps = eps
        
        self.ir_to_text = nn.Linear(channels, text_dim)
        
        self.cross_attn = CrossAttention(
            query_dim=text_dim,
            key_dim=channels,
            value_dim=channels,
            hidden_dim=hidden_dim
        )
        
        self.smooth_conv = SmoothConv(hidden_channels=16)
        
        input_dim = hidden_dim + text_dim + 5
        self.param_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 12)
        )
        
        self._init_param_mlp()
        
    def _init_param_mlp(self):
        """初始化MLP使输出接近恒等变换"""
        with torch.no_grad():
            final_layer = self.param_mlp[-1]
            nn.init.zeros_(final_layer.weight)
            bias_init = torch.tensor([
                0.0, 0.0, -2.0, 0.0, 2.0, 2.0,
                0.0, 0.0, -2.0, 0.0, 2.0, 2.0
            ])
            final_layer.bias.copy_(bias_init)
    
    def forward(self, 
                x_ir: torch.Tensor, 
                x_rgb: torch.Tensor, 
                text: torch.Tensor) -> tuple:
        """
        Args:
            x_ir: IR特征 [B, C, H, W]
            x_rgb: RGB特征 [B, C, H, W]
            text: 文本原型 [num_cls, text_dim]
        Returns:
            x_ir_new: 纠错后的IR特征 [B, C, H, W]
            M: 语义区域图 [B, 1, H, W]
        """
        B, C, H, W = x_ir.shape
        N = H * W
        
        ir_pool = x_ir.mean(dim=[2, 3])
        u_ir = self.ir_to_text(ir_pool)
        u_ir = F.normalize(u_ir, dim=-1)
        
        logits = u_ir @ text.T / self.temperature
        w = F.softmax(logits, dim=-1)
        
        x_rgb_flat = x_rgb.flatten(2).permute(0, 2, 1)
        
        Y_rgb, A_rgb = self.cross_attn(text, x_rgb_flat, x_rgb_flat)
        
        w_expanded = w.unsqueeze(-1)
        y_rgb = (w_expanded * Y_rgb).sum(dim=1)
        
        a_rgb = (w.unsqueeze(-1) * A_rgb).sum(dim=1)
        a_rgb_2d = a_rgb.view(B, 1, H, W)
        M = self.smooth_conv(a_rgb_2d)
        
        stats = compute_ir_statistics(x_ir)
        
        mlp_input = torch.cat([y_rgb, u_ir, stats], dim=-1)
        theta = self.param_mlp(mlp_input)
        
        # 前景参数（带数值范围限制）
        g_fg = F.softplus(theta[:, 0]).clamp(max=10.0) + 1.0  # [1, 11]
        o_fg = theta[:, 1].clamp(-5.0, 5.0)
        kappa_fg = F.softplus(theta[:, 2]).clamp(max=5.0)  # [0, 5]
        b_fg = theta[:, 3].clamp(-5.0, 5.0)
        tau_fg = torch.sigmoid(theta[:, 4]).clamp(min=0.1)  # [0.1, 1] 避免太小
        epsilon_fg = torch.sigmoid(theta[:, 5]).clamp(min=0.1)  # [0.1, 1]
        
        # 背景参数
        g_bg = F.softplus(theta[:, 6]).clamp(max=10.0) + 1.0
        o_bg = theta[:, 7].clamp(-5.0, 5.0)
        kappa_bg = F.softplus(theta[:, 8]).clamp(max=5.0)
        b_bg = theta[:, 9].clamp(-5.0, 5.0)
        tau_bg = torch.sigmoid(theta[:, 10]).clamp(min=0.1)
        epsilon_bg = torch.sigmoid(theta[:, 11]).clamp(min=0.1)
        
        L_rgb = x_rgb.mean(dim=1, keepdim=True)
        L_rgb = (L_rgb - L_rgb.amin(dim=[2, 3], keepdim=True)) / \
                (L_rgb.amax(dim=[2, 3], keepdim=True) - L_rgb.amin(dim=[2, 3], keepdim=True) + self.eps)
        
        x_ir_fg = self._apply_physics(
            x_ir, L_rgb, 
            g_fg, o_fg, kappa_fg, b_fg, tau_fg, epsilon_fg
        )
        
        x_ir_bg = self._apply_physics(
            x_ir, L_rgb,
            g_bg, o_bg, kappa_bg, b_bg, tau_bg, epsilon_bg
        )
        
        # 物理纠错：直接使用校正后的特征
        x_ir_new = M * x_ir_fg + (1 - M) * x_ir_bg
        
        return x_ir_new, M
    
    def _apply_physics(self, 
                       x_ir: torch.Tensor,
                       L_rgb: torch.Tensor,
                       g: torch.Tensor, 
                       o: torch.Tensor,
                       kappa: torch.Tensor, 
                       b: torch.Tensor,
                       tau: torch.Tensor, 
                       epsilon: torch.Tensor) -> torch.Tensor:
        """应用物理纠错公式
        
        公式:
            1. X_tilde = (X_ir - o) / g          # 增益/偏置校正
            2. X_emit = X_tilde - κ * L_rgb - b  # 扣除反射项和环境项
            3. X_out = X_emit / (τ * ε + ε_0)    # 发射率归一化
        """
        B = x_ir.shape[0]
        
        g = g.view(B, 1, 1, 1)
        o = o.view(B, 1, 1, 1)
        kappa = kappa.view(B, 1, 1, 1)
        b = b.view(B, 1, 1, 1)
        tau = tau.view(B, 1, 1, 1)
        epsilon = epsilon.view(B, 1, 1, 1)
        
        # 使用残差形式增强稳定性: x_out = x_ir + delta
        # Step 1: 增益/偏置校正
        x_tilde = (x_ir - o) / (g + self.eps)
        
        # Step 2: 扣除反射项和环境项
        x_emit = x_tilde - kappa * L_rgb - b
        
        # Step 3: 发射率归一化（增大eps防止除零）
        x_out = x_emit / (tau * epsilon + 0.01)
        
        # 限制输出范围，防止极端值
        x_out = x_out.clamp(-100.0, 100.0)
        
        return x_out

