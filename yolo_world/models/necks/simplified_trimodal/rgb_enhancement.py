# Copyright (c) Tencent Inc. All rights reserved.
# RGB Enhancement Module for Trimodal Neck
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from ..trimodal_utils import compute_attention_consistency


class RGBEnhancementModule(BaseModule):
    """RGB增强模块：使用一致性调制的Text+IR来增强RGB
    
    完整流程：
        Step 0: 输入准备
        Step 1: Text-as-Query得到两路注意力
        Step 2: 计算一致性G
        Step 3: MLP生成调制超参数λ
        Step 4: 类别压缩+调制融合
        Step 5: 得到单token输出Y
        Step 6: token回注入到空间
    """
    
    def __init__(self,
                 channels: int,
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 agreement_temperature: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.channels = channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.agreement_temperature = agreement_temperature
        
        self.text_proj_q = nn.Linear(text_dim, hidden_dim)
        
        self.rgb_proj_k = nn.Linear(channels, hidden_dim)
        self.rgb_proj_v = nn.Linear(channels, hidden_dim)
        self.ir_proj_k = nn.Linear(channels, hidden_dim)
        self.ir_proj_v = nn.Linear(channels, hidden_dim)
        
        self.modulation_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        
        self.inject_conv = nn.Conv2d(hidden_dim, channels, 1)
        
        # 初始化为小的正值，确保模块从一开始就起作用
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
        self.scale = hidden_dim ** -0.5
        
    def forward(self,
                x_rgb: torch.Tensor,
                x_ir_new: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB特征 [B, C, H, W]
            x_ir_new: 更新后的IR特征 [B, C, H, W]
            text: 文本原型 [num_cls, text_dim]
        Returns:
            x_rgb_new: 增强后的RGB特征 [B, C, H, W]
        """
        B, C, H, W = x_rgb.shape
        N = H * W
        
        X_rgb = x_rgb.flatten(2).permute(0, 2, 1)
        X_ir = x_ir_new.flatten(2).permute(0, 2, 1)
        
        Q = self.text_proj_q(text)
        Q = F.normalize(Q, dim=-1)
        
        K_rgb = self.rgb_proj_k(X_rgb)
        V_rgb = self.rgb_proj_v(X_rgb)
        K_ir = self.ir_proj_k(X_ir)
        V_ir = self.ir_proj_v(X_ir)
        
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)
        
        A_rgb = torch.bmm(Q_expanded, K_rgb.transpose(-1, -2)) * self.scale
        A_rgb = F.softmax(A_rgb, dim=-1)
        
        A_ir = torch.bmm(Q_expanded, K_ir.transpose(-1, -2)) * self.scale
        A_ir = F.softmax(A_ir, dim=-1)
        
        G = compute_attention_consistency(A_rgb, A_ir)
        
        # 计算统计量（防止NaN）
        g_mean = G.mean(dim=-1)
        g_max = G.amax(dim=-1)
        g_std = G.std(dim=-1).clamp(min=1e-6)  # 防止std为0
        g_stats = torch.stack([g_mean, g_max, g_std], dim=-1)
        
        eta = self.modulation_mlp(g_stats)
        lambdas = F.softmax(eta, dim=-1)
        lambda_rgb = lambdas[:, 0:1]
        lambda_ir = lambdas[:, 1:2]
        
        w = F.softmax(G / self.agreement_temperature, dim=-1)
        
        # w: [B, num_cls], A_rgb: [B, num_cls, N]
        A_rgb_1 = (w.unsqueeze(-1) * A_rgb).sum(dim=1)  # [B, N]
        A_ir_1 = (w.unsqueeze(-1) * A_ir).sum(dim=1)    # [B, N]
        
        # lambda_rgb: [B, 1], A_rgb_1: [B, N] -> 广播得到 [B, N]
        A_bar = lambda_rgb * A_rgb_1 + lambda_ir * A_ir_1  # [B, N]
        
        A_fuse = A_bar / (A_bar.sum(dim=-1, keepdim=True) + 1e-6)  # [B, N]
        A_fuse = A_fuse.unsqueeze(1)  # [B, 1, N]
        
        # lambda_rgb: [B, 1] -> [B, 1, 1], V_rgb: [B, N, hidden_dim]
        V_fuse = lambda_rgb.view(-1, 1, 1) * V_rgb + lambda_ir.view(-1, 1, 1) * V_ir  # [B, N, hidden_dim]
        
        # A_fuse: [B, 1, N], V_fuse: [B, N, hidden_dim] -> Y: [B, 1, hidden_dim]
        Y = torch.bmm(A_fuse, V_fuse)
        
        # A_fuse_T: [B, N, 1], Y: [B, 1, hidden_dim] -> X_inj: [B, N, hidden_dim]
        A_fuse_T = A_fuse.transpose(-1, -2)  # [B, N, 1]
        X_inj = A_fuse_T * Y  # 广播: [B, N, 1] * [B, 1, hidden_dim] -> [B, N, hidden_dim]
        
        F_inj = X_inj.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        delta_F = self.inject_conv(F_inj)
        
        x_rgb_new = x_rgb + self.gamma * delta_F
        
        return x_rgb_new

