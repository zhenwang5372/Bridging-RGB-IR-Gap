# Copyright (c) Tencent Inc. All rights reserved.
# Text Update Module for Trimodal Neck
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from ..trimodal_utils import IRGuidedCBAM


class TextUpdateModule(BaseModule):
    """Text更新模块：使用IR_new和RGB_new更新Text原型
    
    完整流程：
        Step A: IR语义锚点 → 类别权重w
        Step B: IR-Guided CBAM处理RGB
        Step C: Text作为Query，检索RGB视觉证据
        Step D: 逐类别残差更新Text
    
    默认使用P4尺度特征，可通过配置切换尺度进行消融实验
    """
    
    def __init__(self,
                 channels: int,
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.07,
                 gamma: float = 0.1,
                 cbam_reduction: int = 16,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.channels = channels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        # 改为可学习参数，初始化为传入值
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        
        self.ir_to_text = nn.Linear(channels, text_dim)
        
        self.ir_guided_cbam = IRGuidedCBAM(
            channels=channels,
            reduction=cbam_reduction,
            kernel_size=7
        )
        
        self.text_proj_q = nn.Linear(text_dim, hidden_dim)
        self.rgb_proj_k = nn.Linear(channels, hidden_dim)
        self.rgb_proj_v = nn.Linear(channels, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        
        self.align_mlp = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, text_dim)
        )
        
    def forward(self,
                x_rgb: torch.Tensor,
                x_ir: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB特征 [B, C, H, W]（更新后的RGB）
            x_ir: IR特征 [B, C, H, W]（更新后的IR）
            text: 文本原型 [num_cls, text_dim]
        Returns:
            text_new: 更新后的文本原型 [B, num_cls, text_dim]
        """
        B, C, H, W = x_rgb.shape
        N = H * W
        num_cls = text.shape[0]
        
        ir_pool = x_ir.mean(dim=[2, 3])
        u_ir = self.ir_to_text(ir_pool)
        u_ir = F.normalize(u_ir, dim=-1)
        
        logits = u_ir @ text.T / self.temperature
        w = F.softmax(logits, dim=-1)
        
        x_rgb_enhanced = self.ir_guided_cbam(x_rgb, x_ir)
        
        X_rgb = x_rgb_enhanced.flatten(2).permute(0, 2, 1)
        
        Q = self.text_proj_q(text)
        K = self.rgb_proj_k(X_rgb)
        V = self.rgb_proj_v(X_rgb)
        
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)
        
        A = torch.bmm(Q_expanded, K.transpose(-1, -2)) * self.scale
        A = F.softmax(A, dim=-1)
        
        Y_rgb = torch.bmm(A, V)
        
        Y_aligned = self.align_mlp(Y_rgb)
        
        w_expanded = w.unsqueeze(-1)
        delta = self.gamma * w_expanded * Y_aligned
        
        text_expanded = text.unsqueeze(0).expand(B, -1, -1)
        text_new = text_expanded + delta
        
        text_new = F.normalize(text_new, dim=-1)
        
        return text_new

