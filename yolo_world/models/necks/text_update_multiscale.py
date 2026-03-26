# Copyright (c) Tencent Inc. All rights reserved.
# Multi-scale Text Update Module for Trimodal Neck
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from .trimodal_utils import IRGuidedCBAM


class TextUpdateMultiScale(BaseModule):
    """多尺度Text更新模块：保持原始Text更新策略，但融合多尺度信息
    
    核心思路：
        1. 对P3/P4/P5分别执行完整的Text更新流程
        2. 得到三个Text更新增量 delta_P3, delta_P4, delta_P5
        3. 加权融合三个增量，得到最终的Text更新
    
    优势：
        - 保持原始的IR-Guided CBAM + Cross-Attention策略
        - 充分利用多尺度信息（小/中/大物体）
        - 可学习的尺度权重，自适应调整
    """
    
    def __init__(self,
                 in_channels: list = [128, 256, 512],  # P3, P4, P5的通道数
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.07,
                 gamma: float = 0.1,
                 cbam_reduction: int = 16,
                 fusion_method: str = 'learned_weight',  # 'learned_weight', 'equal', 'attention'
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.fusion_method = fusion_method
        
        # 每个尺度独立的处理模块
        self.scale_modules = nn.ModuleList()
        for ch in in_channels:
            scale_module = nn.ModuleDict({
                'ir_to_text': nn.Linear(ch, text_dim),
                'ir_guided_cbam': IRGuidedCBAM(
                    channels=ch,
                    reduction=cbam_reduction,
                    kernel_size=7
                ),
                'text_proj_q': nn.Linear(text_dim, hidden_dim),
                'rgb_proj_k': nn.Linear(ch, hidden_dim),
                'rgb_proj_v': nn.Linear(ch, hidden_dim),
                'align_mlp': nn.Sequential(
                    nn.Linear(hidden_dim, text_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(text_dim, text_dim)
                )
            })
            self.scale_modules.append(scale_module)
        
        self.scale = hidden_dim ** -0.5
        
        # 多尺度融合权重
        if fusion_method == 'learned_weight':
            # 可学习的固定权重
            self.scale_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)
        elif fusion_method == 'attention':
            # 基于IR特征的动态权重
            self.scale_attention = nn.Sequential(
                nn.Linear(text_dim * self.num_levels, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.num_levels)
            )
        # 'equal' 不需要额外参数
        
    def forward(self,
                x_rgb_list: list,
                x_ir_list: list,
                text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb_list: RGB特征列表 [P3, P4, P5]，每个 [B, C, H, W]
            x_ir_list: IR特征列表 [P3, P4, P5]，每个 [B, C, H, W]
            text: 文本原型 [num_cls, text_dim]
        Returns:
            text_new: 更新后的文本原型 [B, num_cls, text_dim]
        """
        B = x_rgb_list[0].shape[0]
        num_cls = text.shape[0]
        
        # 对每个尺度分别计算Text更新增量
        scale_deltas = []
        scale_ir_features = []  # 用于动态权重
        
        for idx, (x_rgb, x_ir, module) in enumerate(zip(x_rgb_list, x_ir_list, self.scale_modules)):
            # Step A: IR语义锚点
            ir_pool = x_ir.mean(dim=[2, 3])
            u_ir = module['ir_to_text'](ir_pool)
            u_ir = F.normalize(u_ir, dim=-1)
            scale_ir_features.append(u_ir)
            
            logits = u_ir @ text.T / self.temperature
            w = F.softmax(logits, dim=-1)  # [B, num_cls]
            
            # Step B: IR-Guided CBAM
            x_rgb_enhanced = module['ir_guided_cbam'](x_rgb, x_ir)
            
            # Flatten to tokens
            H, W = x_rgb_enhanced.shape[2:]
            X_rgb = x_rgb_enhanced.flatten(2).permute(0, 2, 1)  # [B, N, C]
            
            # Step C: Text作为Query检索RGB
            Q = module['text_proj_q'](text)  # [num_cls, hidden_dim]
            K = module['rgb_proj_k'](X_rgb)  # [B, N, hidden_dim]
            V = module['rgb_proj_v'](X_rgb)  # [B, N, hidden_dim]
            
            Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, hidden_dim]
            
            A = torch.bmm(Q_expanded, K.transpose(-1, -2)) * self.scale
            A = F.softmax(A, dim=-1)  # [B, num_cls, N]
            
            Y_rgb = torch.bmm(A, V)  # [B, num_cls, hidden_dim]
            
            # Step D: 投影到text维度
            Y_aligned = module['align_mlp'](Y_rgb)  # [B, num_cls, text_dim]
            
            # 计算该尺度的增量（不加到text上，先保存）
            w_expanded = w.unsqueeze(-1)  # [B, num_cls, 1]
            delta = w_expanded * Y_aligned  # [B, num_cls, text_dim]
            scale_deltas.append(delta)
        
        # 融合多尺度增量
        if self.fusion_method == 'equal':
            # 等权重融合
            fused_delta = sum(scale_deltas) / self.num_levels
            
        elif self.fusion_method == 'learned_weight':
            # 可学习固定权重
            weights = F.softmax(self.scale_weights, dim=0)
            fused_delta = sum(w * delta for w, delta in zip(weights, scale_deltas))
            
        elif self.fusion_method == 'attention':
            # 动态权重（基于IR特征）
            ir_concat = torch.cat(scale_ir_features, dim=-1)  # [B, text_dim*3]
            attn_logits = self.scale_attention(ir_concat)  # [B, 3]
            attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 3]
            
            # 加权融合
            fused_delta = torch.zeros_like(scale_deltas[0])
            for i, delta in enumerate(scale_deltas):
                fused_delta += attn_weights[:, i:i+1, None] * delta
        
        # 最终更新Text
        text_expanded = text.unsqueeze(0).expand(B, -1, -1)
        text_new = text_expanded + self.gamma * fused_delta
        text_new = F.normalize(text_new, dim=-1)
        
        return text_new

