# Copyright (c) Tencent Inc. All rights reserved.
# Text Update with Multi-scale Fusion First
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from ..trimodal_utils import IRGuidedCBAM


class TextUpdateFusionFirst(BaseModule):
    """先融合多尺度特征，再更新Text
    
    核心思路：
        1. 将P3/P4/P5的RGB和IR特征融合成统一的多尺度表示
        2. 用融合后的特征执行一次Text更新
        3. 计算量约为多次更新的1/3
    
    支持多种融合方法：
        - 'fpn': 类似FPN的top-down融合
        - 'concat': 直接拼接后降维
        - 'attention': 注意力加权融合
        - 'deformable': 可变形注意力融合
    """
    
    def __init__(self,
                 in_channels: list = [128, 256, 512],
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.07,
                 gamma: float = 0.1,
                 cbam_reduction: int = 16,
                 fusion_method: str = 'fpn',  # 'fpn', 'concat', 'attention', 'deformable'
                 target_size: int = 40,  # 融合后的特征图大小 (对应P4: 40x40)
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.fusion_method = fusion_method
        self.target_size = target_size
        
        # 统一通道数（融合到中间尺度的通道）
        self.unified_channels = in_channels[1]  # 256
        
        # IR语义锚点（使用P4）
        self.ir_to_text = nn.Linear(in_channels[1], text_dim)
        
        # 每个尺度的IR-Guided CBAM
        self.ir_guided_cbam = nn.ModuleList([
            IRGuidedCBAM(channels=ch, reduction=cbam_reduction, kernel_size=7)
            for ch in in_channels
        ])
        
        # 多尺度融合模块
        if fusion_method == 'fpn':
            self._build_fpn_fusion()
        elif fusion_method == 'concat':
            self._build_concat_fusion()
        elif fusion_method == 'attention':
            self._build_attention_fusion()
        elif fusion_method == 'deformable':
            self._build_deformable_fusion()
        
        # Text更新的投影层（只需要一套）
        self.text_proj_q = nn.Linear(text_dim, hidden_dim)
        self.rgb_proj_k = nn.Linear(self.unified_channels, hidden_dim)
        self.rgb_proj_v = nn.Linear(self.unified_channels, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        
        self.align_mlp = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, text_dim)
        )
    
    def _build_fpn_fusion(self):
        """FPN风格：top-down + lateral连接"""
        # 通道对齐
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, self.unified_channels, 1) for ch in self.in_channels
        ])
        # 融合后的平滑卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.unified_channels, self.unified_channels, 3, padding=1)
            for _ in range(self.num_levels)
        ])
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.unified_channels * self.num_levels, self.unified_channels, 1),
            nn.BatchNorm2d(self.unified_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_concat_fusion(self):
        """直接拼接：上采样到统一尺寸后concat"""
        total_channels = sum(self.in_channels)
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(total_channels, self.unified_channels * 2, 3, padding=1),
            nn.BatchNorm2d(self.unified_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.unified_channels * 2, self.unified_channels, 1),
            nn.BatchNorm2d(self.unified_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_attention_fusion(self):
        """注意力加权融合"""
        # 每个尺度先对齐通道
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(ch, self.unified_channels, 1) for ch in self.in_channels
        ])
        # 注意力权重生成
        self.attention_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.unified_channels * self.num_levels, self.num_levels, 1),
            nn.Softmax(dim=1)
        )
    
    def _build_deformable_fusion(self):
        """可变形注意力融合（简化版）"""
        # 通道对齐
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(ch, self.unified_channels, 1) for ch in self.in_channels
        ])
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.unified_channels, self.target_size, self.target_size))
        # 跨尺度注意力
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=self.unified_channels,
            num_heads=8,
            batch_first=True
        )
    
    def _fuse_fpn(self, feats_list):
        """FPN融合"""
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats_list)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )
        
        # Smooth
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        # 上采样到统一尺寸并拼接
        outs_resized = [
            F.interpolate(out, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            for out in outs
        ]
        fused = torch.cat(outs_resized, dim=1)
        fused = self.final_fusion(fused)
        
        return fused
    
    def _fuse_concat(self, feats_list):
        """直接拼接融合"""
        # 上采样到统一尺寸
        feats_resized = [
            F.interpolate(feat, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            for feat in feats_list
        ]
        # 拼接
        fused = torch.cat(feats_resized, dim=1)
        fused = self.concat_fusion(fused)
        return fused
    
    def _fuse_attention(self, feats_list):
        """注意力加权融合"""
        # 通道对齐
        feats_aligned = [conv(feat) for conv, feat in zip(self.scale_convs, feats_list)]
        
        # 上采样到统一尺寸
        feats_resized = [
            F.interpolate(feat, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            for feat in feats_aligned
        ]
        
        # 拼接用于计算注意力权重
        feats_concat = torch.cat(feats_resized, dim=1)
        weights = self.attention_weights(feats_concat)  # [B, 3, 1, 1]
        
        # 加权融合
        fused = sum(w * feat for w, feat in zip(weights.split(1, dim=1), feats_resized))
        return fused
    
    def _fuse_deformable(self, feats_list):
        """可变形注意力融合"""
        B = feats_list[0].shape[0]
        
        # 通道对齐并上采样
        feats_aligned = [conv(feat) for conv, feat in zip(self.scale_convs, feats_list)]
        feats_resized = [
            F.interpolate(feat, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            for feat in feats_aligned
        ]
        
        # 展平为序列 [B, N, C]
        feats_flat = [feat.flatten(2).permute(0, 2, 1) for feat in feats_resized]
        feats_concat = torch.cat(feats_flat, dim=1)  # [B, N*3, C]
        
        # 添加位置编码
        pos = self.pos_embed.flatten(2).permute(0, 2, 1).expand(B, -1, -1)  # [B, N, C]
        
        # 跨尺度注意力
        fused_flat, _ = self.cross_scale_attn(pos, feats_concat, feats_concat)
        
        # 恢复空间维度
        fused = fused_flat.permute(0, 2, 1).view(B, self.unified_channels, self.target_size, self.target_size)
        return fused
    
    def forward(self,
                x_rgb_list: list,
                x_ir_list: list,
                text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb_list: RGB特征列表 [P3, P4, P5]
            x_ir_list: IR特征列表 [P3, P4, P5]
            text: 文本原型 [num_cls, text_dim]
        Returns:
            text_new: 更新后的文本原型 [B, num_cls, text_dim]
        """
        B = x_rgb_list[0].shape[0]
        num_cls = text.shape[0]
        
        # Step A: IR语义锚点（使用P4）
        ir_pool = x_ir_list[1].mean(dim=[2, 3])
        u_ir = self.ir_to_text(ir_pool)
        u_ir = F.normalize(u_ir, dim=-1)
        
        logits = u_ir @ text.T / self.temperature
        w = F.softmax(logits, dim=-1)  # [B, num_cls]
        
        # Step B: IR-Guided CBAM处理所有尺度
        x_rgb_enhanced = [
            cbam(rgb, ir) 
            for cbam, rgb, ir in zip(self.ir_guided_cbam, x_rgb_list, x_ir_list)
        ]
        
        # Step C: 多尺度融合
        if self.fusion_method == 'fpn':
            x_fused = self._fuse_fpn(x_rgb_enhanced)
        elif self.fusion_method == 'concat':
            x_fused = self._fuse_concat(x_rgb_enhanced)
        elif self.fusion_method == 'attention':
            x_fused = self._fuse_attention(x_rgb_enhanced)
        elif self.fusion_method == 'deformable':
            x_fused = self._fuse_deformable(x_rgb_enhanced)
        
        # Step D: Text作为Query检索融合后的RGB特征
        X_rgb = x_fused.flatten(2).permute(0, 2, 1)  # [B, N, unified_channels]
        
        Q = self.text_proj_q(text)  # [num_cls, hidden_dim]
        K = self.rgb_proj_k(X_rgb)  # [B, N, hidden_dim]
        V = self.rgb_proj_v(X_rgb)  # [B, N, hidden_dim]
        
        Q_expanded = Q.unsqueeze(0).expand(B, -1, -1)
        
        A = torch.bmm(Q_expanded, K.transpose(-1, -2)) * self.scale
        A = F.softmax(A, dim=-1)
        
        Y_rgb = torch.bmm(A, V)  # [B, num_cls, hidden_dim]
        
        # Step E: 投影并更新Text
        Y_aligned = self.align_mlp(Y_rgb)
        
        w_expanded = w.unsqueeze(-1)
        delta = self.gamma * w_expanded * Y_aligned
        
        text_expanded = text.unsqueeze(0).expand(B, -1, -1)
        text_new = text_expanded + delta
        text_new = F.normalize(text_new, dim=-1)
        
        return text_new

