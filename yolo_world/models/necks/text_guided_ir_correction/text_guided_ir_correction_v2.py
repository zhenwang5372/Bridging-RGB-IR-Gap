# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module V2 (基于 ir_correction.markdown 方案)
# 
# 核心改进：
# 1. 空间门控机制 (Spatial Gating): 用 M_err 作为乘法 mask 筛选错误区域
# 2. Min-Max 归一化: 替代 Softmax 加权，保持更好的数值分布
# 3. 更强的物理可解释性: M_err ⊙ X_ir 直接提取错误特征

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedIRCorrectionV2(BaseModule):
    """
    Text-guided IR Correction Module V2 (基于 ir_correction.markdown)
    
    使用文本作为锚点，衡量 RGB 和 IR 特征图在语义空间的一致性，
    并通过空间门控机制纠正IR特征中的错误。
    
    工作流程：
    1. Text-as-Query: 使用文本特征查询RGB和IR的注意力图 A_rgb, A_ir
    2. 计算一致性: 类别级别的 RGB-IR 响应一致性 G
    3. 加权差异图: (1-G) 加权聚合，生成 M_err
    4. 空间门控: F_extracted = X_ir ⊙ M_err，提取错误特征
    5. 纠正IR: X_ir^corrected = X_ir - α × f_conv(F_extracted)
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认512
        num_classes (int): 类别数，默认4
        correction_alpha (float): 纠错强度初始值，默认0.3
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        correction_alpha: float = 0.3,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        
        # 每个尺度独立的 alpha
        self.correction_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(correction_alpha))
            for _ in range(self.num_levels)
        ])
        
        # 为每个尺度创建纠错模块
        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelTextGuidedCorrectionV2(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                )
            )
    
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
            ir_corrected_feats: 纠正后的IR特征 (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Feature levels mismatch: RGB={len(rgb_feats)}, IR={len(ir_feats)}, expected={self.num_levels}"
        
        # 统一处理txt_feats（只处理一次）
        if isinstance(txt_feats, tuple):
            txt_feats, text_mask = txt_feats
        else:
            text_mask = None
        
        # 处理维度：确保有batch维度且与 rgb_feats 的batch一致
        B = rgb_feats[0].size(0)
        
        if txt_feats.dim() == 2:  # [num_cls, text_dim]
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, text_dim]
        elif txt_feats.dim() == 3:  # [B', num_cls, text_dim]
            B_txt = txt_feats.size(0)
            if B_txt != B:
                # batch size 不匹配，需要调整（通常发生在验证时 B_txt=1 但 B 可能>1）
                if B_txt == 1:
                    txt_feats = txt_feats.expand(B, -1, -1)  # [B, num_cls, text_dim]
                else:
                    # 如果 B_txt > B，取前B个（不应该发生，但做防御性处理）
                    txt_feats = txt_feats[:B]
        
        ir_corrected_feats = []
        
        for i in range(self.num_levels):
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats,
                self.correction_alphas[i]
            )
            ir_corrected_feats.append(ir_corrected)
        
        return rgb_feats, tuple(ir_corrected_feats)


class SingleLevelTextGuidedCorrectionV2(nn.Module):
    """
    单尺度的 Text-guided IR Correction V2 (基于 ir_correction.markdown)
    
    核心改进：
    1. 空间门控机制 (Spatial Gating): M_err 作为乘法 mask
    2. Min-Max 归一化: 替代 Softmax 加权
    3. 更强的物理可解释性
    
    注意：num_classes 仅用于配置参考，实际运行时会动态适应 txt_feats 的类别数
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes  # 仅作为参考，实际会动态适应
        
        # ═══ Step 1: Query/Key 投影 ═══
        # Q = Text @ W_q  [N, d_k]
        d_k = 128  # Key/Query维度
        self.text_query_proj = nn.Linear(text_dim, d_k)
        
        # K_rgb = φ(X_rgb)  [B, d_k, H*W]
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        
        # K_ir = φ(X_ir)  [B, d_k, H*W]
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        
        # ═══ Step 4: Error Estimator (f_conv) ═══
        # 轻量级卷积网络，将门控后的特征转化为纠正量
        # f_conv: 1×1 Conv + 3×3 Conv (上下文平滑)
        self.error_estimator = nn.Sequential(
            nn.Conv2d(ir_channels, ir_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=1),  # 输出纠正量
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
            x_rgb: [B, C_rgb, H, W]
            x_ir: [B, C_ir, H, W]
            txt_feats: [B, N, text_dim] (N是实际的类别数，可能包含负样本)
            alpha: 纠错强度参数
        
        Returns:
            ir_corrected: [B, C_ir, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N_actual = txt_feats.size(1)  # 实际的类别数（动态）
        
        # ═══ Step 0: 尺寸对齐检查 ═══
        # 如果 IR 和 RGB 尺寸不同，将 IR 插值到 RGB 尺寸
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(
                x_ir,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: 文本引导的语义激活 (Semantic Activation)
        # ═══════════════════════════════════════════════════════════════════
        # Q_text = Text_emb @ W_Q  [B, N, d_k]
        Q = self.text_query_proj(txt_feats)  # [B, num_cls, d_k]
        
        # K_rgb = φ(X_rgb)  [B, d_k, H*W]
        K_rgb = self.rgb_key_proj(x_rgb)  # [B, d_k, H, W]
        K_rgb_flat = K_rgb.view(B, K_rgb.size(1), H * W)  # [B, d_k, H*W]
        
        # K_ir = φ(X_ir)  [B, d_k, H*W]
        K_ir = self.ir_key_proj(x_ir)  # [B, d_k, H, W]
        K_ir_flat = K_ir.view(B, K_ir.size(1), H * W)  # [B, d_k, H*W]
        
        # A_rgb = Softmax(Q @ K_rgb^T / sqrt(d_k))  [B, N, H*W]
        d_k_sqrt = Q.size(-1) ** 0.5
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat)  # [B, N, H*W]
        A_rgb = F.softmax(attn_logits_rgb / d_k_sqrt, dim=-1)  # [B, N, H*W]
        
        # A_ir = Softmax(Q @ K_ir^T / sqrt(d_k))  [B, N, H*W]
        attn_logits_ir = torch.bmm(Q, K_ir_flat)  # [B, N, H*W]
        A_ir = F.softmax(attn_logits_ir / d_k_sqrt, dim=-1)  # [B, N, H*W]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: 语义一致性度量 (Semantic Consistency)
        # ═══════════════════════════════════════════════════════════════════
        # G_c = cosine(A_rgb[:, c, :], A_ir[:, c, :])  [B, N]
        # 归一化
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)  # [B, N, H*W]
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)    # [B, N, H*W]
        
        # 余弦相似度（按类别）
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # [B, N]
        
        # 将 G 限制在 [0, 1]，负相关视为不一致
        G = torch.clamp(G, 0.0, 1.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 3: 加权差异图生成 (Weighted Difference Aggregation)
        # ═══════════════════════════════════════════════════════════════════
        # D_spatial^c = |A_rgb[:, c, :] - A_ir[:, c, :]|  [B, N, H*W]
        D_spatial = torch.abs(A_rgb - A_ir)  # [B, N, H*W]
        
        # ⭐ 关键改进：使用 (1-G) 直接加权，而不是 Softmax
        # disagreement = 1 - G  [B, N]
        disagreement = 1.0 - G  # [B, N]
        
        # M_err = Σ_c (1 - G_c) × D_spatial^c  [B, H*W]
        # 使用 einsum 实现加权求和: [B, N] × [B, N, H*W] -> [B, H*W]
        M_err = torch.einsum('bn,bnh->bh', disagreement, D_spatial)  # [B, H*W]
        
        # ⭐ Min-Max 归一化 (替代 Softmax 加权)
        # M_err = (M_err - min) / (max - min + ε)
        M_err_min = M_err.min(dim=-1, keepdim=True)[0]  # [B, 1]
        M_err_max = M_err.max(dim=-1, keepdim=True)[0]  # [B, 1]
        # 增强数值稳定性：避免除零
        M_err_range = M_err_max - M_err_min
        M_err_range = torch.clamp(M_err_range, min=1e-6)  # 确保至少是 1e-6
        M_err = (M_err - M_err_min) / M_err_range  # [B, H*W]
        
        # Reshape 到空间维度 [B, 1, H, W]
        M_err_spatial = M_err.view(B, 1, H, W)  # [B, 1, H, W]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 4: 错误特征提取 (Error Feature Extraction) - 空间门控机制
        # ═══════════════════════════════════════════════════════════════════
        # ⭐ 核心改进：使用 M_err 作为空间选择器（乘法 mask）
        # F_extracted = X_ir ⊙ M_err  [B, C_ir, H, W]
        F_extracted = x_ir * M_err_spatial  # 广播乘法
        
        # 通过轻量级卷积网络变换为纠正量
        # Error_map = f_conv(F_extracted)  [B, C_ir, H, W]
        Error_map = self.error_estimator(F_extracted)  # [B, C_ir, H, W]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 5: 特征纠正 (Feature Rectification)
        # ═══════════════════════════════════════════════════════════════════
        # X_ir^corrected = X_ir - α × Error_map
        ir_corrected = x_ir - alpha * Error_map
        
        return ir_corrected


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithCorrectionV2(BaseModule):
    """
    带IR纠错的双流多模态YOLO Backbone V2 (基于 ir_correction.markdown)
    
    在MultiLevelRGBIRFusion之前先进行Text-guided IR Correction
    
    工作流程：
    1. RGB Backbone: 提取RGB特征
    2. IR Backbone: 提取IR特征
    3. Text Model: 提取文本特征
    4. IR Correction V2: 使用空间门控机制纠正IR特征（核心改进）
    5. RGB-IR Fusion: 融合纠正后的IR和RGB特征
    
    Args:
        image_model (dict): RGB backbone配置
        ir_model (dict): IR backbone配置
        fusion_module (dict): RGB-IR融合模块配置
        text_model (dict): 文本模型配置
        ir_correction (dict): IR纠错模块配置 V2
        frozen_stages (int): 冻结阶段数
        with_text_model (bool): 是否使用文本模型
        init_cfg (dict): 初始化配置
    """
    
    def __init__(
        self,
        image_model: dict,
        ir_model: dict,
        fusion_module: dict,
        text_model: dict = None,
        ir_correction: dict = None,
        frozen_stages: int = -1,
        with_text_model: bool = True,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # Build RGB image backbone
        self.image_model = MODELS.build(image_model)
        
        # Build IR backbone
        self.ir_model = MODELS.build(ir_model)
        
        # Build IR correction module V2
        if ir_correction is not None:
            self.ir_correction = MODELS.build(ir_correction)
            self.with_ir_correction = True
        else:
            self.ir_correction = None
            self.with_ir_correction = False
        
        # Build RGB-IR fusion module
        self.fusion_module = MODELS.build(fusion_module)
        
        # Build text model
        if self.with_text_model and text_model is not None:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze the parameters of specified stages."""
        if self.frozen_stages >= 0:
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen stages."""
        super().train(mode)
        self._freeze_stages()
    
    def forward(
        self,
        image: torch.Tensor,
        text: List[List[str]],
        img_ir: torch.Tensor = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through dual-stream multi-modal backbone with IR correction V2.
        
        Args:
            image: RGB input [B, 3, H, W]
            text: List of text prompts
            img_ir: IR input [B, 3, H, W]. If None, uses image as IR input.
        
        Returns:
            img_feats: 融合后的图像特征 (P3, P4, P5)
            txt_feats: 文本特征 (可能是tensor或tuple)
        """
        if img_ir is None:
            img_ir = image
        
        # Extract RGB features
        rgb_feats = self.image_model(image)
        
        # Extract IR features
        ir_feats = self.ir_model(img_ir)
        
        # Extract text features
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
        else:
            txt_feats = None
        
        # ⭐ Text-guided IR Correction V2 (空间门控机制)
        if self.with_ir_correction and txt_feats is not None:
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
        # Fuse RGB and corrected IR features
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return img_feats, txt_feats
    
    def forward_text(self, text: List[List[str]]):
        """Forward text only."""
        assert self.with_text_model and self.text_model is not None, \
            "forward_text() requires a text model"
        return self.text_model(text)
    
    def forward_image(
        self, 
        image: torch.Tensor, 
        img_ir: torch.Tensor = None,
        text: List[List[str]] = None
    ):
        """
        Forward image only (with optional IR correction).
        
        Args:
            image: RGB input [B, 3, H, W]
            img_ir: IR input [B, 3, H, W]
            text: Text prompts (needed for IR correction)
        
        Returns:
            img_feats: 融合后的图像特征 (P3, P4, P5)
        """
        if img_ir is None:
            img_ir = image
        
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        
        # IR Correction V2 需要文本特征
        if self.with_ir_correction and text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return img_feats

