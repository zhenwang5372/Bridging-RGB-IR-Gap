# Copyright (c) Tencent Inc. All rights reserved.
# IR-Only Unsupported by RGB Module (G-weighted version)
# 
# 核心公式: M_err = A_ir * (1 - A_rgb)
# 特点:
# - 使用 sigmoid 激活（而非 softmax）
# - 保留 G 一致性度量，用 disagreement 加权聚合 N 维度
# - A_rgb ≈ 1 表示 RGB 支持该位置，(1 - A_rgb) ≈ 0，不需要纠正
# - A_rgb ≈ 0 表示 RGB 不支持，(1 - A_rgb) ≈ 1，需要纠正
# - A_ir 高表示 IR 认为该位置重要

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


@MODELS.register_module()
class IR_Only_Unsupported_by_RGB_G(BaseModule):
    """
    IR-Only Unsupported by RGB Module (G-weighted version)
    
    核心改动：
    - M_err = A_ir * (1 - A_rgb)
    - 使用 sigmoid 激活
    - 保留 G 一致性度量，用 disagreement 加权
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认512
        num_classes (int): 类别数，默认4
        correction_alpha (float): 纠错强度初始值，默认 -0.3
        d_k (int): Attention 的 key 维度，默认 128
        log_alpha (bool): 是否打印 Alpha 值到日志，默认 True
        log_interval (int): 训练时打印 Alpha 的间隔（iteration 数），默认 50
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        correction_alpha: float = -0.3,
        d_k: int = 128,
        log_alpha: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        self.d_k = d_k
        self.log_alpha = log_alpha
        self.log_interval = log_interval
        
        # 训练时的 iteration 计数器
        self._train_iter_count = 0
        # 用于控制验证时每个 epoch 只打印一次
        self._alpha_logged_this_epoch = False
        
        # 每个尺度独立的 alpha（初始值 -0.3）
        self.correction_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(correction_alpha))
            for _ in range(self.num_levels)
        ])
        
        # 为每个尺度创建纠错模块
        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelIROnlyUnsupportedByRGB_G(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    d_k=d_k,
                )
            )
    
    def get_alpha_values(self):
        """获取当前的 Alpha 值"""
        alphas = {
            'P3': self.correction_alphas[0].item(),
            'P4': self.correction_alphas[1].item(),
            'P5': self.correction_alphas[2].item(),
        }
        alphas['mean'] = sum(alphas.values()) / 3
        return alphas
    
    def log_alpha_values(self, stage: str = 'val', iter_num: int = None):
        """
        打印 Alpha 值到日志
        
        Args:
            stage: 'train' 或 'val'
            iter_num: 当前 iteration 数（仅 train 时使用）
        """
        if not self.log_alpha:
            return
        
        # 验证阶段：每个 epoch 只打印一次
        if stage == 'val' and self._alpha_logged_this_epoch:
            return
        
        alphas = self.get_alpha_values()
        logger = MMLogger.get_current_instance()
        
        if stage == 'train' and iter_num is not None:
            logger.info(
                f"[IR-Only Unsupported by RGB (G)] Iter [{iter_num}] Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}"
            )
        else:
            logger.info(
                f"[IR-Only Unsupported by RGB (G)] Val Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}"
            )
        
        if stage == 'val':
            self._alpha_logged_this_epoch = True
    
    def reset_alpha_log_flag(self):
        """重置 Alpha 打印标志（在新 epoch 开始时调用）"""
        self._alpha_logged_this_epoch = False
        self._train_iter_count = 0
    
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
        
        # 打印 Alpha 值
        if self.training:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self.log_alpha_values(stage='train', iter_num=self._train_iter_count)
        else:
            self.log_alpha_values(stage='val')
        
        # 统一处理txt_feats
        if isinstance(txt_feats, tuple):
            txt_feats, text_mask = txt_feats
        else:
            text_mask = None
        
        # 处理维度
        B = rgb_feats[0].size(0)
        
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)
        elif txt_feats.dim() == 3:
            B_txt = txt_feats.size(0)
            if B_txt != B:
                if B_txt == 1:
                    txt_feats = txt_feats.expand(B, -1, -1)
                else:
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


class SingleLevelIROnlyUnsupportedByRGB_G(nn.Module):
    """
    单尺度的 IR-Only Unsupported by RGB (G-weighted version)
    
    核心改动：
    - 使用 sigmoid 激活（而非 softmax）
    - M_err = A_ir * (1 - A_rgb)
    - 保留 G 一致性度量，用 disagreement 加权聚合
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        d_k: int = 128,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.d_k = d_k
        
        # Query/Key 投影
        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        
        # Error Estimator
        self.error_estimator = nn.Sequential(
            nn.Conv2d(ir_channels, ir_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=1),
        )
    
    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        txt_feats: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        核心公式: M_err = A_ir * (1 - A_rgb)
        使用 sigmoid 激活，保留 G 一致性度量加权
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N_actual = txt_feats.size(1)
        
        # 尺寸对齐
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        # Step 1: 语义激活（使用 sigmoid）
        Q = self.text_query_proj(txt_feats)  # (B, N, d_k)
        K_rgb = self.rgb_key_proj(x_rgb)  # (B, d_k, H, W)
        K_rgb_flat = K_rgb.view(B, K_rgb.size(1), H * W)  # (B, d_k, H*W)
        K_ir = self.ir_key_proj(x_ir)  # (B, d_k, H, W)
        K_ir_flat = K_ir.view(B, K_ir.size(1), H * W)  # (B, d_k, H*W)
        
        d_k_sqrt = Q.size(-1) ** 0.5
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat)  # (B, N, H*W)
        A_rgb = torch.sigmoid(attn_logits_rgb / d_k_sqrt)  # (B, N, H*W), 值在 [0, 1]
        attn_logits_ir = torch.bmm(Q, K_ir_flat)  # (B, N, H*W)
        A_ir = torch.sigmoid(attn_logits_ir / d_k_sqrt)  # (B, N, H*W), 值在 [0, 1]
        
        # Step 2: 一致性度量（保留）
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # (B, N)
        G = torch.clamp(G, 0.0, 1.0)
        
        # Step 3: IR-Only Unsupported by RGB
        # M_err = A_ir * (1 - A_rgb)
        # A_rgb ≈ 1 → RGB 支持 → (1 - A_rgb) ≈ 0 → 不纠正
        # A_rgb ≈ 0 → RGB 不支持 → (1 - A_rgb) ≈ 1 → 纠正
        # A_ir 高 → IR 认为重要 → M_err 高
        M_err_per_class = A_ir * (1 - A_rgb)  # (B, N, H*W)
        
        # 使用 disagreement 加权聚合 N 维度
        disagreement = 1.0 - G  # (B, N)
        M_err = torch.einsum('bn,bnh->bh', disagreement, M_err_per_class)  # (B, H*W)
        
        # 归一化
        M_err_min = M_err.min(dim=-1, keepdim=True)[0]
        M_err_max = M_err.max(dim=-1, keepdim=True)[0]
        M_err_range = torch.clamp(M_err_max - M_err_min, min=1e-6)
        M_err = (M_err - M_err_min) / M_err_range
        M_err_spatial = M_err.view(B, 1, H, W)
        
        # Step 4: 空间门控
        F_extracted = x_ir * M_err_spatial
        Error_map = self.error_estimator(F_extracted)
        
        # Step 5: 特征纠正
        # alpha 初始值为 -0.3
        # X_ir - (-0.3) × Error_map = X_ir + 0.3 × Error_map（增强效果）
        ir_corrected = x_ir - alpha * Error_map
        
        return ir_corrected


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithIROnlyUnsupportedByRGB_G(BaseModule):
    """
    带 IR-Only Unsupported by RGB (G-weighted) 纠错的双流多模态 YOLO Backbone
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
        
        # Build IR correction module
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
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        
        if mode and self.with_ir_correction:
            self.ir_correction.reset_alpha_log_flag()
    
    def forward(
        self,
        image: torch.Tensor,
        text: List[List[str]],
        img_ir: torch.Tensor = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass."""
        if img_ir is None:
            img_ir = image
        
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
        else:
            txt_feats = None
        
        if self.with_ir_correction and txt_feats is not None:
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
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
        """Forward image only."""
        if img_ir is None:
            img_ir = image
        
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        
        if self.with_ir_correction and text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return img_feats
