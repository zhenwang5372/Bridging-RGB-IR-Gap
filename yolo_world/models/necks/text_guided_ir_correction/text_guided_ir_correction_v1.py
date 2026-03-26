# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module - Final Version

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedIRCorrection(BaseModule):
    """
    Text-guided IR Correction Module (最终完善版)
    
    使用文本作为query计算RGB-IR一致性，并纠正IR特征中的错误。
    
    工作流程：
    1. Text-as-Query: 使用文本特征查询RGB和IR的注意力图
    2. 计算一致性：类别级别的RGB-IR响应一致性
    3. 计算差异图：空间级别的RGB-IR差异
    4. 生成错误图：基于差异图估计IR错误
    5. 纠正IR：减去估计的错误
    
    改进点：
    1. 每个尺度独立的纠错强度 alpha
    2. 可学习的相对纠错缩放因子
    3. 每个尺度可学习的 Softmax 温度
    4. 统一的 txt_feats 维度处理
    5. 数值稳定性改进
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认512
        num_classes (int): 类别数，默认4
        correction_alpha (float): 纠错强度初始值，默认0.3
        temperature (float): Softmax温度初始值，默认0.5
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        correction_alpha: float = 0.3,
        temperature: float = 0.5,
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
                SingleLevelTextGuidedCorrection(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    temperature=temperature  # 每个模块内部可学习
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
                - P3: [B, C_rgb[0], H/8, W/8]
                - P4: [B, C_rgb[1], H/16, W/16]
                - P5: [B, C_rgb[2], H/32, W/32]
            ir_feats: IR特征 (P3, P4, P5)
                - P3: [B, C_ir[0], H/8, W/8]
                - P4: [B, C_ir[1], H/16, W/16]
                - P5: [B, C_ir[2], H/32, W/32]
            txt_feats: 文本特征
                - 如果是tuple: (txt_feats, text_mask)，其中txt_feats是[B, num_classes, text_dim]
                - 如果是tensor: [B, num_classes, text_dim] 或 [num_classes, text_dim]
        
        Returns:
            rgb_feats: 原始RGB特征（不变）
            ir_corrected_feats: 纠正后的IR特征 (P3, P4, P5)
        """
        assert len(rgb_feats) == len(ir_feats) == self.num_levels, \
            f"Feature levels mismatch: RGB={len(rgb_feats)}, IR={len(ir_feats)}, expected={self.num_levels}"
        
        # ⭐ 统一处理txt_feats（只处理一次）
        # 处理tuple返回值
        if isinstance(txt_feats, tuple):
            txt_feats, text_mask = txt_feats
        else:
            text_mask = None
        
        # 处理维度：确保有batch维度
        if txt_feats.dim() == 2:  # [num_cls, text_dim]
            B = rgb_feats[0].size(0)
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, text_dim]
        elif txt_feats.size(0) == 1:  # [1, num_cls, text_dim]
            B = rgb_feats[0].size(0)
            if B > 1:
                txt_feats = txt_feats.expand(B, -1, -1)  # [B, num_cls, text_dim]
        
        # 现在 txt_feats 一定是 [B, num_cls, text_dim]
        ir_corrected_feats = []
        
        for i in range(self.num_levels):
            # ⭐ 直接传入，不需要再处理维度
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats,
                self.correction_alphas[i]
            )
            ir_corrected_feats.append(ir_corrected)
        
        return rgb_feats, tuple(ir_corrected_feats)


class SingleLevelTextGuidedCorrection(nn.Module):
    """
    单尺度的Text-guided IR Correction (最终完善版)
    
    核心机制：
    1. 使用文本特征作为query，分别查询RGB和IR特征
    2. 计算RGB-IR注意力图的一致性
    3. 基于一致性加权聚合差异图
    4. 估计IR特征的错误并进行纠正
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        temperature: float = 0.5
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        
        # ⭐ 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Q = Text @ W_q
        d_k = 128  # Key/Query维度
        self.text_query_proj = nn.Linear(text_dim, d_k)
        
        # K_rgb = RGB @ W_k
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        
        # K_ir = IR @ W_k
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        
        # Error Estimator（最后一层不用激活函数，让网络自己学习幅度）
        self.error_estimator = nn.Sequential(
            nn.Conv2d(ir_channels + 1, ir_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ir_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ir_channels, ir_channels, kernel_size=1),  # 让网络学习合适的幅度
        )
        
        # ⭐ 可学习的误差缩放因子
        self.error_scale = nn.Parameter(torch.tensor(0.5))
    
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
            txt_feats: [B, num_classes, text_dim] (已经在父类处理过维度)
            alpha: 纠错强度参数
        
        Returns:
            ir_corrected: [B, C_ir, H, W]
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        
        # ⭐ Step 0: 尺寸对齐检查（参考 MultiLevelRGBIRFusion）
        # 如果 IR 和 RGB 尺寸不同，将 IR 插值到 RGB 尺寸
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(
                x_ir,
                size=(H, W),             # 对齐到 RGB 的尺寸
                mode='bilinear',         # 双线性插值
                align_corners=False      # PyTorch 推荐
            )
            # 现在 x_ir: [B, C_ir, H, W] ✅
        
        # ═══ Step 1: Text-as-Query 得到两路注意力 ═══
        # Q = Text @ W_q  [B, num_cls, d_k]
        Q = self.text_query_proj(txt_feats)  # [B, num_cls, d_k]
        
        # K_rgb = RGB @ W_k  [B, d_k, H, W]
        K_rgb = self.rgb_key_proj(x_rgb)  # [B, d_k, H, W]
        K_rgb_flat = K_rgb.view(B, K_rgb.size(1), H * W)  # [B, d_k, H*W]
        
        # K_ir = IR @ W_k  [B, d_k, H, W]
        K_ir = self.ir_key_proj(x_ir)  # [B, d_k, H, W]
        K_ir_flat = K_ir.view(B, K_ir.size(1), H * W)  # [B, d_k, H*W]
        
        # A_rgb = Softmax(Q @ K_rgb.T)  [B, num_cls, H*W]
        attn_logits_rgb = torch.bmm(Q, K_rgb_flat)  # [B, num_cls, H*W]
        A_rgb = F.softmax(attn_logits_rgb / (Q.size(-1) ** 0.5), dim=-1)
        
        # A_ir = Softmax(Q @ K_ir.T)  [B, num_cls, H*W]
        attn_logits_ir = torch.bmm(Q, K_ir_flat)  # [B, num_cls, H*W]
        A_ir = F.softmax(attn_logits_ir / (Q.size(-1) ** 0.5), dim=-1)
        
        # ═══ Step 2a: 计算一致性（类别级别）═══
        # G_c = cosine(A_rgb[:, c, :], A_ir[:, c, :])  [B, num_cls]
        # 归一化
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)  # [B, num_cls, H*W]
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)   # [B, num_cls, H*W]
        
        # 余弦相似度（按类别）
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # [B, num_cls]
        
        # ⭐ 改进：将 G 限制在 [0, 1]，负相关视为不一致
        G = torch.clamp(G, 0.0, 1.0)
        
        # ═══ Step 2b: 计算差异图（空间级别）═══
        # Diff_c = |A_rgb[:, c, :] - A_ir[:, c, :]|  [B, num_cls, H*W]
        Diff = torch.abs(A_rgb - A_ir)  # [B, num_cls, H*W]
        
        # ═══ Step 3: 聚合差异图（用一致性加权）═══
        # 一致性低的类别，其差异图更重要
        # w = Softmax((1 - G) / temperature)
        disagreement = 1.0 - G  # [B, num_cls]，现在在 [0, 1] 范围内
        
        # ⭐ 使用可学习的temperature，避免权重过度集中
        w = F.softmax(disagreement / self.temperature, dim=1)  # [B, num_cls]
        
        # Diff_map = Σ_c (w[:, c] × Diff_c[:, c, :])  [B, H*W]
        Diff_map = torch.sum(w.unsqueeze(-1) * Diff, dim=1)  # [B, H*W]
        
        # ═══ Step 4: 生成错误图 ═══
        # Reshape到空间维度
        Diff_spatial = Diff_map.view(B, 1, H, W)  # [B, 1, H, W]
        
        # 拼接IR特征和差异图
        concat = torch.cat([x_ir, Diff_spatial], dim=1)  # [B, C_ir+1, H, W]
        
        # 通过ErrorEstimator生成错误图
        error_map = self.error_estimator(concat)  # [B, C_ir, H, W]
        
        # ═══ Step 5: 纠错 - 相对纠错 + 可学习缩放 ═══
        # 计算IR特征的标准差（通道级别，空间维度）
        ir_std = x_ir.std(dim=[2, 3], keepdim=True) + 1e-6  # [B, C_ir, 1, 1]
        
        # ⭐ 使用可学习的缩放因子，将error_map缩放到IR特征的幅度量级
        error_map_scaled = error_map * ir_std * self.error_scale
        
        # 纠错：IR_corrected = IR - alpha × error_map
        ir_corrected = x_ir - alpha * error_map_scaled
        
        return ir_corrected


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithCorrection(BaseModule):
    """
    带IR纠错的双流多模态YOLO Backbone (最终完善版)
    
    在MultiLevelRGBIRFusion之前先进行Text-guided IR Correction
    
    工作流程：
    1. RGB Backbone: 提取RGB特征
    2. IR Backbone: 提取IR特征
    3. Text Model: 提取文本特征
    4. IR Correction: 使用文本指导纠正IR特征（新增）
    5. RGB-IR Fusion: 融合纠正后的IR和RGB特征
    
    Args:
        image_model (dict): RGB backbone配置
        ir_model (dict): IR backbone配置
        fusion_module (dict): RGB-IR融合模块配置
        text_model (dict): 文本模型配置
        ir_correction (dict): IR纠错模块配置（新增）
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
        
        # Build IR correction module (新增)
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
        Forward pass through dual-stream multi-modal backbone with IR correction.
        
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
        
        # ⭐ Step 0: Text-guided IR Correction (如果启用)
        if self.with_ir_correction and txt_feats is not None:
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
        # Step 1: Fuse RGB and corrected IR features
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
        
        ⭐ 修复：添加完整的检查逻辑，与forward保持一致
        
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
        
        # ⭐ 修复：添加完整的条件检查
        if self.with_ir_correction and text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats)
        
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return img_feats