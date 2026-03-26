# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module V5 Plus
# 
# 基于 V5，核心改动：
# - 多尺度动态投影维度 embed_channels（参考 YOLO-World 的设计）
#   - P3: embed_channels[0] (如 128)
#   - P4: embed_channels[1] (如 256)
#   - P5: embed_channels[2] (如 512)
# - 保持 V5 的余弦相似度机制
# - Alpha 初始值保持 -0.3
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedIRCorrectionV5Plus(BaseModule):
    """
    Text-guided IR Correction Module V5 Plus
    
    基于 V5，核心改动：
    - 多尺度动态投影维度 embed_channels（参考 YOLO-World 的设计）
    - 每个尺度使用独立的投影维度，更好地适应不同层级的特征
    
    理论改进：
        V5: 所有尺度使用固定 d_k = 128
        V5Plus: P3 → embed_channels[0], P4 → embed_channels[1], P5 → embed_channels[2]
        
        参考 YOLO-World 的 MaxSigmoidAttnBlock:
            - guide_fc = Linear(guide_channels, embed_channels)
            - embed_conv = Conv2d(in_channels, embed_channels, 1)
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]，如 [128, 256, 512]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]，如 [128, 256, 512]
        text_dim (int): 文本特征维度，默认512
        embed_channels (List[int]): 各尺度的投影维度 [P3, P4, P5]，如 [128, 256, 512]
        num_classes (int): 类别数，默认4
        correction_alpha (float): 纠错强度初始值，默认 -0.3
        cosine_scale (float): 余弦相似度缩放因子，默认 10.0
        log_alpha (bool): 是否打印 Alpha 值到日志，默认 True
        log_interval (int): 训练时打印 Alpha 的间隔（iteration 数），默认 50
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        embed_channels: List[int] = [128, 256, 512],  # ⭐ V5Plus 新增：多尺度投影维度
        num_classes: int = 4,
        correction_alpha: float = -0.3,
        cosine_scale: float = 10.0,
        log_alpha: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.embed_channels = embed_channels
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        self.cosine_scale = cosine_scale
        self.log_alpha = log_alpha
        self.log_interval = log_interval
        
        # 验证 embed_channels 长度
        assert len(embed_channels) == self.num_levels, \
            f"embed_channels length ({len(embed_channels)}) must match num_levels ({self.num_levels})"
        
        # 训练时的 iteration 计数器
        self._train_iter_count = 0
        # 用于控制验证时每个 epoch 只打印一次
        self._alpha_logged_this_epoch = False
        
        # 设置随机种子以确保可复现性
        torch.manual_seed(3407)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(3407)
            torch.cuda.manual_seed_all(3407)
        
        # 每个尺度独立的 alpha（初始值 -0.3）
        self.correction_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(correction_alpha))
            for _ in range(self.num_levels)
        ])
        
        # ⭐ V5Plus: 为每个尺度创建纠错模块（使用不同的 embed_channels）
        self.correction_modules = nn.ModuleList()
        for i, (rgb_ch, ir_ch) in enumerate(zip(rgb_channels, ir_channels)):
            self.correction_modules.append(
                SingleLevelTextGuidedCorrectionV5Plus(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    embed_channels=embed_channels[i],  # ⭐ 使用该尺度的投影维度
                    num_classes=num_classes,
                    cosine_scale=cosine_scale,
                )
            )
    
    def get_alpha_values(self):
        """获取当前的 Alpha 值"""
        return [alpha.item() for alpha in self.correction_alphas]
    
    def reset_alpha_log_flag(self):
        """重置 epoch 打印标记（每个 epoch 开始时调用）"""
        self._alpha_logged_this_epoch = False
    
    def _log_alpha_values(self):
        """打印 Alpha 值到日志"""
        try:
            logger = MMLogger.get_current_instance()
        except Exception:
            logger = None
            
        alphas = self.get_alpha_values()
        alpha_str = ', '.join([f'Level_{i}: {a:.6f}' for i, a in enumerate(alphas)])
        msg = f'[V5Plus IR Correction] Alpha values: {alpha_str}'
        
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    
    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        对各尺度的 IR 特征进行文本引导纠错
        
        Args:
            rgb_feats: RGB 特征元组 (P3, P4, P5)
            ir_feats: IR 特征元组 (P3, P4, P5)
            txt_feats: 文本特征 [B, N, text_dim]
            
        Returns:
            rgb_feats: RGB 特征（不变）
            ir_corrected_feats: 纠正后的 IR 特征元组
        """
        # 训练时打印 Alpha
        if self.training and self.log_alpha:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self._log_alpha_values()
        
        # 验证时每个 epoch 只打印一次
        if not self.training and self.log_alpha and not self._alpha_logged_this_epoch:
            self._log_alpha_values()
            self._alpha_logged_this_epoch = True
        
        ir_corrected_feats = []
        for i, module in enumerate(self.correction_modules):
            ir_corrected = module(
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats,
                self.correction_alphas[i]
            )
            ir_corrected_feats.append(ir_corrected)
        
        return rgb_feats, tuple(ir_corrected_feats)


class SingleLevelTextGuidedCorrectionV5Plus(nn.Module):
    """
    单尺度的 Text-guided IR Correction V5 Plus
    
    核心改进：
    1. 多尺度动态投影维度 embed_channels（参考 YOLO-World）
    2. 保持 V5 的余弦相似度机制
    3. 移除 Key 投影的 bias
    
    数学对比：
        V5: 所有尺度 d_k = 128
        V5Plus: d_k = embed_channels (每尺度独立)
        
    参考 YOLO-World 的设计：
        MaxSigmoidAttnBlock:
            guide_fc = Linear(guide_channels, embed_channels)  # 文本投影
            embed_conv = Conv2d(in_channels, embed_channels, 1)  # 图像投影
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        embed_channels: int = 128,  # ⭐ V5Plus: 使用 embed_channels 替代固定 d_k
        num_classes: int = 4,
        cosine_scale: float = 10.0,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.embed_channels = embed_channels  # ⭐ 该尺度的投影维度
        self.num_classes = num_classes
        self.cosine_scale = cosine_scale
        
        # ⭐ V5Plus: Query/Key 投影使用 embed_channels（动态维度）
        # 参考 YOLO-World 的 guide_fc 和 embed_conv
        
        # Text Query 投影: text_dim → embed_channels（保留 bias）
        # 类似 YOLO-World: guide_fc = Linear(guide_channels, embed_channels)
        self.text_query_proj = nn.Linear(text_dim, embed_channels)
        
        # RGB Key 投影: rgb_channels → embed_channels（移除 bias）
        # 类似 YOLO-World: embed_conv = Conv2d(in_channels, embed_channels, 1)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, embed_channels, kernel_size=1, bias=False)
        
        # IR Key 投影: ir_channels → embed_channels（移除 bias）
        self.ir_key_proj = nn.Conv2d(ir_channels, embed_channels, kernel_size=1, bias=False)
                
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
        V5Plus 改进的 forward 逻辑：使用动态 embed_channels + 余弦相似度
        
        核心变化：
            V5: d_k = 128 (固定)
            V5Plus: d_k = embed_channels (每尺度独立)
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N_actual = txt_feats.size(1)
        d_k = self.embed_channels  # ⭐ 使用该尺度的投影维度
        
        # 尺寸对齐
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: 语义激活（余弦相似度 + 动态 embed_channels）
        # ═══════════════════════════════════════════════════════════════════
        
        # Query 投影: [B, N, text_dim] → [B, N, embed_channels]
        Q = self.text_query_proj(txt_feats)  # [B, N_actual, d_k]
        
        # RGB Key 投影: [B, C_rgb, H, W] → [B, d_k, H*W]
        K_rgb = self.rgb_key_proj(x_rgb)  # [B, d_k, H, W]
        K_rgb_flat = K_rgb.view(B, d_k, H * W)  # [B, d_k, H*W]
        
        # IR Key 投影: [B, C_ir, H, W] → [B, d_k, H*W]
        K_ir = self.ir_key_proj(x_ir)  # [B, d_k, H, W]
        K_ir_flat = K_ir.view(B, d_k, H * W)  # [B, d_k, H*W]
        
        # ⭐ L2 归一化（沿投影维度）→ 余弦相似度
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [B, N_actual, d_k]
        K_rgb_norm = F.normalize(K_rgb_flat, p=2, dim=1)  # [B, d_k, H*W]
        K_ir_norm = F.normalize(K_ir_flat, p=2, dim=1)  # [B, d_k, H*W]
        
        # ⭐ 余弦相似度 + scale
        scale = self.cosine_scale  # 10.0
        
        # RGB 注意力: [B, N, d_k] @ [B, d_k, H*W] = [B, N, H*W]
        attn_logits_rgb = torch.bmm(Q_norm, K_rgb_norm) * scale
        A_rgb = F.softmax(attn_logits_rgb, dim=-1)  # [B, N_actual, H*W]
        
        # IR 注意力
        attn_logits_ir = torch.bmm(Q_norm, K_ir_norm) * scale
        A_ir = F.softmax(attn_logits_ir, dim=-1)  # [B, N_actual, H*W]
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2-5: 与 V5 完全相同
        # ═══════════════════════════════════════════════════════════════════
        
        # Step 2: 一致性度量
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # [B, N_actual]
        G = torch.clamp(G, 0.0, 1.0)
        
        # Step 3: 加权差异图
        D_spatial = torch.abs(A_rgb - A_ir)  # [B, N_actual, H*W]
        disagreement = 1.0 - G  # [B, N_actual]
        M_err = torch.einsum('bn,bnh->bh', disagreement, D_spatial)  # [B, H*W]
        
        # Min-Max 归一化
        M_err_min = M_err.min(dim=-1, keepdim=True)[0]
        M_err_max = M_err.max(dim=-1, keepdim=True)[0]
        M_err_range = torch.clamp(M_err_max - M_err_min, min=1e-6)
        M_err = (M_err - M_err_min) / M_err_range
        M_err_spatial = M_err.view(B, 1, H, W)
        
        # Step 4: 空间门控
        F_extracted = x_ir * M_err_spatial
        Error_map = self.error_estimator(F_extracted)
        
        # Step 5: 特征纠正
        ir_corrected = x_ir - alpha * Error_map
        
        return ir_corrected


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithCorrectionV5Plus(BaseModule):
    """
    带IR纠错的双流多模态YOLO Backbone V5 Plus
    
    基于 V5，核心改动：
    - 多尺度动态投影维度 embed_channels（参考 YOLO-World）
    - 保持余弦相似度机制
    - Alpha 初始值保持 -0.3
    """
    
    def __init__(
        self,
        image_model: dict,
        ir_model: dict,
        text_model: dict = None,
        fusion_module: dict = None,
        ir_correction: dict = None,
        with_text_model: bool = True,
        frozen_stages: int = -1,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        # RGB backbone
        self.image_model = MODELS.build(image_model)
        
        # IR backbone
        self.ir_model = MODELS.build(ir_model)
        
        # Text model (CLIP)
        self.with_text_model = with_text_model
        if with_text_model and text_model is not None:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        
        # ⭐ V5Plus: Text-guided IR correction (多尺度动态投影)
        self.with_ir_correction = ir_correction is not None
        if self.with_ir_correction:
            self.ir_correction = MODELS.build(ir_correction)
        else:
            self.ir_correction = None
        
        # RGB-IR fusion
        if fusion_module is not None:
            self.fusion_module = MODELS.build(fusion_module)
        else:
            self.fusion_module = None
        
        self.frozen_stages = frozen_stages
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages based on frozen_stages parameter."""
        if self.frozen_stages >= 0:
            if hasattr(self.image_model, 'stem'):
                self.image_model.stem.eval()
                for param in self.image_model.stem.parameters():
                    param.requires_grad = False
            
            for i in range(min(self.frozen_stages + 1, 4)):
                if hasattr(self.image_model, f'stage{i}'):
                    m = getattr(self.image_model, f'stage{i}')
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
        
        # ⭐ V5Plus: 使用多尺度动态投影的 IR 纠错
        if self.with_ir_correction and txt_feats is not None:
            # 处理 text_model 返回的 tuple: (txt_feats_tensor, txt_masks)
            if isinstance(txt_feats, tuple):
                txt_feats_tensor = txt_feats[0]  # 提取实际的 tensor
            else:
                txt_feats_tensor = txt_feats
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats_tensor)
        
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
        """Forward image only (for inference)."""
        if img_ir is None:
            img_ir = image
        
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        
        # Get text features if available
        txt_feats = None
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
        
        # IR correction
        if self.with_ir_correction and txt_feats is not None:
            # 处理 text_model 返回的 tuple: (txt_feats_tensor, txt_masks)
            if isinstance(txt_feats, tuple):
                txt_feats_tensor = txt_feats[0]
            else:
                txt_feats_tensor = txt_feats
            rgb_feats, ir_feats = self.ir_correction(rgb_feats, ir_feats, txt_feats_tensor)
        
        # Fusion
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return img_feats, txt_feats
