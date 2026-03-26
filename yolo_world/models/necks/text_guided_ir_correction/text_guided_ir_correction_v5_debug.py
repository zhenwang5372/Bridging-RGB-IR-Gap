# Copyright (c) Tencent Inc. All rights reserved.
# Text-guided IR Correction Module V5 - Debug Version
# 
# 基于 V5，添加：
# - Key 投影权重相似度分析
# - 投影后特征相似度分析
# - 注意力分布统计信息
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


@MODELS.register_module()
class TextGuidedIRCorrectionV5Debug(BaseModule):
    """
    Text-guided IR Correction Module V5 - Debug Version
    
    基于 V5，添加了详细的调试信息输出：
    - Key 投影权重的相似度分析
    - 投影后特征的相似度分析
    - 注意力分布的统计信息
    
    Args:
        rgb_channels (List[int]): RGB特征通道数 [P3, P4, P5]
        ir_channels (List[int]): IR特征通道数 [P3, P4, P5]
        text_dim (int): 文本特征维度，默认512
        num_classes (int): 类别数，默认4
        correction_alpha (float): 纠错强度初始值，默认 -0.3
        cosine_scale (float): 余弦相似度缩放因子，默认 10.0
        log_alpha (bool): 是否打印 Alpha 值到日志，默认 True
        log_interval (int): 训练时打印 Alpha 的间隔（iteration 数），默认 50
        debug_mode (bool): 是否启用调试模式（打印详细信息），默认 True
        init_cfg (dict, optional): 初始化配置
    """
    
    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        correction_alpha: float = -0.3,
        cosine_scale: float = 10.0,
        log_alpha: bool = True,
        log_interval: int = 50,
        debug_mode: bool = True,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = len(rgb_channels)
        self.cosine_scale = cosine_scale
        self.log_alpha = log_alpha
        self.log_interval = log_interval
        self.debug_mode = debug_mode
        
        # 训练时的 iteration 计数器
        self._train_iter_count = 0
        # 用于控制验证时每个 epoch 只打印一次
        self._alpha_logged_this_epoch = False
        # 用于控制调试信息每个 epoch 只打印一次
        self._debug_logged_this_epoch = False
        
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
        
        # 为每个尺度创建纠错模块
        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelTextGuidedCorrectionV5Debug(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    cosine_scale=cosine_scale,
                    debug_mode=debug_mode,
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
                f"[IR Correction V5 Debug] Iter [{iter_num}] Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}, "
                f"Scale={self.cosine_scale:.2f}"
            )
        else:
            logger.info(
                f"[IR Correction V5 Debug] Val Alpha: "
                f"P3={alphas['P3']:.6f}, P4={alphas['P4']:.6f}, "
                f"P5={alphas['P5']:.6f}, Mean={alphas['mean']:.6f}, "
                f"Scale={self.cosine_scale:.2f}"
            )
        
        if stage == 'val':
            self._alpha_logged_this_epoch = True
    
    def reset_alpha_log_flag(self):
        """重置 Alpha 打印标志（在新 epoch 开始时调用）"""
        self._alpha_logged_this_epoch = False
        self._debug_logged_this_epoch = False
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
        
        # 是否打印调试信息（仅验证时且每个epoch只打印一次）
        enable_debug = self.debug_mode and not self.training and not self._debug_logged_this_epoch
        
        for i in range(self.num_levels):
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], 
                ir_feats[i], 
                txt_feats,
                self.correction_alphas[i],
                level_name=f"P{i+3}",
                enable_debug=enable_debug
            )
            ir_corrected_feats.append(ir_corrected)
        
        if enable_debug:
            self._debug_logged_this_epoch = True
        
        return rgb_feats, tuple(ir_corrected_feats)


class SingleLevelTextGuidedCorrectionV5Debug(nn.Module):
    """
    单尺度的 Text-guided IR Correction V5 - Debug Version
    
    添加了详细的调试信息输出
    """
    
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        cosine_scale: float = 10.0,
        debug_mode: bool = True,
    ):
        super().__init__()
        
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.cosine_scale = cosine_scale
        self.debug_mode = debug_mode
        
        # Query/Key 投影
        d_k = 128
        self.d_k = d_k
        
        # Text Query 投影（保留 bias）
        self.text_query_proj = nn.Linear(text_dim, d_k)
        
        # RGB Key 投影（移除 bias）
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1, bias=False)
        
        # IR Key 投影（移除 bias）
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1, bias=False)
        
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
        alpha: torch.Tensor,
        level_name: str = "P3",
        enable_debug: bool = False
    ) -> torch.Tensor:
        """
        V5 改进的 forward 逻辑：使用余弦相似度
        
        核心变化：
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            attn_logits = Q_norm @ K_norm^T * scale
        """
        B, C_rgb, H, W = x_rgb.shape
        _, C_ir, _, _ = x_ir.shape
        N_actual = txt_feats.size(1)
        
        # 尺寸对齐
        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=(H, W), mode='bilinear', align_corners=False)
        
        # ═══════════════════════════════════════════════════════════════════
        # 🔍 调试信息：Key 投影权重相似度分析
        # ═══════════════════════════════════════════════════════════════════
        if enable_debug and self.debug_mode:
            logger = MMLogger.get_current_instance()
            logger.info("\n" + "="*70)
            logger.info(f"🔍 Key Projection Analysis - {level_name}")
            logger.info("="*70)
            
            # 1. 权重层面的分析
            # RGB: [d_k, C_rgb, 1, 1], IR: [d_k, C_ir, 1, 1]
            # 由于输入通道数不同，我们比较投影到相同维度(d_k)后的分布特性
            W_rgb = self.rgb_key_proj.weight  # [128, C_rgb, 1, 1]
            W_ir = self.ir_key_proj.weight    # [128, C_ir, 1, 1]
            
            logger.info(f"1. Weight shape analysis:")
            logger.info(f"   - RGB Key weight shape: {W_rgb.shape} (out={W_rgb.shape[0]}, in={W_rgb.shape[1]})")
            logger.info(f"   - IR Key weight shape:  {W_ir.shape} (out={W_ir.shape[0]}, in={W_ir.shape[1]})")
            
            # 权重统计（按输出维度分析）
            W_rgb_flat = W_rgb.view(W_rgb.shape[0], -1)  # [d_k, C_rgb]
            W_ir_flat = W_ir.view(W_ir.shape[0], -1)      # [d_k, C_ir]
            
            logger.info(f"   - RGB Key weight: mean={W_rgb_flat.mean().item():.6f}, "
                       f"std={W_rgb_flat.std().item():.6f}, "
                       f"norm={W_rgb_flat.norm().item():.2f}")
            logger.info(f"   - IR Key weight:  mean={W_ir_flat.mean().item():.6f}, "
                       f"std={W_ir_flat.std().item():.6f}, "
                       f"norm={W_ir_flat.norm().item():.2f}")
            
            # 计算每个输出维度的权重向量的余弦相似度（如果维度相同）
            # 这里比较投影后128维输出空间中的每个维度的"投影方向"是否相似
            W_rgb_per_out = F.normalize(W_rgb_flat, p=2, dim=1)  # [d_k, C_rgb] 归一化
            W_ir_per_out = F.normalize(W_ir_flat, p=2, dim=1)    # [d_k, C_ir] 归一化
            
            # 计算每个输出通道的平均投影强度
            logger.info(f"2. Projection direction analysis:")
            logger.info(f"   - RGB channels avg norm per output dim: {W_rgb_flat.norm(dim=1).mean().item():.6f}")
            logger.info(f"   - IR channels avg norm per output dim:  {W_ir_flat.norm(dim=1).mean().item():.6f}")
            
            # 如果想看它们是否学习到了相似的"模式"，可以看权重矩阵的协方差
            # 这里我们计算权重在输出维度上的相关性
            W_rgb_mean = W_rgb_flat.mean(dim=1, keepdim=True)
            W_ir_mean = W_ir_flat.mean(dim=1, keepdim=True)
            W_rgb_centered = W_rgb_flat - W_rgb_mean
            W_ir_centered = W_ir_flat - W_ir_mean
            
            # 计算权重分布的"形状"相似度（不考虑具体数值）
            rgb_std = W_rgb_centered.std(dim=1)
            ir_std = W_ir_centered.std(dim=1)
            std_similarity = F.cosine_similarity(rgb_std.unsqueeze(0), ir_std.unsqueeze(0))
            logger.info(f"   - Weight std pattern similarity: {std_similarity.item():.6f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: 语义激活（⭐ V5 改进：余弦相似度）
        # ═══════════════════════════════════════════════════════════════════
        
        # Query 投影: [B, N, d_k]
        Q = self.text_query_proj(txt_feats)  # [B, N_actual, 128]
        
        # RGB Key 投影: [B, d_k, H, W] → [B, d_k, H*W]
        K_rgb = self.rgb_key_proj(x_rgb)  # [B, 128, H, W]
        K_rgb_flat = K_rgb.view(B, self.d_k, H * W)  # [B, 128, H*W]
        
        # IR Key 投影: [B, d_k, H, W] → [B, d_k, H*W]
        K_ir = self.ir_key_proj(x_ir)  # [B, 128, H, W]
        K_ir_flat = K_ir.view(B, self.d_k, H * W)  # [B, 128, H*W]
        
        # 🔍 调试信息：投影后特征的相似度（归一化前）
        if enable_debug and self.debug_mode:
            # 2. 投影后特征的整体相似度（归一化前）
            K_rgb_flat_vec = K_rgb_flat.view(B, -1)  # [B, d_k * H * W]
            K_ir_flat_vec = K_ir_flat.view(B, -1)    # [B, d_k * H * W]
            feat_cos_sim = F.cosine_similarity(K_rgb_flat_vec, K_ir_flat_vec, dim=1).mean()
            logger.info(f"2. Projected features cosine similarity (before norm): {feat_cos_sim.item():.6f}")
            
            # 特征统计
            logger.info(f"   - RGB Key feat: mean={K_rgb_flat.mean().item():.6f}, "
                       f"std={K_rgb_flat.std().item():.6f}")
            logger.info(f"   - IR Key feat:  mean={K_ir_flat.mean().item():.6f}, "
                       f"std={K_ir_flat.std().item():.6f}")
        
        # ⭐ V5 核心改动：L2 归一化（沿 d_k 维度）
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [B, N_actual, 128], ||Q|| = 1
        K_rgb_norm = F.normalize(K_rgb_flat, p=2, dim=1)  # [B, 128, H*W], ||K|| = 1
        K_ir_norm = F.normalize(K_ir_flat, p=2, dim=1)  # [B, 128, H*W], ||K|| = 1
        
        # 🔍 调试信息：归一化后的特征相似度
        if enable_debug and self.debug_mode:
            # 3. 归一化后的相似度
            K_rgb_norm_vec = K_rgb_norm.view(B, -1)
            K_ir_norm_vec = K_ir_norm.view(B, -1)
            norm_cos_sim = F.cosine_similarity(K_rgb_norm_vec, K_ir_norm_vec, dim=1).mean()
            logger.info(f"3. Normalized features cosine similarity: {norm_cos_sim.item():.6f}")
            
            # 4. 每个空间位置的相似度分布
            spatial_cos_sim = F.cosine_similarity(K_rgb_norm, K_ir_norm, dim=1)  # [B, H*W]
            logger.info(f"4. Per-position cosine similarity:")
            logger.info(f"   - mean={spatial_cos_sim.mean().item():.6f}, "
                       f"std={spatial_cos_sim.std().item():.6f}")
            logger.info(f"   - min={spatial_cos_sim.min().item():.6f}, "
                       f"max={spatial_cos_sim.max().item():.6f}")
            logger.info(f"   - median={spatial_cos_sim.median().item():.6f}")
        
        # ⭐ V5 改进：余弦相似度 + scale
        scale = self.cosine_scale  # 10.0
        
        # RGB 注意力: [B, N, d_k] @ [B, d_k, H*W] = [B, N, H*W]
        attn_logits_rgb = torch.bmm(Q_norm, K_rgb_norm) * scale  # [B, N_actual, H*W]
        A_rgb = F.softmax(attn_logits_rgb, dim=-1)  # [B, N_actual, H*W]
        
        # IR 注意力: [B, N, d_k] @ [B, d_k, H*W] = [B, N, H*W]
        attn_logits_ir = torch.bmm(Q_norm, K_ir_norm) * scale  # [B, N_actual, H*W]
        A_ir = F.softmax(attn_logits_ir, dim=-1)  # [B, N_actual, H*W]
        
        # 🔍 调试信息：注意力分布分析
        if enable_debug and self.debug_mode:
            logger.info(f"5. Attention distribution analysis:")
            
            # 注意力 logits 统计
            logger.info(f"   - RGB attn_logits: mean={attn_logits_rgb.mean().item():.4f}, "
                       f"std={attn_logits_rgb.std().item():.4f}, "
                       f"min={attn_logits_rgb.min().item():.4f}, "
                       f"max={attn_logits_rgb.max().item():.4f}")
            logger.info(f"   - IR attn_logits:  mean={attn_logits_ir.mean().item():.4f}, "
                       f"std={attn_logits_ir.std().item():.4f}, "
                       f"min={attn_logits_ir.min().item():.4f}, "
                       f"max={attn_logits_ir.max().item():.4f}")
            
            # 注意力权重统计
            logger.info(f"   - RGB attention: entropy={-(A_rgb * torch.log(A_rgb + 1e-10)).sum(dim=-1).mean().item():.4f}")
            logger.info(f"   - IR attention:  entropy={-(A_ir * torch.log(A_ir + 1e-10)).sum(dim=-1).mean().item():.4f}")
            
            # 注意力差异
            A_diff = torch.abs(A_rgb - A_ir)
            logger.info(f"   - |A_rgb - A_ir|: mean={A_diff.mean().item():.6f}, "
                       f"max={A_diff.max().item():.6f}")
            
            # 注意力相似度（逐类别）
            A_rgb_flat = A_rgb.view(B, N_actual, -1)
            A_ir_flat = A_ir.view(B, N_actual, -1)
            for n in range(min(N_actual, 4)):  # 只显示前4个类别
                attn_cos_sim = F.cosine_similarity(A_rgb_flat[:, n:n+1, :], A_ir_flat[:, n:n+1, :], dim=-1).mean()
                logger.info(f"   - Class {n} attention cosine similarity: {attn_cos_sim.item():.6f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2-5: 与 V5 完全相同
        # ═══════════════════════════════════════════════════════════════════
        
        # Step 2: 一致性度量
        A_rgb_norm = F.normalize(A_rgb, p=2, dim=-1)  # [B, N_actual, H*W]
        A_ir_norm = F.normalize(A_ir, p=2, dim=-1)  # [B, N_actual, H*W]
        G = torch.sum(A_rgb_norm * A_ir_norm, dim=-1)  # [B, N_actual]
        G = torch.clamp(G, 0.0, 1.0)
        
        # 🔍 调试信息：一致性度量
        if enable_debug and self.debug_mode:
            logger.info(f"6. Consistency metric (G):")
            logger.info(f"   - mean={G.mean().item():.6f}, "
                       f"std={G.std().item():.6f}, "
                       f"min={G.min().item():.6f}, "
                       f"max={G.max().item():.6f}")
            disagreement = 1.0 - G
            logger.info(f"   - disagreement: mean={disagreement.mean().item():.6f}")
        
        # Step 3: 加权差异图
        D_spatial = torch.abs(A_rgb - A_ir)  # [B, N_actual, H*W]
        disagreement = 1.0 - G  # [B, N_actual]
        M_err = torch.einsum('bn,bnh->bh', disagreement, D_spatial)  # [B, H*W]
        
        # Min-Max 归一化
        M_err_min = M_err.min(dim=-1, keepdim=True)[0]  # [B, 1]
        M_err_max = M_err.max(dim=-1, keepdim=True)[0]  # [B, 1]
        M_err_range = torch.clamp(M_err_max - M_err_min, min=1e-6)
        M_err = (M_err - M_err_min) / M_err_range  # [B, H*W]
        M_err_spatial = M_err.view(B, 1, H, W)  # [B, 1, H, W]
        
        # 🔍 调试信息：误差图统计
        if enable_debug and self.debug_mode:
            logger.info(f"7. Error map (M_err):")
            logger.info(f"   - mean={M_err.mean().item():.6f}, "
                       f"std={M_err.std().item():.6f}")
            logger.info(f"   - pixels > 0.5: {(M_err > 0.5).float().mean().item()*100:.2f}%")
            logger.info("="*70 + "\n")
        
        # Step 4: 空间门控
        F_extracted = x_ir * M_err_spatial  # [B, C_ir, H, W]
        Error_map = self.error_estimator(F_extracted)  # [B, C_ir, H, W]
        
        # Step 5: 特征纠正
        ir_corrected = x_ir - alpha * Error_map
        
        return ir_corrected


@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithCorrectionV5Debug(BaseModule):
    """
    带IR纠错的双流多模态YOLO Backbone V5 - Debug Version
    
    基于 V5，添加调试信息输出
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
        
        # Build IR correction module V5 Debug
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

