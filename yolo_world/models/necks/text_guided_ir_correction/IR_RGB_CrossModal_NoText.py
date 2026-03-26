# Ablation: No Text Anchor IR Correction
#
# 消融实验：去掉 text 语义引导，仅用 RGB-IR 视觉特征做跨模态交互
#
# 两个方案：
#   1. IR_RGB_CrossModal_CrossAttn: Q=IR全空间, K=RGB池化, cross-attention
#   2. IR_RGB_CrossModal_CosineSim: 逐位置 cosine similarity
#
# 与 IR_RGB_Merr_Cons 的关键区别：
#   - 没有 text Q → 无法区分类别 → 无法做 err/cons 分解
#   - 只有单一 response_map + 单一 gamma + 单一 estimator
#   - 融合公式: ir_corrected = x_ir * (1 + gamma * estimator(x_ir * response_map))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


# ============================================================================
# 单尺度模块
# ============================================================================

class SingleLevelCrossAttn(nn.Module):
    """
    单尺度 Cross-Attention (No Text)
    
    Q = proj(X_ir) 全空间, K = GAP(proj(X_rgb)) 池化
    response = sigmoid(Q^T @ K / sqrt(d_k))
    """

    def __init__(self, rgb_channels: int, ir_channels: int, d_k: int = 128):
        super().__init__()
        self.d_k = d_k

        self.ir_query_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)

        self.estimator = nn.Sequential(
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
        gamma: torch.Tensor
    ) -> torch.Tensor:
        B, C_ir, H, W = x_ir.shape

        if x_rgb.shape[-2:] != x_ir.shape[-2:]:
            x_rgb = F.interpolate(x_rgb, size=(H, W), mode='bilinear', align_corners=False)

        # Q = IR 全空间
        Q = self.ir_query_proj(x_ir).flatten(2)          # (B, d_k, H*W)
        # K = RGB 全局池化
        K = F.adaptive_avg_pool2d(
            self.rgb_key_proj(x_rgb), 1
        ).flatten(2)                                       # (B, d_k, 1)

        d_k_sqrt = self.d_k ** 0.5
        # Q^T @ K → (B, H*W, 1)
        attn = torch.bmm(Q.transpose(1, 2), K) / d_k_sqrt  # (B, H*W, 1)
        response = torch.sigmoid(attn)                       # (B, H*W, 1)
        response_map = response.transpose(1, 2).view(B, 1, H, W)  # (B, 1, H, W)

        # 归一化
        r_min = response_map.flatten(2).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        r_max = response_map.flatten(2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        response_map = (response_map - r_min) / (r_max - r_min + 1e-6)

        # 融合
        F_gated = x_ir * response_map
        enhanced = self.estimator(F_gated)
        ir_corrected = x_ir * (1 + gamma * enhanced)

        return ir_corrected


class SingleLevelCosineSim(nn.Module):
    """
    单尺度 Cosine Similarity (No Text)
    
    逐位置 cosine_similarity(proj(X_rgb), proj(X_ir))
    """

    def __init__(self, rgb_channels: int, ir_channels: int, d_k: int = 128):
        super().__init__()
        self.d_k = d_k

        self.rgb_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        self.ir_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)

        self.estimator = nn.Sequential(
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
        gamma: torch.Tensor
    ) -> torch.Tensor:
        B, C_ir, H, W = x_ir.shape

        if x_rgb.shape[-2:] != x_ir.shape[-2:]:
            x_rgb = F.interpolate(x_rgb, size=(H, W), mode='bilinear', align_corners=False)

        F_rgb = self.rgb_proj(x_rgb)  # (B, d_k, H, W)
        F_ir = self.ir_proj(x_ir)     # (B, d_k, H, W)

        # 逐位置 cosine similarity → (B, H, W)
        cos_sim = F.cosine_similarity(F_rgb, F_ir, dim=1)  # (B, H, W)
        # 映射到 [0, 1]
        response_map = (cos_sim + 1) / 2  # cosine sim 范围 [-1,1] → [0,1]
        response_map = response_map.unsqueeze(1)  # (B, 1, H, W)

        # 归一化
        r_min = response_map.flatten(2).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        r_max = response_map.flatten(2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        response_map = (response_map - r_min) / (r_max - r_min + 1e-6)

        # 融合
        F_gated = x_ir * response_map
        enhanced = self.estimator(F_gated)
        ir_corrected = x_ir * (1 + gamma * enhanced)

        return ir_corrected


class SingleLevelFullCrossAttn(nn.Module):
    """
    单尺度 Full Spatial Cross-Attention (No Text)

    Q = proj(X_ir) 全空间 (B, d_k, H*W)
    K = proj(X_rgb) 全空间 (B, d_k, H*W)   ← 不池化
    attn = Q^T @ K / sqrt(d_k)              → (B, H*W, H*W)  O(H²W²)
    response = sigmoid(attn).mean(dim=-1)   → 每个 IR 位置对所有 RGB 位置的平均注意力
    """

    def __init__(self, rgb_channels: int, ir_channels: int, d_k: int = 128):
        super().__init__()
        self.d_k = d_k

        self.ir_query_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)

        self.estimator = nn.Sequential(
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
        gamma: torch.Tensor
    ) -> torch.Tensor:
        B, C_ir, H, W = x_ir.shape

        if x_rgb.shape[-2:] != x_ir.shape[-2:]:
            x_rgb = F.interpolate(x_rgb, size=(H, W), mode='bilinear', align_corners=False)

        Q = self.ir_query_proj(x_ir).flatten(2)    # (B, d_k, H*W)
        K = self.rgb_key_proj(x_rgb).flatten(2)     # (B, d_k, H*W)

        d_k_sqrt = self.d_k ** 0.5
        # (B, H*W, H*W) — full spatial attention, O(H²W²)
        attn = torch.bmm(Q.transpose(1, 2), K) / d_k_sqrt
        attn = torch.sigmoid(attn)                   # (B, H*W, H*W)
        response = attn.mean(dim=-1)                  # (B, H*W)
        response_map = response.view(B, 1, H, W)     # (B, 1, H, W)

        r_min = response_map.flatten(2).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        r_max = response_map.flatten(2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        response_map = (response_map - r_min) / (r_max - r_min + 1e-6)

        F_gated = x_ir * response_map
        enhanced = self.estimator(F_gated)
        ir_corrected = x_ir * (1 + gamma * enhanced)

        return ir_corrected


# ============================================================================
# 多尺度模块（顶层，与 IR_RGB_Merr_Cons 接口兼容）
# ============================================================================

@MODELS.register_module()
class IR_RGB_CrossModal_CrossAttn(BaseModule):
    """
    消融实验: Cross-Attention (No Text Anchor)
    
    Q=IR全空间, K=RGB池化, 单一response_map + gamma
    接口与 IR_RGB_Merr_Cons 完全兼容（接收 text_feats 但忽略）
    """

    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        d_k: int = 128,
        init_gamma: float = 0.1,
        log_gamma: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.num_levels = len(rgb_channels)
        self.log_gamma = log_gamma
        self.log_interval = log_interval
        self._train_iter_count = 0
        self._gamma_logged_this_epoch = False

        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_gamma))
            for _ in range(self.num_levels)
        ])

        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelCrossAttn(rgb_ch, ir_ch, d_k=d_k)
            )

    def _log_gamma(self, stage: str = 'val', iter_num: int = None):
        if not self.log_gamma:
            return
        if stage == 'val' and self._gamma_logged_this_epoch:
            return

        vals = [g.item() for g in self.gammas]
        logger = MMLogger.get_current_instance()
        prefix = f"Iter [{iter_num}]" if stage == 'train' and iter_num else "Val"
        logger.info(
            f"[CrossAttn_NoText] {prefix} "
            f"Gamma: P3={vals[0]:.4f}, P4={vals[1]:.4f}, P5={vals[2]:.4f}, "
            f"Mean={sum(vals)/len(vals):.4f}"
        )
        if stage == 'val':
            self._gamma_logged_this_epoch = True

    def reset_alpha_log_flag(self):
        self._gamma_logged_this_epoch = False
        self._train_iter_count = 0

    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # txt_feats is accepted for interface compatibility but ignored

        if self.training:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self._log_gamma(stage='train', iter_num=self._train_iter_count)
        else:
            self._log_gamma(stage='val')

        ir_corrected_feats = []
        for i in range(self.num_levels):
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], ir_feats[i], self.gammas[i]
            )
            ir_corrected_feats.append(ir_corrected)

        return rgb_feats, tuple(ir_corrected_feats)


@MODELS.register_module()
class IR_RGB_CrossModal_CosineSim(BaseModule):
    """
    消融实验: Cosine Similarity (No Text Anchor)
    
    逐位置 cosine_similarity(proj(X_rgb), proj(X_ir))
    接口与 IR_RGB_Merr_Cons 完全兼容（接收 text_feats 但忽略）
    """

    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        d_k: int = 128,
        init_gamma: float = 0.1,
        log_gamma: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.num_levels = len(rgb_channels)
        self.log_gamma = log_gamma
        self.log_interval = log_interval
        self._train_iter_count = 0
        self._gamma_logged_this_epoch = False

        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_gamma))
            for _ in range(self.num_levels)
        ])

        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelCosineSim(rgb_ch, ir_ch, d_k=d_k)
            )

    def _log_gamma(self, stage: str = 'val', iter_num: int = None):
        if not self.log_gamma:
            return
        if stage == 'val' and self._gamma_logged_this_epoch:
            return

        vals = [g.item() for g in self.gammas]
        logger = MMLogger.get_current_instance()
        prefix = f"Iter [{iter_num}]" if stage == 'train' and iter_num else "Val"
        logger.info(
            f"[CosineSim_NoText] {prefix} "
            f"Gamma: P3={vals[0]:.4f}, P4={vals[1]:.4f}, P5={vals[2]:.4f}, "
            f"Mean={sum(vals)/len(vals):.4f}"
        )
        if stage == 'val':
            self._gamma_logged_this_epoch = True

    def reset_alpha_log_flag(self):
        self._gamma_logged_this_epoch = False
        self._train_iter_count = 0

    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        # txt_feats is accepted for interface compatibility but ignored (CosineSim)

        if self.training:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self._log_gamma(stage='train', iter_num=self._train_iter_count)
        else:
            self._log_gamma(stage='val')

        ir_corrected_feats = []
        for i in range(self.num_levels):
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], ir_feats[i], self.gammas[i]
            )
            ir_corrected_feats.append(ir_corrected)

        return rgb_feats, tuple(ir_corrected_feats)


@MODELS.register_module()
class IR_RGB_CrossModal_FullCrossAttn(BaseModule):
    """
    消融实验: Full Spatial Cross-Attention (No Text Anchor)

    Q=IR全空间, K=RGB全空间 (不池化), O(H²W²)
    接口与 IR_RGB_Merr_Cons 完全兼容（接收 text_feats 但忽略）
    """

    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        d_k: int = 128,
        init_gamma: float = 0.1,
        log_gamma: bool = True,
        log_interval: int = 50,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.num_levels = len(rgb_channels)
        self.log_gamma = log_gamma
        self.log_interval = log_interval
        self._train_iter_count = 0
        self._gamma_logged_this_epoch = False

        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_gamma))
            for _ in range(self.num_levels)
        ])

        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelFullCrossAttn(rgb_ch, ir_ch, d_k=d_k)
            )

    def _log_gamma(self, stage: str = 'val', iter_num: int = None):
        if not self.log_gamma:
            return
        if stage == 'val' and self._gamma_logged_this_epoch:
            return

        vals = [g.item() for g in self.gammas]
        logger = MMLogger.get_current_instance()
        prefix = f"Iter [{iter_num}]" if stage == 'train' and iter_num else "Val"
        logger.info(
            f"[FullCrossAttn_NoText] {prefix} "
            f"Gamma: P3={vals[0]:.4f}, P4={vals[1]:.4f}, P5={vals[2]:.4f}, "
            f"Mean={sum(vals)/len(vals):.4f}"
        )
        if stage == 'val':
            self._gamma_logged_this_epoch = True

    def reset_alpha_log_flag(self):
        self._gamma_logged_this_epoch = False
        self._train_iter_count = 0

    def forward(
        self,
        rgb_feats: Tuple[torch.Tensor, ...],
        ir_feats: Tuple[torch.Tensor, ...],
        txt_feats: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.training:
            self._train_iter_count += 1
            if self._train_iter_count % self.log_interval == 0:
                self._log_gamma(stage='train', iter_num=self._train_iter_count)
        else:
            self._log_gamma(stage='val')

        ir_corrected_feats = []
        for i in range(self.num_levels):
            ir_corrected = self.correction_modules[i](
                rgb_feats[i], ir_feats[i], self.gammas[i]
            )
            ir_corrected_feats.append(ir_corrected)

        return rgb_feats, tuple(ir_corrected_feats)
