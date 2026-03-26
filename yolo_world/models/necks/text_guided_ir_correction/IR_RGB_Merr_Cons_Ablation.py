# Ablation variants of IR_RGB_Merr_Cons
#
# Controlled by three flags:
#   use_cons (bool):        enable M_cons branch
#   use_err (bool):         enable M_err (M_dis) branch
#   learnable_coeffs (bool): if False, alpha/beta are fixed (register_buffer)
#
# Variants:
#   Only M_cons:        use_cons=True,  use_err=False, learnable_coeffs=False
#   Only M_dis:         use_cons=False, use_err=True,  learnable_coeffs=False
#   M_cons+M_dis fixed: use_cons=True,  use_err=True,  learnable_coeffs=False

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmyolo.registry import MODELS


class SingleLevelMerrConsAblation(nn.Module):
    """Single-level ablation module with conditional branches."""

    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        d_k: int = 128,
        use_cons: bool = True,
        use_err: bool = True,
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.d_k = d_k
        self.use_cons = use_cons
        self.use_err = use_err

        self.text_query_proj = nn.Linear(text_dim, d_k)
        self.rgb_key_proj = nn.Conv2d(rgb_channels, d_k, kernel_size=1)
        self.ir_key_proj = nn.Conv2d(ir_channels, d_k, kernel_size=1)

        def _make_estimator(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1),
            )

        if use_err:
            self.error_estimator = _make_estimator(ir_channels)
        if use_cons:
            self.cons_estimator = _make_estimator(ir_channels)

    def forward(self, x_rgb, x_ir, txt_feats, alpha, beta):
        B, C_rgb, H, W = x_rgb.shape

        if x_ir.shape[-2:] != x_rgb.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=(H, W), mode='bilinear',
                                 align_corners=False)

        Q = self.text_query_proj(txt_feats)
        K_rgb = self.rgb_key_proj(x_rgb).view(B, -1, H * W)
        K_ir = self.ir_key_proj(x_ir).view(B, -1, H * W)

        s = Q.size(-1) ** 0.5
        A_rgb = torch.sigmoid(torch.bmm(Q, K_rgb) / s)
        A_ir = torch.sigmoid(torch.bmm(Q, K_ir) / s)

        ir_corrected = x_ir

        if self.use_err:
            M_err = (A_ir * (1 - A_rgb)).mean(dim=1)
            lo = M_err.min(-1, keepdim=True)[0]
            hi = M_err.max(-1, keepdim=True)[0]
            M_err = (M_err - lo) / (hi - lo).clamp(min=1e-6)
            M_err_sp = M_err.view(B, 1, H, W)
            Error_map = self.error_estimator(x_ir * M_err_sp)
            ir_corrected = ir_corrected * (1 - alpha * Error_map)

        if self.use_cons:
            M_cons = (A_ir * A_rgb).mean(dim=1)
            lo = M_cons.min(-1, keepdim=True)[0]
            hi = M_cons.max(-1, keepdim=True)[0]
            M_cons = (M_cons - lo) / (hi - lo).clamp(min=1e-6)
            M_cons_sp = M_cons.view(B, 1, H, W)
            Cons_map = self.cons_estimator(x_ir * M_cons_sp)
            ir_corrected = ir_corrected * (1 + beta * Cons_map)

        return ir_corrected


@MODELS.register_module()
class IR_RGB_Merr_Cons_Ablation(BaseModule):
    """
    Ablation wrapper for Merr/Cons separation experiments.

    Args:
        use_cons: enable M_cons (consensus) branch
        use_err: enable M_err (discrepancy) branch
        learnable_coeffs: if True, alpha/beta are nn.Parameter;
                          if False, they are fixed buffers
    """

    def __init__(
        self,
        rgb_channels: List[int],
        ir_channels: List[int],
        text_dim: int = 512,
        num_classes: int = 4,
        correction_alpha: float = -0.5,
        enhancement_beta: float = 0.5,
        d_k: int = 128,
        use_cons: bool = True,
        use_err: bool = True,
        learnable_coeffs: bool = False,
        log_alpha: bool = True,
        log_interval: int = 50,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.num_levels = len(rgb_channels)
        self.use_cons = use_cons
        self.use_err = use_err
        self.learnable_coeffs = learnable_coeffs
        self.log_alpha = log_alpha
        self.log_interval = log_interval
        self._train_iter_count = 0
        self._alpha_logged_this_epoch = False

        tag = []
        if use_cons:
            tag.append('Mcons')
        if use_err:
            tag.append('Mdis')
        coeff_tag = 'learnable' if learnable_coeffs else 'fixed'
        self._variant_name = '+'.join(tag) + f'({coeff_tag})'

        if learnable_coeffs:
            self.correction_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(correction_alpha))
                for _ in range(self.num_levels)
            ])
            self.enhancement_betas = nn.ParameterList([
                nn.Parameter(torch.tensor(enhancement_beta))
                for _ in range(self.num_levels)
            ])
        else:
            self.correction_alphas = []
            self.enhancement_betas = []
            for i in range(self.num_levels):
                buf_a = torch.tensor(correction_alpha)
                buf_b = torch.tensor(enhancement_beta)
                self.register_buffer(f'alpha_{i}', buf_a)
                self.register_buffer(f'beta_{i}', buf_b)
                self.correction_alphas.append(buf_a)
                self.enhancement_betas.append(buf_b)

        self.correction_modules = nn.ModuleList()
        for rgb_ch, ir_ch in zip(rgb_channels, ir_channels):
            self.correction_modules.append(
                SingleLevelMerrConsAblation(
                    rgb_channels=rgb_ch,
                    ir_channels=ir_ch,
                    text_dim=text_dim,
                    num_classes=num_classes,
                    d_k=d_k,
                    use_cons=use_cons,
                    use_err=use_err,
                )
            )

    def get_alpha_beta_values(self):
        vals = {}
        for i, name in enumerate(['P3', 'P4', 'P5']):
            a = self.correction_alphas[i]
            b = self.enhancement_betas[i]
            vals[f'alpha_{name}'] = a.item() if torch.is_tensor(a) else a
            vals[f'beta_{name}'] = b.item() if torch.is_tensor(b) else b
        vals['alpha_mean'] = sum(vals[f'alpha_{n}'] for n in ['P3','P4','P5']) / 3
        vals['beta_mean'] = sum(vals[f'beta_{n}'] for n in ['P3','P4','P5']) / 3
        return vals

    def log_alpha_beta_values(self, stage='val', iter_num=None):
        if not self.log_alpha:
            return
        if stage == 'val' and self._alpha_logged_this_epoch:
            return
        v = self.get_alpha_beta_values()
        logger = MMLogger.get_current_instance()
        prefix = f'[{self._variant_name}]'
        if stage == 'train' and iter_num is not None:
            logger.info(
                f"{prefix} Iter [{iter_num}] "
                f"Alpha: P3={v['alpha_P3']:.4f}, P4={v['alpha_P4']:.4f}, "
                f"P5={v['alpha_P5']:.4f}, Mean={v['alpha_mean']:.4f} | "
                f"Beta: P3={v['beta_P3']:.4f}, P4={v['beta_P4']:.4f}, "
                f"P5={v['beta_P5']:.4f}, Mean={v['beta_mean']:.4f}"
            )
        else:
            logger.info(
                f"{prefix} Val "
                f"Alpha: P3={v['alpha_P3']:.4f}, P4={v['alpha_P4']:.4f}, "
                f"P5={v['alpha_P5']:.4f}, Mean={v['alpha_mean']:.4f} | "
                f"Beta: P3={v['beta_P3']:.4f}, P4={v['beta_P4']:.4f}, "
                f"P5={v['beta_P5']:.4f}, Mean={v['beta_mean']:.4f}"
            )
        if stage == 'val':
            self._alpha_logged_this_epoch = True

    def reset_alpha_log_flag(self):
        self._alpha_logged_this_epoch = False
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
                self.log_alpha_beta_values('train', self._train_iter_count)
        else:
            self.log_alpha_beta_values('val')

        if isinstance(txt_feats, tuple):
            txt_feats, _ = txt_feats

        B = rgb_feats[0].size(0)
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(0).expand(B, -1, -1)
        elif txt_feats.dim() == 3 and txt_feats.size(0) != B:
            txt_feats = txt_feats[:1].expand(B, -1, -1) \
                if txt_feats.size(0) == 1 else txt_feats[:B]

        ir_corrected_feats = []
        for i in range(self.num_levels):
            a = self.correction_alphas[i]
            b = self.enhancement_betas[i]
            out = self.correction_modules[i](
                rgb_feats[i], ir_feats[i], txt_feats, a, b
            )
            ir_corrected_feats.append(out)

        return rgb_feats, tuple(ir_corrected_feats)
