# Copyright (c) Tencent Inc. All rights reserved.
# IR-Only YOLO-World Detector (Baseline for comparison)
"""
IR-Only 单模态检测器

用于与 RGB-IR 双模态检测器进行对比实验。
只使用 IR 图像 + Text，不使用 RGB 图像。

数据流程:
    IR Image ──→ LiteFFTIRBackbone ──→ [64, 128, 256]
                                            │
                                            ▼
                               ChannelAlign + Interpolate (在 IROnlyBackbone 内)
                                            │
                                            ▼
                                      [128, 256, 512]
                                            │
                                            ▼
                               SimpleChannelAlign (pass-through) ──→ YOLOWorldHead
    
    Text ──────→ CLIP Encoder ────────────────────────────────────→ (cls contrast)
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class IROnlyBackbone(BaseModule):
    """
    IR-Only Backbone with Text Model.
    
    只使用 IR backbone 提取特征，然后通过 Channel Align 对齐到
    标准 YOLO Head 需要的通道数 [128, 256, 512]。
    
    Args:
        ir_model (dict): Config for IR backbone (LiteFFTIRBackbone).
        text_model (dict): Config for text model (CLIP).
        ir_channels (list): IR backbone output channels. Default: [64, 128, 256].
        out_channels (list): Target output channels. Default: [128, 256, 512].
        target_sizes (list): Target spatial sizes for each level. 
                            Default: [(160, 160), (80, 80), (40, 40)] for 1280x1280 input.
        with_text_model (bool): Whether to use text model. Default: True.
        frozen_stages (int): Stages to be frozen. Default: -1.
        init_cfg (dict, optional): Initialization config.
    """
    
    def __init__(self,
                 ir_model: ConfigType,
                 text_model: ConfigType = None,
                 ir_channels: List[int] = [64, 128, 256],
                 out_channels: List[int] = [128, 256, 512],
                 target_sizes: List[Tuple[int, int]] = None,
                 with_text_model: bool = True,
                 frozen_stages: int = -1,
                 init_cfg: OptMultiConfig = None):
        super(IROnlyBackbone, self).__init__(init_cfg)
        
        self.ir_channels = ir_channels
        self.out_channels = out_channels
        self.target_sizes = target_sizes  # 如果为 None，则使用 RGB backbone 的输出尺寸
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # Build IR backbone
        self.ir_model = MODELS.build(ir_model)
        
        # Build text model
        if self.with_text_model and text_model is not None:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        
        # Build channel alignment layers (IR channels → RGB channels)
        # P3: 64 → 128, P4: 128 → 256, P5: 256 → 512
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            for in_ch, out_ch in zip(ir_channels, out_channels)
        ])
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze the parameters of specified stages."""
        if self.frozen_stages >= 0:
            if hasattr(self.ir_model, '_freeze_stages'):
                self.ir_model._freeze_stages()
    
    def forward(self,
                image: Tensor,
                text: List[List[str]],
                img_ir: Tensor = None) -> Tuple[Tuple[Tensor], Tuple[Tensor, Tensor]]:
        """
        Forward pass through IR-only backbone.
        
        Args:
            image: RGB input tensor [B, 3, H, W] (不使用，但需要获取目标尺寸)
            text: List of text prompts
            img_ir: IR input tensor [B, 3, H, W]
            
        Returns:
            Tuple of (aligned_ir_feats, (text_feats, text_masks))
        """
        if img_ir is None:
            raise ValueError("IR input is required for IROnlyBackbone")
        
        # Extract IR features: [P3, P4, P5] with channels [64, 128, 256]
        ir_feats = self.ir_model(img_ir)
        
        # 获取目标空间尺寸 (从 RGB 图像推断)
        B, _, H, W = image.shape
        # 标准 YOLOv8 的下采样率: P3=8, P4=16, P5=32
        target_sizes = [
            (H // 8, W // 8),    # P3
            (H // 16, W // 16),  # P4
            (H // 32, W // 32),  # P5
        ]
        
        # Channel align + spatial interpolate
        aligned_feats = []
        for i, (ir_feat, align_layer, target_size) in enumerate(
            zip(ir_feats, self.channel_align, target_sizes)
        ):
            # Channel align: [64, 128, 256] → [128, 256, 512]
            feat = align_layer(ir_feat)
            
            # Spatial interpolate if sizes don't match
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            
            aligned_feats.append(feat)
        
        aligned_feats = tuple(aligned_feats)
        
        # Extract text features
        if self.text_model is not None and text is not None:
            txt_feats = self.text_model(text)
            if isinstance(txt_feats, dict):
                txt_feats = txt_feats['text_feats']
                txt_masks = txt_feats.get('text_masks', None)
            elif isinstance(txt_feats, tuple):
                txt_feats, txt_masks = txt_feats
            else:
                txt_masks = None
        else:
            txt_feats = None
            txt_masks = None
        
        return aligned_feats, (txt_feats, txt_masks)
    
    def forward_text(self, texts: List[List[str]]) -> Tensor:
        """Forward pass for text only."""
        if self.text_model is not None:
            txt_feats = self.text_model(texts)
            if isinstance(txt_feats, tuple):
                txt_feats = txt_feats[0]
            return txt_feats
        return None
    
    def forward_image(self, image: Tensor, img_ir: Tensor = None) -> Tuple[Tensor]:
        """Forward pass for image only (used when text_feats are cached)."""
        if img_ir is None:
            raise ValueError("IR input is required for IROnlyBackbone")
        
        # Extract IR features
        ir_feats = self.ir_model(img_ir)
        
        # 获取目标空间尺寸
        B, _, H, W = image.shape
        target_sizes = [
            (H // 8, W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]
        
        # Channel align + spatial interpolate
        aligned_feats = []
        for ir_feat, align_layer, target_size in zip(
            ir_feats, self.channel_align, target_sizes
        ):
            feat = align_layer(ir_feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            aligned_feats.append(feat)
        
        return tuple(aligned_feats)
    
    def train(self, mode: bool = True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


@MODELS.register_module()
class IROnlyYOLOWorldDetector(YOLODetector):
    """
    IR-Only YOLO-World Detector.
    
    Baseline detector using only IR images + Text, without RGB images.
    Used for ablation study to compare with RGB-IR dual-modal detection.
    
    Args:
        mm_neck (bool): Whether to use multi-modal neck. Default: False.
        num_train_classes (int): Number of training classes. Default: 80.
        num_test_classes (int): Number of test classes. Default: 80.
    """
    
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        self._ir_input = None
    
    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                inputs_ir: Tensor = None,
                **kwargs) -> Union[dict, list, Tensor]:
        """
        Unified forward for IR-only detection.
        
        Args:
            inputs: RGB input tensor [B, C, H, W] (used only for size reference)
            data_samples: Data samples
            mode: Forward mode ('loss', 'predict', 'tensor')
            inputs_ir: IR input tensor [B, C, H, W]
        """
        self._ir_input = inputs_ir
        
        try:
            if mode == 'loss':
                return self.loss(inputs, data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}".')
        finally:
            self._ir_input = None
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        
        img_ir = self._get_ir_input(batch_data_samples)
        
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        return losses
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs."""
        img_ir = self._get_ir_input(batch_data_samples)
        
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        
        self.bbox_head.num_classes = txt_feats.shape[1] if txt_feats is not None and txt_feats.dim() == 3 else self.num_test_classes
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)
        
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process."""
        img_ir = self._get_ir_input(batch_data_samples)
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results
    
    def _get_ir_input(self, batch_data_samples) -> Tensor:
        """Extract IR input tensor from batch_data_samples or stored input."""
        if self._ir_input is not None:
            return self._ir_input
        
        if batch_data_samples is None:
            return None
        
        if isinstance(batch_data_samples, dict):
            return batch_data_samples.get('inputs_ir', None)
        elif isinstance(batch_data_samples, list) and len(batch_data_samples) > 0:
            if hasattr(batch_data_samples[0], 'img_ir'):
                ir_list = [ds.img_ir for ds in batch_data_samples]
                return torch.stack(ir_list, dim=0)
        
        return None
    
    def extract_feat(self,
                     batch_inputs: Tensor,
                     batch_data_samples: SampleList,
                     img_ir: Tensor = None) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features from IR images and text."""
        txt_feats = None
        txt_masks = None
        
        # Get text prompts
        if batch_data_samples is None:
            texts = getattr(self, 'texts', None)
            txt_feats = getattr(self, 'text_feats', None)
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            texts = None
        
        # Forward through backbone (IR-Only)
        if txt_feats is not None:
            # Use cached text features
            img_feats = self.backbone.forward_image(batch_inputs, img_ir)
        else:
            # Full forward with text encoding
            img_feats, (txt_feats, txt_masks) = self.backbone(
                batch_inputs, texts, img_ir)
        
        # Apply neck
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats, txt_feats, txt_masks
    
    def reparameterize(self, texts: List[List[str]]) -> None:
        """Encode text embeddings into the detector."""
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)
