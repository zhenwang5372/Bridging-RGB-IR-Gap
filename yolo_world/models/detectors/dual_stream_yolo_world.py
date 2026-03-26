# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream RGB-IR YOLO-World Detector
from typing import List, Tuple, Union

import torch
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class DualStreamRGBIRBackbone(BaseModule):
    """
    Dual-stream backbone that combines RGB and IR feature extraction with fusion.
    
    This module wraps:
        - RGB backbone: Main RGB feature extractor (e.g., YOLOv8 CSPDarknet)
        - IR backbone: Lightweight IR feature extractor (LiteFFTIRBackbone)
        - Fusion modules: Cross-modal fusion at each FPN level
    
    Args:
        rgb_backbone (dict): Config dict for RGB backbone.
        ir_backbone (dict): Config dict for IR backbone.
        fusion_module (dict): Config dict for fusion module.
        frozen_stages (int): Stages to be frozen. Defaults to -1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    
    def __init__(self,
                 rgb_backbone: ConfigType,
                 ir_backbone: ConfigType,
                 fusion_module: ConfigType,
                 frozen_stages: int = -1,
                 init_cfg: OptMultiConfig = None):
        super(DualStreamRGBIRBackbone, self).__init__(init_cfg)
        
        self.frozen_stages = frozen_stages
        
        # Build RGB backbone
        self.rgb_backbone = MODELS.build(rgb_backbone)
        
        # Build IR backbone
        self.ir_backbone = MODELS.build(ir_backbone)
        
        # Build fusion module
        self.fusion_module = MODELS.build(fusion_module)
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze the parameters of specified stages."""
        if self.frozen_stages >= 0:
            # Freeze RGB backbone stages
            if hasattr(self.rgb_backbone, '_freeze_stages'):
                self.rgb_backbone._freeze_stages()
            
            # Freeze IR backbone stages
            if hasattr(self.ir_backbone, '_freeze_stages'):
                self.ir_backbone._freeze_stages()
    
    def forward(self, 
                img_rgb: Tensor,
                img_ir: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass through dual-stream backbone.
        
        Args:
            img_rgb: RGB input tensor [B, 3, H, W]
            img_ir: IR input tensor [B, 3, H, W]
               
        Returns:
            Tuple of fused feature tensors (P3, P4, P5)
        """
        # Extract RGB features
        rgb_feats = self.rgb_backbone(img_rgb)
        
        # Extract IR features
        ir_feats = self.ir_backbone(img_ir)
        
        # Fuse features at each pyramid level
        fused_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return fused_feats
    
    def train(self, mode: bool = True):
        """Convert the model into training mode while keeping frozen stages."""
        super().train(mode)
        self._freeze_stages()


@MODELS.register_module()
class DualStreamMultiModalYOLOBackbone(BaseModule):
    """
    Dual-stream multi-modal YOLO backbone for RGB-IR + Text fusion.
    
    Combines:
        - RGB image backbone (e.g., YOLOv8CSPDarknet)
        - IR image backbone (LiteFFTIRBackbone)
        - RGB-IR fusion module
        - Text model (e.g., CLIP)
    
    Args:
        image_model (dict): Config for RGB image backbone.
        ir_model (dict): Config for IR image backbone.
        fusion_module (dict): Config for RGB-IR fusion module.
        text_model (dict): Config for text model.
        frozen_stages (int): Stages to be frozen. Defaults to -1.
        with_text_model (bool): Whether to use text model. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    
    def __init__(self,
                 image_model: ConfigType,
                 ir_model: ConfigType,
                 fusion_module: ConfigType,
                 text_model: ConfigType = None,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None):
        super(DualStreamMultiModalYOLOBackbone, self).__init__(init_cfg)
        
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # Build RGB image backbone
        self.image_model = MODELS.build(image_model)
        
        # Build IR backbone
        self.ir_model = MODELS.build(ir_model)
        
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
    
    def forward(self,
                image: Tensor,
                text: List[List[str]],
                img_ir: Tensor = None,
                **kwargs) -> Tuple[Tuple[Tensor], Tensor]:
        """
        Forward pass through dual-stream multi-modal backbone.
        
        Args:
            image: RGB input tensor [B, 3, H, W]
            text: List of text prompts
            img_ir: IR input tensor [B, 3, H, W]. If None, uses image as IR input.
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Tuple of (fused_img_feats, text_feats)
        """
        if img_ir is None:
            img_ir = image
        
        # Extract RGB features
        rgb_feats = self.image_model(image)
        
        # Extract IR features
        ir_feats = self.ir_model(img_ir)
        
        # Fuse RGB and IR features
        img_feats = self.fusion_module(rgb_feats, ir_feats)
        
        # Extract text features
        if text is not None and self.with_text_model and self.text_model is not None:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None
    
    def forward_text(self, text: List[List[str]]) -> Tensor:
        """Forward text only."""
        assert self.with_text_model and self.text_model is not None, \
            "forward_text() requires a text model"
        return self.text_model(text)
    
    def forward_image(self, image: Tensor, img_ir: Tensor = None) -> Tuple[Tensor]:
        """Forward image only (RGB + IR fusion)."""
        if img_ir is None:
            img_ir = image
        
        rgb_feats = self.image_model(image)
        ir_feats = self.ir_model(img_ir)
        fused_feats = self.fusion_module(rgb_feats, ir_feats)
        
        return fused_feats


@MODELS.register_module()
class DualStreamYOLOWorldDetector(YOLODetector):
    """
    Dual-stream RGB-IR YOLO-World Detector.
    
    Extends YOLOWorldDetector to support dual-modal (RGB + IR) input.
    
    Args:
        mm_neck (bool): Whether to use multi-modal neck. Defaults to False.
        num_train_classes (int): Number of training classes. Defaults to 80.
        num_test_classes (int): Number of test classes. Defaults to 80.
    """
    
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 aggregator: ConfigType = None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        self._ir_input = None  # Store IR input for current batch
        
        # Build class dimension aggregator if provided
        if aggregator is not None:
            self.aggregator = MODELS.build(aggregator)
        else:
            self.aggregator = None
    
    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                inputs_ir: Tensor = None,
                **kwargs) -> Union[dict, list, Tensor]:
        """
        Unified forward for dual-stream detection.
        
        Overrides base forward to handle IR input from data preprocessor.
        
        Args:
            inputs: RGB input tensor [B, C, H, W]
            data_samples: Data samples
            mode: Forward mode ('loss', 'predict', 'tensor')
            inputs_ir: IR input tensor [B, C, H, W] (from preprocessor)
            **kwargs: Additional arguments
        """
        # Store IR input for use in loss/predict methods
        self._ir_input = inputs_ir
        
        try:
            if mode == 'loss':
                return self.loss(inputs, data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode.')
        finally:
            # Clear stored IR input after processing
            self._ir_input = None
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        
        # Extract IR input from data_samples
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
        # Extract IR input from data_samples
        img_ir = self._get_ir_input(batch_data_samples)
        
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        
        # txt_feats shape: [B, N, text_dim], so shape[1] is num_classes
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
        # First check stored IR input from forward()
        if self._ir_input is not None:
            ir_input = self._ir_input
            # Don't clear here, let forward() manage it
            return ir_input
        
        if batch_data_samples is None:
            return None
        
        if isinstance(batch_data_samples, dict):
            # Training mode with dict samples
            return batch_data_samples.get('inputs_ir', None)
        elif isinstance(batch_data_samples, list) and len(batch_data_samples) > 0:
            # Inference mode with list of DataSample
            if hasattr(batch_data_samples[0], 'img_ir'):
                ir_list = [ds.img_ir for ds in batch_data_samples]
                return torch.stack(ir_list, dim=0)
        
        return None
    
    def extract_feat(self,
                     batch_inputs: Tensor,
                     batch_data_samples: SampleList,
                     img_ir: Tensor = None) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features from RGB+IR images and text."""
        txt_feats = None
        txt_masks = None
        fused_feats = None  # ⭐ 新增：用于存储Fused特征
        
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
        
        # ⭐ 提取 gt_labels（用于 V5 的 class mask）
        gt_labels = None
        if self.training and batch_data_samples is not None:
            if isinstance(batch_data_samples, dict):
                # 字典格式（yolow_collate）
                # bboxes_labels: [batch_idx, label, x1, y1, x2, y2]
                bboxes_labels = batch_data_samples.get('bboxes_labels', None)
                if bboxes_labels is not None and bboxes_labels.numel() > 0:
                    batch_size = batch_inputs.shape[0]
                    gt_labels = []
                    for b in range(batch_size):
                        mask = bboxes_labels[:, 0] == b
                        labels = bboxes_labels[mask, 1].long()  # 第1列是 label
                        gt_labels.append(labels)
            elif isinstance(batch_data_samples, list):
                # 列表格式（标准 mmdet）
                gt_labels = []
                for data_sample in batch_data_samples:
                    if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'labels'):
                        gt_labels.append(data_sample.gt_instances.labels)
                    else:
                        gt_labels.append(None)
        
        # Forward through backbone
        if hasattr(self.backbone, 'forward_image'):
            # DualStreamMultiModalYOLOBackbone
            if txt_feats is not None:
                img_feats = self.backbone.forward_image(batch_inputs, img_ir)
            else:
                # ⭐ 传递 gt_labels 给 Backbone（用于 V5）
                backbone_output = self.backbone(batch_inputs, texts, img_ir, gt_labels=gt_labels)
                # 处理返回 2 个或 3 个值的情况
                if len(backbone_output) == 3:
                    # ⭐ 修改：保存fused_feats用于aggregator融合
                    img_feats, (txt_feats, txt_masks), fused_feats = backbone_output
                else:
                    img_feats, (txt_feats, txt_masks) = backbone_output
        else:
            # Fallback for standard backbone
            if txt_feats is not None:
                img_feats = self.backbone.forward_image(batch_inputs)
            else:
                backbone_output = self.backbone(batch_inputs, texts)
                if len(backbone_output) == 3:
                    # ⭐ 修改：保存fused_feats用于aggregator融合
                    img_feats, (txt_feats, txt_masks), fused_feats = backbone_output
                else:
                    img_feats, (txt_feats, txt_masks) = backbone_output
        
        # Apply neck
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        # Apply class dimension aggregator if exists
        # This aggregates [B, num_cls, C, H, W] → [B, C, H, W]
        # so that traditional YOLO head can be used
        # ⭐ 修改：传递fused_feats给aggregator进行融合
        if hasattr(self, 'aggregator') and self.aggregator is not None:
            img_feats = self.aggregator(img_feats, fused_feats)
        
        return img_feats, txt_feats, txt_masks
    
    def reparameterize(self, texts: List[List[str]]) -> None:
        """Encode text embeddings into the detector."""
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

