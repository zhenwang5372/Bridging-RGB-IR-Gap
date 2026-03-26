# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Modal Data Preprocessor for RGB-IR Input
from typing import Optional, Sequence, Union

import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import BaseDataElement

from mmyolo.registry import MODELS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


@MODELS.register_module()
class DualModalDataPreprocessor(DetDataPreprocessor):
    """
    Data preprocessor for dual-modal RGB-IR detection.
    
    Handles separate normalization for RGB and IR inputs with different mean/std values.
    
    Args:
        mean (Sequence[float]): Mean values for RGB normalization.
        std (Sequence[float]): Std values for RGB normalization.
        mean_ir (Sequence[float]): Mean values for IR normalization.
        std_ir (Sequence[float]): Std values for IR normalization.
        bgr_to_rgb (bool): Whether to convert BGR to RGB. Defaults to True.
        rgb_to_bgr (bool): Whether to convert RGB to BGR. Defaults to False.
        pad_size_divisor (int): Pad size divisor. Defaults to 32.
        pad_value (float): Pad value. Defaults to 0.
        non_blocking (bool): Whether to use non-blocking. Defaults to True.
    """
    
    def __init__(self,
                 mean: Sequence[float] = (0., 0., 0.),
                 std: Sequence[float] = (255., 255., 255.),
                 mean_ir: Sequence[float] = (0., 0., 0.),
                 std_ir: Sequence[float] = (255., 255., 255.),
                 bgr_to_rgb: bool = True,
                 rgb_to_bgr: bool = False,
                 pad_size_divisor: int = 32,
                 pad_value: Union[float, int] = 0,
                 non_blocking: Optional[bool] = True,
                 **kwargs):
        super().__init__(
            mean=mean,
            std=std,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            non_blocking=non_blocking,
            **kwargs
        )
        
        # Register IR normalization parameters
        self.register_buffer('mean_ir',
                            torch.tensor(mean_ir).view(-1, 1, 1), False)
        self.register_buffer('std_ir',
                            torch.tensor(std_ir).view(-1, 1, 1), False)
    
    def forward(self, data: dict, training: bool = False) -> dict:
        """
        Preprocess both RGB and IR inputs.
        
        Args:
            data (dict): Data sampled from dataloader containing:
                - inputs: RGB images [B, C, H, W]
                - inputs_ir: IR images [B, C, H, W] (optional)
                - data_samples: Data samples
            training (bool): Whether in training mode.
            
        Returns:
            dict: Preprocessed data with normalized RGB and IR inputs.
        """
        if not training:
            # Inference mode - use parent class for RGB
            result = super().forward(data, training)
            
            # Also preprocess IR if present in data
            if 'inputs_ir' in data:
                inputs_ir = data['inputs_ir']
                if isinstance(inputs_ir, torch.Tensor):
                    # Convert to float32 and move to device (critical fix!)
                    inputs_ir = inputs_ir.to(self.device).float()
                    if self._channel_conversion and inputs_ir.shape[1] == 3:
                        inputs_ir = inputs_ir[:, [2, 1, 0], ...]
                    if self._enable_normalize:
                        inputs_ir = (inputs_ir - self.mean_ir) / self.std_ir
                    result['inputs_ir'] = inputs_ir
            
            # Also check data_samples for IR (when using PackDualModalInputs)
            if 'inputs_ir' not in result and 'data_samples' in result:
                data_samples = result['data_samples']
                if isinstance(data_samples, list) and len(data_samples) > 0:
                    # Stack IR images from data_samples
                    ir_list = []
                    for ds in data_samples:
                        if hasattr(ds, 'img_ir') and ds.img_ir is not None:
                            ir_list.append(ds.img_ir)
                    if ir_list:
                        inputs_ir = torch.stack(ir_list, dim=0)
                        inputs_ir = inputs_ir.to(self.device).float()
                        if self._channel_conversion and inputs_ir.shape[1] == 3:
                            inputs_ir = inputs_ir[:, [2, 1, 0], ...]
                        if self._enable_normalize:
                            inputs_ir = (inputs_ir - self.mean_ir) / self.std_ir
                        result['inputs_ir'] = inputs_ir
            
            return result
        
        # Training mode
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples = data['data_samples']
        
        # Handle IR inputs
        inputs_ir = data.get('inputs_ir', None)
        
        assert isinstance(data_samples, dict), \
            f"data_samples should be dict in training, got {type(data_samples)}"
        
        # Preprocess RGB - convert to float if needed
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std
        
        # Preprocess IR - convert to float if needed
        if inputs_ir is not None:
            if inputs_ir.dtype != torch.float32:
                inputs_ir = inputs_ir.float()
            if self._channel_conversion and inputs_ir.shape[1] == 3:
                inputs_ir = inputs_ir[:, [2, 1, 0], ...]
            if self._enable_normalize:
                inputs_ir = (inputs_ir - self.mean_ir) / self.std_ir
        
        # Apply batch augmentations
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)
                # Note: batch_aug should also handle inputs_ir if needed
        
        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'texts': data_samples['texts'],
            'img_metas': img_metas
        }
        
        # Pass IR inputs through data_samples
        if inputs_ir is not None:
            data_samples_output['inputs_ir'] = inputs_ir
        
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        if 'is_detection' in data_samples:
            data_samples_output['is_detection'] = data_samples['is_detection']
        
        return {'inputs': inputs, 'data_samples': data_samples_output}


@MODELS.register_module()
class FLIRDataPreprocessor(DualModalDataPreprocessor):
    """
    Data preprocessor specifically configured for FLIR dataset.
    
    Uses pre-computed mean/std for FLIR RGB and IR images:
        - RGB Mean/Std: [160.05, 162.13, 159.79] / [56.20, 59.30, 63.78]
        - IR Mean/Std: [135.67, 135.67, 135.67] / [64.49, 64.49, 64.49]
    """
    
    def __init__(self,
                 mean: Sequence[float] = (160.05, 162.13, 159.79),
                 std: Sequence[float] = (56.20, 59.30, 63.78),
                 mean_ir: Sequence[float] = (135.67, 135.67, 135.67),
                 std_ir: Sequence[float] = (64.49, 64.49, 64.49),
                 bgr_to_rgb: bool = True,
                 **kwargs):
        super().__init__(
            mean=mean,
            std=std,
            mean_ir=mean_ir,
            std_ir=std_ir,
            bgr_to_rgb=bgr_to_rgb,
            **kwargs
        )

