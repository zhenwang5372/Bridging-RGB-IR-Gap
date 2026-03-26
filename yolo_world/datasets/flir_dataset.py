# Copyright (c) Tencent Inc. All rights reserved.
# FLIR Dataset for RGB-IR Dual-Modal Detection
import os.path as osp
from typing import List, Optional, Sequence

from mmdet.datasets import CocoDataset

from mmyolo.registry import DATASETS

# Parameters that are YOLO-specific and not supported by CocoDataset
YOLO_SPECIFIC_PARAMS = ['batch_shapes_cfg']


@DATASETS.register_module()
class FLIRDataset(CocoDataset):
    """
    FLIR Dataset for RGB-IR thermal detection.
    
    Supports loading paired RGB and IR images for dual-modal detection.
    
    RGB and IR paths follow the naming convention:
        - RGB: *_RGB.jpg
        - IR: *_PreviewData.jpeg
    
    Args:
        ann_file (str): Annotation file path.
        data_root (str): Data root path.
        data_prefix (dict): Prefix for data (img for RGB, img_ir for IR).
        ir_suffix (str): IR image suffix. Defaults to '_PreviewData.jpeg'.
        rgb_suffix (str): RGB image suffix. Defaults to '_RGB.jpg'.
        metainfo (dict): Meta information. Defaults to None.
    """
    
    # FLIR dataset classes (adjust based on actual dataset)
    METAINFO = {
        'classes': ('car', 'person', 'bicycle', 'dog'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 60, 100)]
    }
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img=''),
                 ir_suffix: str = '_PreviewData.jpeg',
                 rgb_suffix: str = '_RGB.jpg',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.ir_suffix = ir_suffix
        self.rgb_suffix = rgb_suffix
        
        # Filter out YOLO-specific parameters that CocoDataset doesn't support
        for param in YOLO_SPECIFIC_PARAMS:
            kwargs.pop(param, None)
        
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            metainfo=metainfo,
            **kwargs
        )
    
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation info.
        
        Adds IR image path based on RGB image path.
        """
        data_info = super().parse_data_info(raw_data_info)
        
        # Infer IR image path from RGB path
        img_path = data_info.get('img_path', '')
        if img_path:
            ir_path = self._get_ir_path(img_path)
            data_info['img_ir_path'] = ir_path
        
        return data_info
    
    def _get_ir_path(self, rgb_path: str) -> str:
        """Infer IR image path from RGB image path."""
        if self.rgb_suffix in rgb_path:
            ir_path = rgb_path.replace(self.rgb_suffix, self.ir_suffix)
        else:
            # Fallback: change extension
            base, _ = osp.splitext(rgb_path)
            ir_path = base + self.ir_suffix
        return ir_path


@DATASETS.register_module()
class FLIRAlignedDataset(FLIRDataset):
    """
    FLIR Aligned Dataset for the aligned RGB-IR pairs.
    
    Uses the aligned dataset structure:
        - RGB: JPEGImages/*_RGB.jpg
        - IR: JPEGImages/*_PreviewData.jpeg
    """
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img='_hf_cache/_extracted/aligned/align/JPEGImages/'),
                 ir_suffix: str = '_PreviewData.jpeg',
                 rgb_suffix: str = '_RGB.jpg',
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            ir_suffix=ir_suffix,
            rgb_suffix=rgb_suffix,
            **kwargs
        )


@DATASETS.register_module()
class MultiModalFLIRDataset(FLIRDataset):
    """
    Multi-modal FLIR Dataset with text prompts support.
    
    Combines FLIR RGB-IR loading with YOLO-World text prompts.
    
    Args:
        class_text_path (str): Path to class text descriptions.
    """
    
    def __init__(self,
                 class_text_path: str = '',
                 **kwargs):
        self.class_text_path = class_text_path
        super().__init__(**kwargs)
        
        # Load class texts if provided
        if class_text_path:
            self._load_class_texts()
    
    def _load_class_texts(self):
        """Load class text descriptions."""
        import json
        if osp.exists(self.class_text_path):
            with open(self.class_text_path, 'r') as f:
                self.class_texts = json.load(f)
        else:
            # Default to class names
            self.class_texts = {
                cls: [cls] for cls in self.metainfo['classes']
            }
    
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse data info with text support."""
        data_info = super().parse_data_info(raw_data_info)
        
        # Add text prompts if available
        if hasattr(self, 'class_texts'):
            data_info['texts'] = self.class_texts
        
        return data_info

