# Copyright (c) Tencent Inc. All rights reserved.
# DroneVehicle Dataset for RGB-IR Dual-Modal Detection
#
# ============================================================================
# 模块说明 (Module Description):
# ============================================================================
# 本文件实现 DroneVehicle 数据集的加载类，用于 RGB-IR 双模态目标检测训练。
#
# DroneVehicle 数据集结构:
#   data/dronevehicle/
#   ├── train/
#   │   ├── trainimg/00001.jpg      # RGB 可见光图像
#   │   ├── trainimgr/00001.jpg     # IR 红外图像
#   │   └── train.json              # COCO格式标注
#   ├── val/
#   │   ├── valimg/00001.jpg
#   │   ├── valimgr/00001.jpg
#   │   └── val.json
#   └── test/
#       ├── testimg/00001.jpg
#       ├── testimgr/00001.jpg
#       └── test.json
#
# 与 FLIR/LLVIP 数据集的区别:
#   - FLIR: RGB/IR 通过文件名后缀区分 (*_RGB.jpg / *_PreviewData.jpeg)
#   - LLVIP: RGB/IR 通过目录区分 (visible/ / infrared/)，文件名相同
#   - DroneVehicle: RGB/IR 通过目录后缀区分 (trainimg/ / trainimgr/)，文件名相同
#
# 类别:
#   5类: bus, car, feright_car, truck, van
#
# 使用方式:
#   在配置文件中使用 type='DroneVehicleDataset'
# ============================================================================

import os.path as osp
from typing import Optional

from mmdet.datasets import CocoDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDataset(CocoDataset):
    """
    DroneVehicle Dataset for RGB-IR vehicle detection from drone perspective.
    
    DroneVehicle dataset contains paired visible and infrared images 
    captured from drones for vehicle detection.
    
    Key features:
        - RGB and IR images are in separate directories with 'r' suffix for IR
          (e.g., trainimg/ vs trainimgr/)
        - Filenames are identical, only directory differs
        - 5 classes: bus, car, feright_car, truck, van
        - Image resolution: 840×712
    
    Directory structure:
        data/dronevehicle/
        ├── train/trainimg/00001.jpg (RGB)
        ├── train/trainimgr/00001.jpg (IR)
        ├── train/train.json
        └── ...
    
    Args:
        ann_file (str): Annotation file path.
        data_root (str): Data root path.
        data_prefix (dict): Prefix for data (img for RGB).
        rgb_dir (str): RGB image directory name. Defaults to 'trainimg'.
        ir_dir (str): IR image directory name. Defaults to 'trainimgr'.
        metainfo (dict): Meta information. Defaults to None.
    """
    
    # DroneVehicle dataset has 5 vehicle classes
    METAINFO = {
        'classes': ('bus', 'car', 'feright_car', 'truck', 'van'),
        'palette': [
            (220, 20, 60),   # bus - red
            (0, 255, 0),     # car - green  
            (0, 0, 255),     # feright_car - blue
            (255, 255, 0),   # truck - yellow
            (255, 0, 255),   # van - magenta
        ]
    }
    
    # Parameters that are YOLO-specific and not supported by CocoDataset
    YOLO_SPECIFIC_PARAMS = ['batch_shapes_cfg']
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img=''),
                 rgb_dir: str = 'trainimg',
                 ir_dir: str = 'trainimgr',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        
        # Filter out YOLO-specific parameters that CocoDataset doesn't support
        for param in self.YOLO_SPECIFIC_PARAMS:
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
        
        Adds IR image path based on RGB image path by replacing directory.
        
        Args:
            raw_data_info (dict): Raw annotation info from COCO format.
            
        Returns:
            dict: Parsed data info with added 'img_ir_path' key.
        """
        data_info = super().parse_data_info(raw_data_info)
        
        # Infer IR image path from RGB path by directory replacement
        img_path = data_info.get('img_path', '')
        if img_path:
            ir_path = self._get_ir_path(img_path)
            data_info['img_ir_path'] = ir_path
        
        return data_info
    
    def _get_ir_path(self, rgb_path: str) -> str:
        """Infer IR image path from RGB image path.
        
        DroneVehicle uses directory-based separation with 'r' suffix for IR:
            trainimg/00001.jpg -> trainimgr/00001.jpg
            valimg/00001.jpg -> valimgr/00001.jpg
            testimg/00001.jpg -> testimgr/00001.jpg
        
        Args:
            rgb_path (str): Path to RGB image.
            
        Returns:
            str: Path to corresponding IR image.
        """
        # Replace directory: trainimg -> trainimgr, valimg -> valimgr, etc.
        # Handle both forward slash and backslash for cross-platform
        ir_path = rgb_path.replace(f'/{self.rgb_dir}/', f'/{self.ir_dir}/')
        ir_path = ir_path.replace(f'\\{self.rgb_dir}\\', f'\\{self.ir_dir}\\')
        
        return ir_path


@DATASETS.register_module()
class MultiModalDroneVehicleDataset(DroneVehicleDataset):
    """
    Multi-modal DroneVehicle Dataset with text prompts support.
    
    Combines DroneVehicle RGB-IR loading with YOLO-World text prompts.
    
    Args:
        class_text_path (str): Path to class text descriptions JSON file.
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
        """Load class text descriptions from JSON file."""
        import json
        if osp.exists(self.class_text_path):
            with open(self.class_text_path, 'r') as f:
                self.class_texts = json.load(f)
        else:
            # Default to class names if file not found
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
