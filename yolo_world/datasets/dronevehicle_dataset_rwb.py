# Copyright (c) Tencent Inc. All rights reserved.
# DroneVehicle Dataset for RGB-IR Dual-Modal Detection (Remove White Borders Version)
#
# ============================================================================
# 模块说明 (Module Description):
# ============================================================================
# 本文件实现 DroneVehicle 数据集的加载类，专门用于去除白边版本 (640x512)。
#
# 与原版 DroneVehicleDataset 的区别:
#   - 原版 (with_white_borders): 840x712, category_id: bus(1), car(2), freight_car(3), truck(4), van(5)
#   - 本版 (remove_white_borders): 640x512, category_id: car(1), freight_car(2), truck(3), bus(4), van(5)
#
# DroneVehicle 数据集结构 (去除白边版本):
#   data/dronevehicle/
#   ├── images/
#   │   ├── train/trainimg/00001.jpg      # RGB 可见光图像 (640x512)
#   │   ├── train/trainimgr/00001.jpg     # IR 红外图像 (640x512)
#   │   ├── val/valimg/
#   │   ├── val/valimgr/
#   │   ├── test/testimg/
#   │   └── test/testimgr/
#   └── annotations/remove_white_borders/
#       ├── clip/
#       ├── minus/
#       ├── remove_clip_less_5px/
#       └── remove_beyond_borders/
#
# 类别 (remove_white_borders 版本的 category_id 顺序):
#   5类: car(1), freight_car(2), truck(3), bus(4), van(5)
#
# 使用方式:
#   在配置文件中使用 type='DroneVehicleDatasetRWB'
# ============================================================================

import os.path as osp
from typing import Optional

from mmdet.datasets import CocoDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDatasetRWB(CocoDataset):
    """
    DroneVehicle Dataset for RGB-IR vehicle detection (Remove White Borders Version).
    
    This dataset class is specifically designed for the cropped 640x512 images
    with white borders removed.
    
    Key features:
        - RGB and IR images are in separate directories with 'r' suffix for IR
          (e.g., trainimg/ vs trainimgr/)
        - Filenames are identical, only directory differs
        - 5 classes: car, freight_car, truck, bus, van
        - Image resolution: 640×512 (cropped from original 840×712)
    
    IMPORTANT: Category ID order is different from with_white_borders version!
        - with_white_borders: bus(1), car(2), freight_car(3), truck(4), van(5)
        - remove_white_borders: car(1), freight_car(2), truck(3), bus(4), van(5)
    
    Directory structure:
        data/dronevehicle/
        ├── images/train/trainimg/00001.jpg (RGB, 640x512)
        ├── images/train/trainimgr/00001.jpg (IR, 640x512)
        ├── annotations/remove_white_borders/clip/DV_train_r_clip.json
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
    # 顺序必须与JSON文件中的category_id一致 (remove_white_borders版本):
    #   id=1: car, id=2: freight_car, id=3: truck, id=4: bus, id=5: van
    METAINFO = {
        'classes': ('car', 'freight_car', 'truck', 'bus', 'van'),
        'palette': [
            (0, 255, 0),     # car - green (id=1)
            (0, 0, 255),     # freight_car - blue (id=2)
            (255, 255, 0),   # truck - yellow (id=3)
            (220, 20, 60),   # bus - red (id=4)
            (255, 0, 255),   # van - magenta (id=5)
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
class MultiModalDroneVehicleDatasetRWB(DroneVehicleDatasetRWB):
    """
    Multi-modal DroneVehicle Dataset with text prompts support (Remove White Borders Version).
    
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
