# Copyright (c) Tencent Inc. All rights reserved.
# DroneVehicle Dataset for RGB-IR Dual-Modal Detection (Remove White Borders Version)
#
# ============================================================================
# 模块说明 (Module Description):
# ============================================================================
# 本文件实现 DroneVehicle 数据集的加载类（去白边版本），用于 RGB-IR 双模态目标检测训练。
#
# 与 dronevehicle_dataset.py 的区别:
#   - 图像分辨率: 640×512 (去除白边后)
#   - category_id 顺序不同:
#     - with_white_borders: bus(1), car(2), freight_car(3), truck(4), van(5)
#     - remove_white_borders: car(1), freight_car(2), truck(3), bus(4), van(5)
#
# DroneVehicle 数据集结构 (去白边版本):
#   data/dronevehicle/
#   ├── images/
#   │   ├── train/trainimg/00001.jpg      # RGB 可见光图像 (640×512)
#   │   ├── train/trainimgr/00001.jpg     # IR 红外图像 (640×512)
#   │   ├── val/valimg/
#   │   ├── val/valimgr/
#   │   ├── test/testimg/
#   │   └── test/testimgr/
#   └── annotations/remove_white_borders/
#       ├── clip/                         # Clip版本
#       ├── minus/                        # 原始版本
#       ├── remove_clip_less_5px/         # 移除<5px的版本 (推荐)
#       └── remove_beyond_borders/        # 移除超界的版本
#
# 类别 (category_id 顺序):
#   id=1: car, id=2: freight_car, id=3: truck, id=4: bus, id=5: van
#
# 使用方式:
#   在配置文件中使用 type='DroneVehicleRWBDataset'
# ============================================================================

import os.path as osp
from typing import Optional

from mmdet.datasets import CocoDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleRWBDataset(CocoDataset):
    """
    DroneVehicle Dataset for RGB-IR vehicle detection (Remove White Borders version).
    
    This dataset class is specifically designed for the remove_white_borders version
    of DroneVehicle dataset, with different category_id mapping.
    
    Key features:
        - RGB and IR images are in separate directories with 'r' suffix for IR
          (e.g., trainimg/ vs trainimgr/)
        - Filenames are identical, only directory differs
        - 5 classes: car, freight_car, truck, bus, van (different order from original)
        - Image resolution: 640×512 (white borders removed)
    
    Category ID mapping (remove_white_borders version):
        id=1: car, id=2: freight_car, id=3: truck, id=4: bus, id=5: van
    
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
