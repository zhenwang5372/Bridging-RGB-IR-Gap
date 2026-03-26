# Copyright (c) Tencent Inc. All rights reserved.
# KAIST Dataset for RGB-IR Dual-Modal Pedestrian Detection
import os.path as osp
from typing import Optional

from mmdet.datasets import CocoDataset
from mmyolo.registry import DATASETS

# Parameters that are YOLO-specific and not supported by CocoDataset
YOLO_SPECIFIC_PARAMS = ['batch_shapes_cfg']


@DATASETS.register_module()
class KAISTDataset(CocoDataset):
    """
    KAIST Dataset for RGB-IR thermal pedestrian detection.
    
    Supports loading paired RGB (visible) and IR (lwir) images for dual-modal detection.
    
    KAIST数据集结构:
        训练集:
            - RGB: train/visible/{basename}.jpg
            - IR:  train/lwir/{basename}.jpg
        测试集:
            - RGB: test/kaist_test_visible/{basename}_visible.png
            - IR:  test/kaist_test_lwir/{basename}_lwir.png
    
    JSON文件中的file_name只包含basename（如 set06_V000_I00019）
    
    Args:
        ann_file (str): Annotation file path.
        data_root (str): Data root path.
        data_prefix (dict): Prefix for data.
        visible_dir (str): Visible/RGB image directory name.
        lwir_dir (str): LWIR/IR image directory name.
        visible_suffix (str): Visible image suffix (for test set).
        lwir_suffix (str): LWIR image suffix (for test set).
        img_ext (str): Image extension.
        metainfo (dict): Meta information.
    """
    
    # KAIST dataset class - only person
    METAINFO = {
        'classes': ('person',),
        'palette': [(220, 20, 60)]
    }
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img=''),
                 visible_dir: str = 'visible',
                 lwir_dir: str = 'lwir',
                 visible_suffix: str = '',
                 lwir_suffix: str = '',
                 img_ext: str = '.jpg',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.visible_dir = visible_dir
        self.lwir_dir = lwir_dir
        self.visible_suffix = visible_suffix
        self.lwir_suffix = lwir_suffix
        self.img_ext = img_ext
        
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
        
        Adds both RGB and IR image paths based on file_name (basename).
        """
        # First call parent's parse_data_info
        data_info = super().parse_data_info(raw_data_info)
        
        # Get img_path from parsed data_info
        # 父类返回的img_path格式是: data_root/data_prefix/file_name
        img_path = data_info.get('img_path', '')
        
        if img_path:
            # Get the basename from img_path (去除扩展名)
            basename = osp.splitext(osp.basename(img_path))[0]
            
            # Get the directory part (data_root/data_prefix)
            # 父类已经构建好的路径格式是: data_root/data_prefix/file_name
            # 我们需要获取 data_root/data_prefix 部分
            parent_dir = osp.dirname(img_path)  # data_root/data_prefix
            
            # Build visible (RGB) image path
            visible_filename = f"{basename}{self.visible_suffix}{self.img_ext}"
            visible_path = osp.join(parent_dir, self.visible_dir, visible_filename)
            
            # Build LWIR (IR) image path
            lwir_filename = f"{basename}{self.lwir_suffix}{self.img_ext}"
            lwir_path = osp.join(parent_dir, self.lwir_dir, lwir_filename)
            
            # Update paths
            data_info['img_path'] = visible_path
            data_info['img_ir_path'] = lwir_path
        
        return data_info


@DATASETS.register_module()
class KAISTTrainDataset(KAISTDataset):
    """
    KAIST Training Dataset.
    
    训练集结构:
        - RGB: train/visible/{basename}.jpg
        - IR:  train/lwir/{basename}.jpg
    """
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img='train'),
                 visible_dir: str = 'visible',
                 lwir_dir: str = 'lwir',
                 img_ext: str = '.jpg',
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            visible_dir=visible_dir,
            lwir_dir=lwir_dir,
            visible_suffix='',
            lwir_suffix='',
            img_ext=img_ext,
            **kwargs
        )


@DATASETS.register_module()
class KAISTTestDataset(KAISTDataset):
    """
    KAIST Test Dataset.
    
    测试集结构:
        - RGB: test/kaist_test_visible/{basename}_visible.png
        - IR:  test/kaist_test_lwir/{basename}_lwir.png
    """
    
    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: dict = dict(img='test'),
                 visible_dir: str = 'kaist_test_visible',
                 lwir_dir: str = 'kaist_test_lwir',
                 visible_suffix: str = '_visible',
                 lwir_suffix: str = '_lwir',
                 img_ext: str = '.png',
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            visible_dir=visible_dir,
            lwir_dir=lwir_dir,
            visible_suffix=visible_suffix,
            lwir_suffix=lwir_suffix,
            img_ext=img_ext,
            **kwargs
        )
