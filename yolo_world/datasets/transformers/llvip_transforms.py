# Copyright (c) Tencent Inc. All rights reserved.
# LLVIP-specific Transforms for RGB-IR Dual-Modal Detection
#
# ============================================================================
# 模块说明 (Module Description):
# ============================================================================
# 本文件实现 LLVIP 数据集专用的图像加载 Transform。
#
# 与 FLIR LoadIRImageFromFile 的区别:
#   - FLIR: 通过文件名后缀替换推断 IR 路径
#           *_RGB.jpg -> *_PreviewData.jpeg
#   - LLVIP: 通过目录替换推断 IR 路径
#           visible/train/000001.jpg -> infrared/train/000001.jpg
#
# 使用方式:
#   在配置文件的 pipeline 中使用:
#   dict(type='LoadIRImageFromFileLLVIP', rgb_dir='visible', ir_dir='infrared')
#
# 注意: 其他同步增强操作 (SyncMosaic, SyncRandomAffine, SyncLetterResize 等)
#       可直接复用 sync_rgb_ir_transforms.py 中的实现，无需修改。
# ============================================================================

import os.path as osp
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadIRImageFromFileLLVIP(BaseTransform):
    """
    Load IR image for LLVIP dataset based on RGB image path.
    
    LLVIP uses directory-based separation for RGB and IR images:
        - RGB: visible/train/000001.jpg
        - IR:  infrared/train/000001.jpg
    
    This transform infers IR path by replacing directory name.
    
    Args:
        rgb_dir (str): RGB image directory name. Defaults to 'visible'.
        ir_dir (str): IR image directory name. Defaults to 'infrared'.
        to_float32 (bool): Whether to convert to float32. Defaults to False.
        color_type (str): Color type for imread. Defaults to 'color'.
        imdecode_backend (str): Backend for imdecode. Defaults to 'cv2'.
        ignore_empty (bool): Whether to ignore missing IR images and use 
            RGB as fallback. Defaults to True.
    """
    
    def __init__(self,
                 rgb_dir: str = 'visible',
                 ir_dir: str = 'infrared',
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = True):
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
    
    def _get_ir_path(self, rgb_path: str) -> str:
        """Infer IR image path from RGB image path.
        
        Replace directory name: visible -> infrared
        
        Args:
            rgb_path (str): Path to RGB image.
            
        Returns:
            str: Path to corresponding IR image.
        """
        # Replace directory: visible -> infrared
        # Handle both forward slash and backslash
        ir_path = rgb_path.replace(f'/{self.rgb_dir}/', f'/{self.ir_dir}/')
        ir_path = ir_path.replace(f'\\{self.rgb_dir}\\', f'\\{self.ir_dir}\\')
        
        return ir_path
    
    def transform(self, results: dict) -> Optional[dict]:
        """Load IR image.
        
        Args:
            results (dict): Result dict containing 'img_path'.
            
        Returns:
            dict: Result dict with added 'img_ir', 'img_ir_path', 
                  and 'img_ir_shape' keys.
        """
        rgb_path = results['img_path']
        ir_path = self._get_ir_path(rgb_path)
        
        try:
            img_ir = mmcv.imread(ir_path, flag=self.color_type, 
                                backend=self.imdecode_backend)
            
            if img_ir is None:
                raise FileNotFoundError(f"IR image is None: {ir_path}")
                
        except Exception as e:
            if self.ignore_empty:
                # Use RGB image as fallback for IR
                img_ir = results.get('img', None)
                if img_ir is None:
                    img_ir = mmcv.imread(rgb_path, flag=self.color_type,
                                        backend=self.imdecode_backend)
                img_ir = img_ir.copy()
                print(f"Warning: IR image not found, using RGB as fallback: {ir_path}")
            else:
                raise FileNotFoundError(f"IR image not found: {ir_path}") from e
        
        if self.to_float32:
            img_ir = img_ir.astype(np.float32)
        
        results['img_ir'] = img_ir
        results['img_ir_path'] = ir_path
        results['img_ir_shape'] = img_ir.shape[:2]
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(rgb_dir={self.rgb_dir}, '
        repr_str += f'ir_dir={self.ir_dir}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f'color_type={self.color_type}, '
        repr_str += f'ignore_empty={self.ignore_empty})'
        return repr_str
