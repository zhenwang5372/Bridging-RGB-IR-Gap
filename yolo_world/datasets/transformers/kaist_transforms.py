# Copyright (c) Tencent Inc. All rights reserved.
# KAIST Dataset Transforms for RGB-IR Dual-Modal Detection
import os.path as osp
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadKAISTIRImageFromFile(BaseTransform):
    """
    Load IR image for KAIST dataset.
    
    KAIST数据集的IR图像路径已经在KAISTDataset的parse_data_info中设置好了，
    这里只需要根据img_ir_path加载图像即可。
    
    Args:
        to_float32 (bool): Whether to convert to float32. Defaults to False.
        color_type (str): Color type for imread. Defaults to 'color'.
        imdecode_backend (str): Backend for imdecode. Defaults to 'cv2'.
        ignore_empty (bool): Whether to ignore empty IR images. Defaults to True.
    """
    
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
    
    def transform(self, results: dict) -> Optional[dict]:
        """Load IR image.
        
        Args:
            results (dict): Result dict containing 'img_ir_path'.
            
        Returns:
            dict: Result dict with 'img_ir' and 'img_ir_shape' keys.
        """
        ir_path = results.get('img_ir_path', None)
        
        if ir_path is None:
            if self.ignore_empty:
                # Fallback: use RGB image as IR
                img_ir = results.get('img', None)
                if img_ir is not None:
                    img_ir = img_ir.copy()
                else:
                    raise ValueError("No img_ir_path and no img available")
            else:
                raise ValueError("img_ir_path not found in results")
        else:
            try:
                img_ir = mmcv.imread(ir_path, flag=self.color_type,
                                    backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    # Fallback: use RGB image
                    img_ir = results.get('img', None)
                    if img_ir is not None:
                        img_ir = img_ir.copy()
                    else:
                        raise FileNotFoundError(f"IR image not found: {ir_path}") from e
                else:
                    raise FileNotFoundError(f"IR image not found: {ir_path}") from e
        
        if self.to_float32:
            img_ir = img_ir.astype(np.float32)
        
        results['img_ir'] = img_ir
        results['img_ir_shape'] = img_ir.shape[:2]
        
        return results


@TRANSFORMS.register_module()
class LoadKAISTImagePair(BaseTransform):
    """
    Load both RGB and IR images for KAIST dataset in one transform.
    
    This is a convenience transform that combines LoadImageFromFile and 
    LoadKAISTIRImageFromFile for KAIST dataset.
    
    Args:
        to_float32 (bool): Whether to convert to float32. Defaults to False.
        color_type (str): Color type for imread. Defaults to 'color'.
        imdecode_backend (str): Backend for imdecode. Defaults to 'cv2'.
        ignore_empty (bool): Whether to ignore empty images. Defaults to True.
    """
    
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
    
    def transform(self, results: dict) -> Optional[dict]:
        """Load both RGB and IR images.
        
        Args:
            results (dict): Result dict containing 'img_path' and 'img_ir_path'.
            
        Returns:
            dict: Result dict with 'img', 'img_ir' and related keys.
        """
        # Load RGB image
        rgb_path = results['img_path']
        try:
            img = mmcv.imread(rgb_path, flag=self.color_type,
                             backend=self.imdecode_backend)
        except Exception as e:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}") from e
        
        if self.to_float32:
            img = img.astype(np.float32)
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        # Load IR image
        ir_path = results.get('img_ir_path', None)
        if ir_path is not None:
            try:
                img_ir = mmcv.imread(ir_path, flag=self.color_type,
                                    backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    img_ir = img.copy()
                else:
                    raise FileNotFoundError(f"IR image not found: {ir_path}") from e
        else:
            if self.ignore_empty:
                img_ir = img.copy()
            else:
                raise ValueError("img_ir_path not found in results")
        
        if self.to_float32:
            img_ir = img_ir.astype(np.float32)
        
        results['img_ir'] = img_ir
        results['img_ir_shape'] = img_ir.shape[:2]
        
        return results
