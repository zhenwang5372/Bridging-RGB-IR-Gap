# Copyright (c) Tencent Inc. All rights reserved.
# Synchronized RGB-IR Transforms for Dual-Modal Detection
import copy
import os
import os.path as osp
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import autocast_box_type
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from mmengine.structures import InstanceData
from numpy import random

from mmyolo.registry import TRANSFORMS

# Import BaseMixImageTransform for Mosaic
try:
    from mmyolo.datasets.transforms.mix_img_transforms import BaseMixImageTransform
except ImportError:
    BaseMixImageTransform = BaseTransform  # Fallback


@TRANSFORMS.register_module()
class LoadIRImageFromFile(BaseTransform):
    """
    Load IR image based on RGB image path.
    
    Infers IR image path from RGB path using suffix replacement rules.
    
    Args:
        ir_suffix (str): IR image suffix. Defaults to '_PreviewData.jpeg'.
        rgb_suffix (str): RGB image suffix to be replaced. Defaults to '_RGB.jpg'.
        to_float32 (bool): Whether to convert to float32. Defaults to False.
        color_type (str): Color type for imread. Defaults to 'color'.
        imdecode_backend (str): Backend for imdecode. Defaults to 'cv2'.
        ignore_empty (bool): Whether to ignore empty IR images. Defaults to True.
    """
    
    def __init__(self,
                 ir_suffix: str = '_PreviewData.jpeg',
                 rgb_suffix: str = '_RGB.jpg',
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = True):
        self.ir_suffix = ir_suffix
        self.rgb_suffix = rgb_suffix
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
    
    def _get_ir_path(self, rgb_path: str) -> str:
        """Infer IR image path from RGB image path."""
        # Replace suffix: *_RGB.jpg -> *_PreviewData.jpeg
        if self.rgb_suffix in rgb_path:
            ir_path = rgb_path.replace(self.rgb_suffix, self.ir_suffix)
        else:
            # Fallback: just change extension and append IR suffix
            base, _ = osp.splitext(rgb_path)
            ir_path = base.replace('_RGB', '') + self.ir_suffix
        return ir_path
    
    def transform(self, results: dict) -> Optional[dict]:
        """Load IR image.
        
        Args:
            results (dict): Result dict containing 'img_path'.
            
        Returns:
            dict: Result dict with 'img_ir' and 'img_ir_path' keys.
        """
        rgb_path = results['img_path']
        ir_path = self._get_ir_path(rgb_path)
        
        try:
            img_ir = mmcv.imread(ir_path, flag=self.color_type, 
                                backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                # Use RGB image as fallback for IR
                img_ir = results.get('img', None)
                if img_ir is None:
                    img_ir = mmcv.imread(rgb_path, flag=self.color_type,
                                        backend=self.imdecode_backend)
                img_ir = img_ir.copy()
            else:
                raise FileNotFoundError(f"IR image not found: {ir_path}") from e
        
        if self.to_float32:
            img_ir = img_ir.astype(np.float32)
        
        results['img_ir'] = img_ir
        results['img_ir_path'] = ir_path
        results['img_ir_shape'] = img_ir.shape[:2]
        
        return results


@TRANSFORMS.register_module()
class SyncResize(BaseTransform):
    """
    Synchronized resize for RGB and IR images.
    """
    
    def __init__(self,
                 scale: Tuple[int, int],
                 keep_ratio: bool = True,
                 interpolation: str = 'bilinear'):
        self.scale = scale  # (width, height)
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
    
    def transform(self, results: dict) -> dict:
        """Apply synchronized resize."""
        img = results['img']
        h, w = img.shape[:2]
        
        if self.keep_ratio:
            scale_w, scale_h = self.scale
            ratio = min(scale_w / w, scale_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
        else:
            new_w, new_h = self.scale
            ratio = new_w / w
        
        # Resize RGB
        results['img'] = mmcv.imresize(img, (new_w, new_h), 
                                       interpolation=self.interpolation)
        results['img_shape'] = results['img'].shape[:2]
        results['scale_factor'] = (ratio, ratio)
        
        # Resize IR (same transformation)
        if 'img_ir' in results:
            results['img_ir'] = mmcv.imresize(results['img_ir'], (new_w, new_h),
                                             interpolation=self.interpolation)
            results['img_ir_shape'] = results['img_ir'].shape[:2]
        
        # Scale bboxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            results['gt_bboxes'].rescale_([ratio, ratio])
        
        return results


@TRANSFORMS.register_module()
class SyncLetterResize(BaseTransform):
    """
    Synchronized LetterResize for RGB and IR images.
    
    Resizes with padding to maintain aspect ratio.
    """
    
    def __init__(self,
                 scale: Tuple[int, int],
                 pad_val: Union[float, dict] = dict(img=114, img_ir=114),
                 allow_scale_up: bool = True):
        self.scale = scale  # (width, height)
        self.allow_scale_up = allow_scale_up
        if isinstance(pad_val, (int, float)):
            self.pad_val = dict(img=pad_val, img_ir=pad_val)
        else:
            self.pad_val = pad_val
    
    def transform(self, results: dict) -> dict:
        """Apply synchronized letter resize."""
        img = results['img']
        h, w = img.shape[:2]
        target_w, target_h = self.scale
        
        # Calculate scale factor
        ratio = min(target_w / w, target_h / h)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)
        
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        
        # Resize and pad RGB
        img_resized = mmcv.imresize(img, (new_w, new_h))
        img_padded = cv2.copyMakeBorder(
            img_resized, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(self.pad_val.get('img', 114),) * 3
        )
        results['img'] = img_padded
        results['img_shape'] = img_padded.shape[:2]
        results['scale_factor'] = (ratio, ratio)
        results['pad_param'] = np.array([top_pad, bottom_pad, left_pad, right_pad],
                                        dtype=np.float32)
        
        # Process IR (same transformation)
        if 'img_ir' in results:
            img_ir = results['img_ir']
            img_ir_resized = mmcv.imresize(img_ir, (new_w, new_h))
            img_ir_padded = cv2.copyMakeBorder(
                img_ir_resized, top_pad, bottom_pad, left_pad, right_pad,
                cv2.BORDER_CONSTANT, value=(self.pad_val.get('img_ir', 114),) * 3
            )
            results['img_ir'] = img_ir_padded
            results['img_ir_shape'] = img_ir_padded.shape[:2]
        
        # Adjust bboxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            results['gt_bboxes'].rescale_([ratio, ratio])
            results['gt_bboxes'].translate_([left_pad, top_pad])
        
        return results


@TRANSFORMS.register_module()
class SyncRandomFlip(BaseTransform):
    """
    Synchronized random flip for RGB and IR images.
    """
    
    def __init__(self, prob: float = 0.5, direction: str = 'horizontal'):
        self.prob = prob
        self.direction = direction
    
    @cache_randomness
    def _do_flip(self) -> bool:
        return random.random() < self.prob
    
    def transform(self, results: dict) -> dict:
        """Apply synchronized flip."""
        if not self._do_flip():
            results['flip'] = False
            results['flip_direction'] = None
            return results
        
        h, w = results['img'].shape[:2]
        
        # Flip RGB
        if self.direction == 'horizontal':
            results['img'] = np.ascontiguousarray(np.fliplr(results['img']))
        elif self.direction == 'vertical':
            results['img'] = np.ascontiguousarray(np.flipud(results['img']))
        
        # Flip IR (same direction)
        if 'img_ir' in results:
            if self.direction == 'horizontal':
                results['img_ir'] = np.ascontiguousarray(np.fliplr(results['img_ir']))
            elif self.direction == 'vertical':
                results['img_ir'] = np.ascontiguousarray(np.flipud(results['img_ir']))
        
        # Flip bboxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            results['gt_bboxes'].flip_([h, w], direction=self.direction)
        
        results['flip'] = True
        results['flip_direction'] = self.direction
        
        return results


@TRANSFORMS.register_module()
class SyncRandomAffine(BaseTransform):
    """
    Synchronized random affine transform for RGB and IR.
    """
    
    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114)):
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
    
    @cache_randomness
    def _get_random_params(self, h: int, w: int) -> dict:
        """Generate random affine parameters."""
        rotation = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        scale = random.uniform(*self.scaling_ratio_range)
        shear_x = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_y = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        translate_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * w
        translate_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * h
        
        return dict(
            rotation=rotation,
            scale=scale,
            shear_x=shear_x,
            shear_y=shear_y,
            translate_x=translate_x,
            translate_y=translate_y
        )
    
    def _get_affine_matrix(self, params: dict, h: int, w: int) -> np.ndarray:
        """Generate affine transformation matrix."""
        cx, cy = w / 2, h / 2
        
        rotation = params['rotation']
        scale = params['scale']
        
        angle_rad = np.deg2rad(rotation)
        cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
        
        M = np.array([
            [cos_val * scale, -sin_val * scale, 0],
            [sin_val * scale, cos_val * scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        T_center = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        T_center_inv = np.array([
            [1, 0, cx + params['translate_x']],
            [0, 1, cy + params['translate_y']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        shear = np.array([
            [1, np.tan(np.deg2rad(params['shear_x'])), 0],
            [np.tan(np.deg2rad(params['shear_y'])), 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        M_combined = T_center_inv @ M @ shear @ T_center
        
        return M_combined[:2]
    
    def transform(self, results: dict) -> dict:
        """Apply synchronized affine transform."""
        img = results['img']
        h, w = img.shape[:2]
        
        # Calculate output size considering border
        height = h + self.border[0] * 2
        width = w + self.border[1] * 2
        
        # Ensure positive dimensions
        height = max(height, 1)
        width = max(width, 1)
        
        # Get random parameters (same for both modalities)
        params = self._get_random_params(height, width)
        M = self._get_affine_matrix(params, height, width)
        
        # Apply to RGB
        results['img'] = cv2.warpAffine(
            img, M, (width, height),
            borderValue=self.border_val
        )
        results['img_shape'] = results['img'].shape[:2]
        
        # Apply to IR (same matrix)
        if 'img_ir' in results:
            results['img_ir'] = cv2.warpAffine(
                results['img_ir'], M, (width, height),
                borderValue=self.border_val
            )
            results['img_ir_shape'] = results['img_ir'].shape[:2]
        
        # Transform bboxes
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes'].tensor.numpy()
            num_bboxes = len(bboxes)
            
            if num_bboxes > 0:
                # Get corners
                xs = bboxes[:, [0, 0, 2, 2]].reshape(-1)
                ys = bboxes[:, [1, 3, 1, 3]].reshape(-1)
                
                # Transform
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])
                transformed = M @ points
                
                # Get new bboxes
                xs_new = transformed[0].reshape(-1, 4)
                ys_new = transformed[1].reshape(-1, 4)
                
                new_bboxes = np.stack([
                    xs_new.min(axis=1),
                    ys_new.min(axis=1),
                    xs_new.max(axis=1),
                    ys_new.max(axis=1)
                ], axis=1)
                
                # Clip to image boundaries
                new_bboxes[:, [0, 2]] = np.clip(new_bboxes[:, [0, 2]], 0, width)
                new_bboxes[:, [1, 3]] = np.clip(new_bboxes[:, [1, 3]], 0, height)
                
                # Filter invalid boxes
                valid = (new_bboxes[:, 2] > new_bboxes[:, 0]) & \
                       (new_bboxes[:, 3] > new_bboxes[:, 1])
                
                # Apply valid mask to filter bboxes
                results['gt_bboxes'].tensor = results['gt_bboxes'].tensor.new_tensor(new_bboxes[valid])
                
                if 'gt_bboxes_labels' in results:
                    results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid]
                if 'gt_ignore_flags' in results:
                    results['gt_ignore_flags'] = results['gt_ignore_flags'][valid]
        
        return results


@TRANSFORMS.register_module()
class SyncMosaic(BaseMixImageTransform):
    """
    Synchronized Mosaic augmentation for RGB-IR pairs.
    
    Inherits from BaseMixImageTransform to properly get mix_results.
    
    Args:
        img_scale (tuple): Image scale (width, height). Defaults to (640, 640).
        center_ratio_range (tuple): Center ratio range. Defaults to (0.5, 1.5).
        pad_val (float): Padding value. Defaults to 114.0.
        prob (float): Probability of applying. Defaults to 1.0.
        pre_transform (list): Pre-transforms. Defaults to None.
        max_refetch (int): Max refetch iterations. Defaults to 15.
    """
    
    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 pad_val: float = 114.0,
                 prob: float = 1.0,
                 pre_transform: List[dict] = None,
                 max_refetch: int = 15,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True):
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch
        )
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
    
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> List[int]:
        """Get random indexes for mosaic (3 other images)."""
        return [random.randint(0, len(dataset) - 1) for _ in range(3)]
    
    def _mosaic_combine(self, loc: str, center: Tuple[int, int], 
                       img_shape: Tuple[int, int]) -> Tuple[Tuple, Tuple]:
        """Calculate mosaic combine parameters."""
        img_w, img_h = img_shape
        center_x, center_y = center
        
        if loc == 'top_left':
            x1, y1, x2, y2 = max(center_x - img_w, 0), max(center_y - img_h, 0), center_x, center_y
            crop_x1 = max(img_w - center_x, 0)
            crop_y1 = max(img_h - center_y, 0)
            crop_x2, crop_y2 = img_w, img_h
        elif loc == 'top_right':
            x1, y1, x2, y2 = center_x, max(center_y - img_h, 0), min(center_x + img_w, self.img_scale[0] * 2), center_y
            crop_x1, crop_y1 = 0, max(img_h - center_y, 0)
            crop_x2, crop_y2 = min(self.img_scale[0] * 2 - center_x, img_w), img_h
        elif loc == 'bottom_left':
            x1, y1, x2, y2 = max(center_x - img_w, 0), center_y, center_x, min(center_y + img_h, self.img_scale[1] * 2)
            crop_x1 = max(img_w - center_x, 0)
            crop_y1 = 0
            crop_x2, crop_y2 = img_w, min(self.img_scale[1] * 2 - center_y, img_h)
        else:  # bottom_right
            x1, y1, x2, y2 = center_x, center_y, min(center_x + img_w, self.img_scale[0] * 2), min(center_y + img_h, self.img_scale[1] * 2)
            crop_x1, crop_y1 = 0, 0
            crop_x2 = min(self.img_scale[0] * 2 - center_x, img_w)
            crop_y2 = min(self.img_scale[1] * 2 - center_y, img_h)
        
        paste_coord = (x1, y1, x2, y2)
        crop_coord = (crop_x1, crop_y1, crop_x2, crop_y2)
        
        return paste_coord, crop_coord
    
    @autocast_box_type()
    def mix_img_transform(self, results: dict) -> dict:
        """Apply synchronized mosaic."""
        assert 'mix_results' in results, 'mix_results not found'
        assert len(results['mix_results']) == 3
        
        img_scale_w, img_scale_h = self.img_scale
        
        # Create mosaic canvases
        mosaic_img = np.full(
            (int(img_scale_h * 2), int(img_scale_w * 2), 3),
            self.pad_val, dtype=results['img'].dtype
        )
        
        has_ir = 'img_ir' in results
        if has_ir:
            mosaic_img_ir = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                self.pad_val, dtype=results['img_ir'].dtype
            )
        
        # Mosaic center
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center = (center_x, center_y)
        
        mosaic_bboxes = []
        mosaic_labels = []
        mosaic_ignore_flags = []
        
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = results
            else:
                result_patch = results['mix_results'][i - 1]
            
            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]
            
            # Keep ratio resize
            scale_ratio = min(img_scale_h / h_i, img_scale_w / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio), int(h_i * scale_ratio)))
            
            # Process IR
            if has_ir and 'img_ir' in result_patch:
                img_ir_i = result_patch['img_ir']
                img_ir_i = mmcv.imresize(img_ir_i, (int(w_i * scale_ratio), int(h_i * scale_ratio)))
            
            # Get paste and crop coords
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord
            
            # Paste RGB
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            
            # Paste IR
            if has_ir and 'img_ir' in result_patch:
                mosaic_img_ir[y1_p:y2_p, x1_p:x2_p] = img_ir_i[y1_c:y2_c, x1_c:x2_c]
            
            # Adjust bboxes
            gt_bboxes_i = result_patch['gt_bboxes']
            gt_labels_i = result_patch['gt_bboxes_labels']
            gt_ignore_i = result_patch.get('gt_ignore_flags', 
                                          np.zeros(len(gt_labels_i), dtype=bool))
            
            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio, scale_ratio])
            gt_bboxes_i.translate_([padw, padh])
            
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            mosaic_ignore_flags.append(gt_ignore_i)
        
        # Update results
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        
        if has_ir:
            results['img_ir'] = mosaic_img_ir
            results['img_ir_shape'] = mosaic_img_ir.shape[:2]
        
        # Concatenate bboxes
        results['gt_bboxes'] = mosaic_bboxes[0].cat(mosaic_bboxes)
        results['gt_bboxes_labels'] = np.concatenate(mosaic_labels, 0)
        results['gt_ignore_flags'] = np.concatenate(mosaic_ignore_flags, 0)
        
        return results


@TRANSFORMS.register_module()
class DualModalityPhotometricDistortion(BaseTransform):
    """
    Modality-specific photometric distortion.
    
    RGB: Full color augmentation (brightness, contrast, saturation, hue)
    IR: Limited augmentation (brightness, contrast only - no hue/saturation)
    """
    
    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Tuple[float, float] = (0.5, 1.5),
                 saturation_range: Tuple[float, float] = (0.5, 1.5),
                 hue_delta: int = 18,
                 ir_brightness_delta: int = 20,
                 ir_contrast_range: Tuple[float, float] = (0.8, 1.2),
                 prob: float = 0.5):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        self.ir_brightness_delta = ir_brightness_delta
        self.ir_contrast_range = ir_contrast_range
        self.prob = prob
    
    def transform(self, results: dict) -> dict:
        """Apply photometric distortion."""
        if random.random() > self.prob:
            return results
        
        # RGB augmentation
        img = results['img'].astype(np.float32)
        
        # Brightness
        if random.randint(0, 2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta
        
        # Contrast
        if random.randint(0, 2):
            alpha = random.uniform(*self.contrast_range)
            img *= alpha
        
        # Convert to HSV for saturation and hue
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = mmcv.bgr2hsv(img)
        
        # Saturation
        if random.randint(0, 2):
            img[:, :, 1] = np.clip(
                img[:, :, 1].astype(np.float32) * random.uniform(*self.saturation_range),
                0, 255
            ).astype(np.uint8)
        
        # Hue
        if random.randint(0, 2):
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + 
                random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
        
        results['img'] = mmcv.hsv2bgr(img)
        
        # IR augmentation (no hue/saturation changes)
        if 'img_ir' in results:
            img_ir = results['img_ir'].astype(np.float32)
            
            # Brightness
            if random.randint(0, 2):
                delta = random.uniform(-self.ir_brightness_delta, self.ir_brightness_delta)
                img_ir += delta
            
            # Contrast
            if random.randint(0, 2):
                alpha = random.uniform(*self.ir_contrast_range)
                img_ir *= alpha
            
            results['img_ir'] = np.clip(img_ir, 0, 255).astype(np.uint8)
        
        return results


@TRANSFORMS.register_module()
class ThermalSpecificAugmentation(BaseTransform):
    """
    Thermal/IR specific augmentations.
    """
    
    def __init__(self,
                 fpa_noise_level: float = 0.02,
                 crossover_prob: float = 0.2,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 shift_range: int = 20,
                 prob: float = 0.5):
        self.fpa_noise_level = fpa_noise_level
        self.crossover_prob = crossover_prob
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.prob = prob
    
    def transform(self, results: dict) -> dict:
        """Apply thermal augmentation to IR image."""
        if 'img_ir' not in results or random.random() > self.prob:
            return results
        
        img_ir = results['img_ir'].astype(np.float32)
        
        # FPA noise simulation
        if random.random() < 0.5:
            noise = np.random.normal(0, self.fpa_noise_level * 255, img_ir.shape)
            img_ir += noise
        
        # Intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            img_ir = img_ir * scale
        
        # Intensity shift
        if random.random() < 0.5:
            shift = random.uniform(-self.shift_range, self.shift_range)
            img_ir += shift
        
        # Thermal crossover simulation
        if random.random() < self.crossover_prob:
            threshold = random.uniform(100, 200)
            crossover_mask = img_ir > threshold
            img_ir[crossover_mask] = threshold - (img_ir[crossover_mask] - threshold) * 0.5
        
        results['img_ir'] = np.clip(img_ir, 0, 255).astype(np.uint8)
        
        return results


@TRANSFORMS.register_module()
class PackDualModalInputs(BaseTransform):
    """
    Pack RGB and IR inputs into the format expected by the dual-stream detector.
    
    This transform packs both RGB and IR images as inputs, and stores IR-related
    metadata properly.
    """
    
    def __init__(self,
                 meta_keys: Tuple[str, ...] = ('img_id', 'img_path', 'ori_shape',
                                               'img_shape', 'scale_factor',
                                               'flip', 'flip_direction',
                                               'pad_param', 'texts',
                                               'img_ir_path', 'img_ir_shape')):
        self.meta_keys = meta_keys
    
    def transform(self, results: dict) -> dict:
        """Pack RGB and IR inputs."""
        packed_results = dict()
        
        # Pack RGB image
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = torch.from_numpy(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = img
        
        # Pack IR image  
        if 'img_ir' in results:
            img_ir = results['img_ir']
            if len(img_ir.shape) < 3:
                img_ir = np.expand_dims(img_ir, -1)
            if not img_ir.flags.c_contiguous:
                img_ir = np.ascontiguousarray(img_ir.transpose(2, 0, 1))
                img_ir = torch.from_numpy(img_ir)
            else:
                img_ir = torch.from_numpy(img_ir).permute(2, 0, 1).contiguous()
            packed_results['inputs_ir'] = img_ir
        
        # Create data_samples
        data_sample = DetDataSample()
        
        # Pack instance data
        instance_data = InstanceData()
        
        if 'gt_bboxes' in results:
            instance_data.bboxes = results['gt_bboxes'].tensor
        if 'gt_bboxes_labels' in results:
            instance_data.labels = torch.LongTensor(results['gt_bboxes_labels'])
        if 'gt_ignore_flags' in results:
            instance_data.ignore_flags = torch.BoolTensor(results['gt_ignore_flags'])
        
        data_sample.gt_instances = instance_data
        
        # Pack metainfo
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        
        data_sample.set_metainfo(img_meta)
        
        # Store IR image in data_sample for inference
        if 'img_ir' in results:
            data_sample.img_ir = packed_results.get('inputs_ir')
        
        packed_results['data_samples'] = data_sample
        
        return packed_results
