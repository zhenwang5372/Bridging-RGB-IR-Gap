# Copyright (c) Tencent Inc. All rights reserved.
from .yolo_world_head import YOLOWorldHead, YOLOWorldHeadModule, RepYOLOWorldHeadModule
from .yolo_world_head_v2 import YOLOWorldHeadModuleV2
from .class_specific_yolo_head import ClassSpecificYOLOHeadModule

__all__ = [
    'YOLOWorldHead', 'YOLOWorldHeadModule', 'RepYOLOWorldHeadModule',
    'YOLOWorldHeadModuleV2',
    'ClassSpecificYOLOHeadModule'
]

# Optional: segmentation head (requires mmyolo.models.dense_heads.yolov5_ins_head)
try:
    from .yolo_world_seg_head import YOLOWorldSegHead, YOLOWorldSegHeadModule
    __all__.extend(['YOLOWorldSegHead', 'YOLOWorldSegHeadModule'])
except ImportError:
    pass  # Segmentation head not available (requires yolov5_ins_head)
