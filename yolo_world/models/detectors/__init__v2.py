# Copyright (c) Tencent Inc. All rights reserved.
# V2版本的__init__.py，包含三模态Detector模块
# 使用方法：将此文件重命名为__init__.py替换原文件，或在配置中通过custom_imports导入

from .yolo_world import YOLOWorldDetector, SimpleYOLOWorldDetector
from .yolo_world_image import YOLOWorldImageDetector
from .dual_stream_yolo_world import (
    DualStreamRGBIRBackbone,
    DualStreamMultiModalYOLOBackbone,
    DualStreamYOLOWorldDetector
)

# V2新增模块
from .dual_stream_yolo_world_v2 import (
    DualStreamMultiModalYOLOBackboneV2,
    DualStreamYOLOWorldDetectorV2
)

__all__ = [
    # 原有模块
    'YOLOWorldDetector', 'SimpleYOLOWorldDetector', 'YOLOWorldImageDetector',
    'DualStreamRGBIRBackbone', 'DualStreamMultiModalYOLOBackbone',
    'DualStreamYOLOWorldDetector',
    # V2新增模块
    'DualStreamMultiModalYOLOBackboneV2',
    'DualStreamYOLOWorldDetectorV2',
]

