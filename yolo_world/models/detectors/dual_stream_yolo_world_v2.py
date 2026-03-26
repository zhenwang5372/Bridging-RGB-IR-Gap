# Copyright (c) Tencent Inc. All rights reserved.
# Dual-Stream RGB-IR YOLO-World Detector V2
#
# ==================== V2 版本核心改进 ====================
#
# 相比 V1 的改进：
# 1. 支持 mask/S_map 监督损失，显式引导融合模块学习
# 2. 从 GT boxes 生成监督信号
#
# ==================== 使用方法 ====================
#
# 在配置文件中：
# model = dict(
#     type='DualStreamYOLOWorldDetectorV2',
#     use_mask_supervision=True,
#     mask_loss_weight=0.1,
#     ...
# )
#

from typing import List, Tuple, Union, Optional, Dict
import torch
from torch import Tensor
import torch.nn.functional as F
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


def generate_mask_from_boxes(
    gt_bboxes: List[Tensor],
    feature_sizes: List[Tuple[int, int]],
    device: torch.device,
) -> List[Tensor]:
    """
    从 GT boxes 生成各尺度的 mask 监督信号
    
    Args:
        gt_bboxes: List[Tensor], 每个 batch 的 GT boxes，格式 [N, 4] (x1, y1, x2, y2) 归一化坐标
        feature_sizes: List[Tuple[int, int]], 各尺度特征图大小 [(H1, W1), (H2, W2), ...]
        device: 设备
    
    Returns:
        gt_masks: List[Tensor], 各尺度的 GT mask，形状 [B, 1, H, W]
    """
    B = len(gt_bboxes)
    gt_masks = []
    
    for H, W in feature_sizes:
        batch_masks = []
        for b in range(B):
            mask = torch.zeros(1, H, W, device=device)
            boxes = gt_bboxes[b]  # [N, 4]
            
            if boxes.numel() > 0:
                # 将归一化坐标转换为特征图坐标
                # 假设 boxes 格式为 [x1, y1, x2, y2]，值在 [0, 1] 或像素坐标
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    
                    # 如果是像素坐标，需要归一化（假设输入图像是 640x640）
                    # 这里简化处理，假设已经是归一化坐标或者需要根据实际情况调整
                    if x2 > 1 or y2 > 1:
                        # 像素坐标，假设原图大小，需要归一化
                        # 这里需要知道原图大小，暂时假设 640
                        x1, y1, x2, y2 = x1 / 640, y1 / 640, x2 / 640, y2 / 640
                    
                    # 转换到特征图坐标
                    fx1 = int(max(0, x1 * W))
                    fy1 = int(max(0, y1 * H))
                    fx2 = int(min(W, x2 * W))
                    fy2 = int(min(H, y2 * H))
                    
                    if fx2 > fx1 and fy2 > fy1:
                        mask[0, fy1:fy2, fx1:fx2] = 1.0
            
            batch_masks.append(mask)
        
        gt_masks.append(torch.stack(batch_masks, dim=0))  # [B, 1, H, W]
    
    return gt_masks


class MaskSupervisionLoss:
    """
    Mask/S_map 监督损失
    
    使用 GT boxes 生成监督信号，用 BCE 损失监督模型生成的 S_map
    
    公式：
        L_mask = BCE(S_map, GT_mask)
    
    Args:
        loss_weight (float): 损失权重，默认 0.1
        use_focal (bool): 是否使用 focal loss，默认 False
        focal_gamma (float): focal loss 的 gamma 参数，默认 2.0
    """
    
    def __init__(
        self,
        loss_weight: float = 0.1,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        self.loss_weight = loss_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
    
    def _parse_gt_bboxes(
        self,
        batch_data_samples: Union[Dict, List],
        batch_size: int,
        device: torch.device
    ) -> List[Tensor]:
        """
        从 batch_data_samples 中解析 GT boxes
        
        支持两种格式：
        1. 字典格式（yolow_collate）: {'bboxes_labels': Tensor[N, 6]}
           格式为 [batch_idx, label, x1, y1, x2, y2]
        2. 列表格式（标准 mmdet）: List[DataSample]
        
        Returns:
            gt_bboxes: List[Tensor], 每个 batch 的 boxes [N, 4]
        """
        if isinstance(batch_data_samples, dict):
            # 字典格式（yolow_collate）
            bboxes_labels = batch_data_samples.get('bboxes_labels', None)
            if bboxes_labels is None or bboxes_labels.numel() == 0:
                return [torch.empty(0, 4, device=device) for _ in range(batch_size)]
            
            # bboxes_labels: [batch_idx, label, x1, y1, x2, y2]
            gt_bboxes = []
            for b in range(batch_size):
                # 筛选当前 batch 的 boxes
                mask = bboxes_labels[:, 0] == b
                boxes = bboxes_labels[mask, 2:6]  # [x1, y1, x2, y2]
                gt_bboxes.append(boxes)
            
            return gt_bboxes
        
        elif isinstance(batch_data_samples, list):
            # 列表格式（标准 mmdet）
            gt_bboxes = []
            for data_sample in batch_data_samples:
                if hasattr(data_sample, 'gt_instances'):
                    bboxes = data_sample.gt_instances.bboxes
                    if hasattr(bboxes, 'tensor'):
                        bboxes = bboxes.tensor
                    gt_bboxes.append(bboxes)
                else:
                    gt_bboxes.append(torch.empty(0, 4, device=device))
            return gt_bboxes
        
        else:
            # 未知格式，返回空
            return [torch.empty(0, 4, device=device) for _ in range(batch_size)]
    
    def __call__(
        self,
        s_maps: List[Tensor],
        batch_data_samples: Union[Dict, List],
        img_shape: Tuple[int, int] = (640, 640),
    ) -> Tensor:
        """
        计算损失
        
        Args:
            s_maps: List[Tensor], 各尺度的 S_map，形状 [B, 1, H, W]
            batch_data_samples: 数据样本，支持字典格式（yolow_collate）或列表格式
            img_shape: 输入图像大小
        
        Returns:
            loss: scalar tensor
        """
        if s_maps is None or len(s_maps) == 0:
            return torch.tensor(0.0)
        
        device = s_maps[0].device
        batch_size = s_maps[0].shape[0]
        
        # 解析 GT boxes（在 Loss 类内部处理）
        gt_bboxes = self._parse_gt_bboxes(batch_data_samples, batch_size, device)
        
        # 检查 batch size 是否匹配
        if len(gt_bboxes) != batch_size:
            return torch.tensor(0.0, device=device)
        
        # 获取各尺度特征图大小
        feature_sizes = [(s.shape[2], s.shape[3]) for s in s_maps]
        
        # 生成 GT mask
        gt_masks = generate_mask_from_boxes(gt_bboxes, feature_sizes, device)
        
        # 计算各尺度损失
        total_loss = torch.tensor(0.0, device=device)
        for s_map, gt_mask in zip(s_maps, gt_masks):
            # 确保 s_map 在 [0, 1] 范围内（用于 BCE）
            s_map_clamped = torch.clamp(s_map, 0.0, 1.0)
            
            if self.use_focal:
                # Focal loss
                bce = F.binary_cross_entropy(s_map_clamped, gt_mask, reduction='none')
                pt = torch.where(gt_mask == 1, s_map_clamped, 1 - s_map_clamped)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss = (focal_weight * bce).mean()
            else:
                # 普通 BCE loss
                loss = F.binary_cross_entropy(s_map_clamped, gt_mask)
            
            total_loss = total_loss + loss
        
        # 平均各尺度损失
        total_loss = total_loss / len(s_maps)
        
        return self.loss_weight * total_loss


@MODELS.register_module()
class DualStreamYOLOWorldDetectorV2(YOLODetector):
    """
    Dual-stream RGB-IR YOLO-World Detector V2
    
    相比 V1 的改进：
    1. 支持 mask/S_map 监督损失
    2. 从 GT boxes 生成监督信号
    
    Args:
        mm_neck (bool): Whether to use multi-modal neck. Defaults to False.
        num_train_classes (int): Number of training classes. Defaults to 80.
        num_test_classes (int): Number of test classes. Defaults to 80.
        use_mask_supervision (bool): 是否使用 mask 监督损失. Defaults to False.
        mask_loss_weight (float): mask 监督损失权重. Defaults to 0.1.
        mask_loss_focal (bool): 是否使用 focal loss. Defaults to False.
    """
    
    def __init__(
        self,
        *args,
        mm_neck: bool = False,
        num_train_classes: int = 80,
        num_test_classes: int = 80,
        aggregator: ConfigType = None,
        use_mask_supervision: bool = False,
        mask_loss_weight: float = 0.1,
        mask_loss_focal: bool = False,
        **kwargs
    ) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.use_mask_supervision = use_mask_supervision
        
        super().__init__(*args, **kwargs)
        self._ir_input = None  # Store IR input for current batch
        
        # Build class dimension aggregator if provided
        if aggregator is not None:
            self.aggregator = MODELS.build(aggregator)
        else:
            self.aggregator = None
        
        # Build mask supervision loss
        if use_mask_supervision:
            self.mask_loss = MaskSupervisionLoss(
                loss_weight=mask_loss_weight,
                use_focal=mask_loss_focal,
            )
            print(f"[DualStreamYOLOWorldDetectorV2] Mask supervision enabled, weight={mask_loss_weight}")
        else:
            self.mask_loss = None
    
    def forward(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None,
        mode: str = 'tensor',
        inputs_ir: Tensor = None,
        **kwargs
    ) -> Union[dict, list, Tensor]:
        """
        Unified forward for dual-stream detection.
        """
        # Store IR input for use in loss/predict methods
        self._ir_input = inputs_ir
        
        try:
            if mode == 'loss':
                return self.loss(inputs, data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}".')
        finally:
            # Clear stored IR input after processing
            self._ir_input = None
    
    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """
        Calculate losses from a batch of inputs and data samples.
        
        V2 改进：添加 mask 监督损失
        """
        self.bbox_head.num_classes = self.num_train_classes
        
        # Extract IR input from data_samples
        img_ir = self._get_ir_input(batch_data_samples)
        
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        
        # 计算检测损失
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        
        # V2 新增：计算 mask 监督损失
        if self.use_mask_supervision and self.mask_loss is not None:
            # 获取 S_map
            s_maps = self._get_s_maps()
            
            if s_maps is not None:
                # 直接传递 batch_data_samples，解析逻辑在 MaskSupervisionLoss 内部处理
                mask_loss = self.mask_loss(s_maps, batch_data_samples)
                losses['loss_mask'] = mask_loss
        
        return losses
    
    def _get_s_maps(self) -> Optional[List[Tensor]]:
        """获取 backbone 的 S_map"""
        if hasattr(self.backbone, 'get_s_maps'):
            return self.backbone.get_s_maps()
        return None
    
    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True
    ) -> SampleList:
        """Predict results from a batch of inputs."""
        # Extract IR input from data_samples
        img_ir = self._get_ir_input(batch_data_samples)
        
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        
        self.bbox_head.num_classes = txt_feats.shape[1] if txt_feats is not None and txt_feats.dim() == 3 else self.num_test_classes
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)
        
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        """Network forward process."""
        img_ir = self._get_ir_input(batch_data_samples)
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples, img_ir=img_ir)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results
    
    def _get_ir_input(self, batch_data_samples) -> Tensor:
        """Extract IR input tensor from batch_data_samples or stored input."""
        # First check stored IR input from forward()
        if self._ir_input is not None:
            return self._ir_input
        
        if batch_data_samples is None:
            return None
        
        if isinstance(batch_data_samples, dict):
            return batch_data_samples.get('inputs_ir', None)
        elif isinstance(batch_data_samples, list) and len(batch_data_samples) > 0:
            if hasattr(batch_data_samples[0], 'img_ir'):
                ir_list = [ds.img_ir for ds in batch_data_samples]
                return torch.stack(ir_list, dim=0)
        
        return None
    
    def extract_feat(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        img_ir: Tensor = None
    ) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features from RGB+IR images and text."""
        txt_feats = None
        txt_masks = None
        fused_feats = None
        
        # Get text prompts
        if batch_data_samples is None:
            texts = getattr(self, 'texts', None)
            txt_feats = getattr(self, 'text_feats', None)
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            texts = None
        
        # Forward through backbone
        if hasattr(self.backbone, 'forward_image'):
            if txt_feats is not None:
                img_feats = self.backbone.forward_image(batch_inputs, img_ir)
            else:
                backbone_output = self.backbone(batch_inputs, texts, img_ir)
                if len(backbone_output) == 3:
                    img_feats, (txt_feats, txt_masks), fused_feats = backbone_output
                else:
                    img_feats, (txt_feats, txt_masks) = backbone_output
        else:
            if txt_feats is not None:
                img_feats = self.backbone.forward_image(batch_inputs)
            else:
                backbone_output = self.backbone(batch_inputs, texts)
                if len(backbone_output) == 3:
                    img_feats, (txt_feats, txt_masks), fused_feats = backbone_output
                else:
                    img_feats, (txt_feats, txt_masks) = backbone_output
        
        # Apply neck
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        # Apply class dimension aggregator if exists
        if hasattr(self, 'aggregator') and self.aggregator is not None:
            img_feats = self.aggregator(img_feats, fused_feats)
        
        return img_feats, txt_feats, txt_masks
    
    def reparameterize(self, texts: List[List[str]]) -> None:
        """Encode text embeddings into the detector."""
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)
