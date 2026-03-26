#!/usr/bin/env python3
"""
mAP50 错误分析工具

分析影响 mAP50 的图像，找出漏检和误检的样本，并可视化保存。

主要功能:
1. 漏检分析: GT 存在但没有被正确检测 (max_IoU < 0.50)
2. 误检分析: 预测框不匹配任何 GT (max_IoU < 0.50)
3. 可视化: 在 RGB 和 IR 图像上绘制检测框和 GT 框

使用方法:
    python tools/analysis/analyze_map50_errors.py \
        --config configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py \
        --checkpoint work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_64.pth \
        --output-dir data/LLVIP/map50 \
        --score-thr 0.3

作者: Assistant
日期: 2026-01-24
"""

import argparse
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
import cv2

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint
from mmengine.registry import DefaultScope
from mmengine.dataset import Compose

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../..')))

# Fix for PyTorch 2.6+ weights_only=True default behavior
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# 导入 MODELS 注册器（使用 mmyolo 的注册器）
from mmyolo.registry import MODELS

# 导入自定义模块（确保所有模块被注册）
import yolo_world


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze mAP50 errors with visualization')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--output-dir', default='data/LLVIP/map50', help='Output directory')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold for detections')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for mAP50')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process (for debugging)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options')
    return parser.parse_args()


def compute_iou(box1, box2):
    """
    计算两个框的 IoU
    
    Args:
        box1: [x1, y1, x2, y2] 格式
        box2: [x1, y1, x2, y2] 格式
    
    Returns:
        float: IoU 值
    """
    # 计算交集
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def coco_to_xyxy(bbox):
    """将 COCO 格式 [x, y, w, h] 转换为 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def analyze_single_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5):
    """
    分析单张图像的漏检和误检
    
    Args:
        gt_boxes: list of [x1, y1, x2, y2]
        pred_boxes: list of [x1, y1, x2, y2]
        pred_scores: list of float
        iou_thr: IoU 阈值
    
    Returns:
        dict: {
            'missed_gt': [(gt_idx, gt_box, max_iou), ...],  # 漏检的 GT
            'false_pos': [(pred_idx, pred_box, score, max_iou), ...],  # 误检的预测
            'matched': [(gt_idx, pred_idx, iou), ...],  # 匹配成功的
        }
    """
    result = {
        'missed_gt': [],      # 漏检: GT 没有被匹配
        'false_pos': [],      # 误检: 预测不匹配任何 GT
        'matched': [],        # 正确匹配
        'gt_boxes': gt_boxes,
        'pred_boxes': pred_boxes,
        'pred_scores': pred_scores,
    }
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return result
    
    # 计算 IoU 矩阵
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0:
        # 没有 GT，所有预测都是误检
        for pred_idx, (pred_box, score) in enumerate(zip(pred_boxes, pred_scores)):
            result['false_pos'].append((pred_idx, pred_box, score, 0.0))
        return result
    
    if num_pred == 0:
        # 没有预测，所有 GT 都是漏检
        for gt_idx, gt_box in enumerate(gt_boxes):
            result['missed_gt'].append((gt_idx, gt_box, 0.0))
        return result
    
    # 计算 IoU 矩阵
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)
    
    # 贪婪匹配 (按置信度从高到低)
    gt_matched = [False] * num_gt
    pred_matched = [False] * num_pred
    
    # 按置信度排序预测框
    sorted_pred_indices = np.argsort(pred_scores)[::-1]
    
    for pred_idx in sorted_pred_indices:
        if pred_matched[pred_idx]:
            continue
        
        # 找到与该预测框 IoU 最大的 GT
        max_iou = 0
        max_gt_idx = -1
        for gt_idx in range(num_gt):
            if gt_matched[gt_idx]:
                continue
            if iou_matrix[gt_idx, pred_idx] > max_iou:
                max_iou = iou_matrix[gt_idx, pred_idx]
                max_gt_idx = gt_idx
        
        if max_iou >= iou_thr and max_gt_idx >= 0:
            # 匹配成功
            gt_matched[max_gt_idx] = True
            pred_matched[pred_idx] = True
            result['matched'].append((max_gt_idx, pred_idx, max_iou))
        else:
            # 误检
            best_iou_for_pred = np.max(iou_matrix[:, pred_idx])
            result['false_pos'].append((
                pred_idx, 
                pred_boxes[pred_idx], 
                pred_scores[pred_idx],
                best_iou_for_pred
            ))
    
    # 检查漏检的 GT
    for gt_idx in range(num_gt):
        if not gt_matched[gt_idx]:
            max_iou = np.max(iou_matrix[gt_idx, :]) if num_pred > 0 else 0.0
            result['missed_gt'].append((gt_idx, gt_boxes[gt_idx], max_iou))
    
    return result


def draw_boxes_on_image(image, analysis_result, draw_all=True):
    """
    在图像上绘制检测框
    
    Args:
        image: numpy array (BGR)
        analysis_result: analyze_single_image 的返回结果
        draw_all: 是否绘制所有框（包括正确匹配的）
    
    Returns:
        image: 绘制后的图像
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    gt_boxes = analysis_result['gt_boxes']
    pred_boxes = analysis_result['pred_boxes']
    pred_scores = analysis_result['pred_scores']
    
    # 绘制正确匹配的框 (蓝色 GT, 绿色预测)
    if draw_all:
        for gt_idx, pred_idx, iou in analysis_result['matched']:
            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]
            score = pred_scores[pred_idx]
            
            # GT 框 - 蓝色
            cv2.rectangle(img, 
                         (int(gt_box[0]), int(gt_box[1])), 
                         (int(gt_box[2]), int(gt_box[3])), 
                         (255, 0, 0), thickness)
            
            # 预测框 - 绿色
            cv2.rectangle(img, 
                         (int(pred_box[0]), int(pred_box[1])), 
                         (int(pred_box[2]), int(pred_box[3])), 
                         (0, 255, 0), thickness)
            
            # 标注
            label = f'IoU:{iou:.2f} conf:{score:.2f}'
            cv2.putText(img, label, 
                       (int(pred_box[0]), int(pred_box[1]) - 5),
                       font, font_scale, (0, 255, 0), 1)
    
    # 绘制漏检的 GT (红色框)
    for gt_idx, gt_box, max_iou in analysis_result['missed_gt']:
        # 红色框 - 漏检
        pt1 = (int(gt_box[0]), int(gt_box[1]))
        pt2 = (int(gt_box[2]), int(gt_box[3]))
        
        # 绘制实线框
        color = (0, 0, 255)  # 红色
        cv2.rectangle(img, pt1, pt2, color, thickness + 1)
        
        # 在框内画 X
        cv2.line(img, pt1, pt2, color, 2)
        cv2.line(img, (pt1[0], pt2[1]), (pt2[0], pt1[1]), color, 2)
        
        # 标注
        label = f'MISS max_IoU:{max_iou:.2f}'
        # 添加背景
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 10), (pt1[0] + text_w, pt1[1]), (0, 0, 255), -1)
        cv2.putText(img, label, 
                   (int(gt_box[0]), int(gt_box[1]) - 5),
                   font, font_scale, (255, 255, 255), 1)
    
    # 绘制误检的预测框 (黄色)
    for pred_idx, pred_box, score, max_iou in analysis_result['false_pos']:
        # 黄色框 - 误检
        pt1 = (int(pred_box[0]), int(pred_box[1]))
        pt2 = (int(pred_box[2]), int(pred_box[3]))
        cv2.rectangle(img, pt1, pt2, (0, 255, 255), thickness + 1)
        
        # 标注
        label = f'FP IoU:{max_iou:.2f} conf:{score:.2f}'
        # 添加背景
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 10), (pt1[0] + text_w, pt1[1]), (0, 255, 255), -1)
        cv2.putText(img, label, 
                   (int(pred_box[0]), int(pred_box[1]) - 5),
                   font, font_scale, (0, 0, 0), 1)
    
    # 添加图例
    legend_y = 30
    cv2.rectangle(img, (5, 10), (250, 130), (0, 0, 0), -1)  # 背景
    cv2.putText(img, 'Legend:', (10, legend_y), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, 'Blue = GT (Ground Truth)', (10, legend_y + 22), font, 0.5, (255, 0, 0), 1)
    cv2.putText(img, 'Green = Matched Prediction', (10, legend_y + 44), font, 0.5, (0, 255, 0), 1)
    cv2.putText(img, 'Red+X = Missed GT (FN)', (10, legend_y + 66), font, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'Yellow = False Positive (FP)', (10, legend_y + 88), font, 0.5, (0, 255, 255), 1)
    
    return img


def build_model_and_pipeline(config_path, checkpoint, device='cuda:0', score_thr=0.3):
    """构建模型和数据处理 pipeline"""
    from mmengine.config import ConfigDict
    
    # 加载配置 (lazy_import=False 确保所有模块被注册)
    cfg = Config.fromfile(config_path, lazy_import=False)
    
    # 设置 DefaultScope
    DefaultScope.get_instance('mmyolo', scope_name='mmyolo')
    
    # ⭐ 设置 test_cfg（NMS、score threshold 等）
    # 这需要在 model 和 bbox_head 两个地方都设置
    test_cfg = ConfigDict(
        multi_label=True,
        nms_pre=30000,
        score_thr=score_thr,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300
    )
    cfg.model.test_cfg = test_cfg
    
    # ⭐ 关键：也要在 bbox_head 配置中设置 test_cfg
    if 'bbox_head' in cfg.model:
        cfg.model.bbox_head.test_cfg = test_cfg
    
    print(f"[DEBUG] Setting test_cfg: score_thr={score_thr}")
    
    # 构建模型
    model = MODELS.build(cfg.model)
    
    # 加载权重
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
    model.load_state_dict(state_dict, strict=False)
    
    # 设置类别信息
    model.CLASSES = ['person']  # LLVIP 只有 person 类
    
    model.to(device)
    model.eval()
    
    # 验证 test_cfg 已设置
    if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'test_cfg'):
        print(f"[DEBUG] bbox_head.test_cfg: {model.bbox_head.test_cfg}")
    else:
        print(f"[WARNING] bbox_head.test_cfg not found!")
    
    # 构建 test pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    
    return model, test_pipeline, cfg


def setup_text_embeddings(model, device='cuda:0'):
    """
    设置文本 embeddings (只需调用一次)
    """
    texts = [['person']]
    
    with torch.no_grad():
        if hasattr(model, 'reparameterize'):
            model.reparameterize(texts)
        elif hasattr(model.backbone, 'text_model') and model.backbone.text_model is not None:
            # 手动设置文本特征
            txt_feats = model.backbone.forward_text(texts)
            model.text_feats = txt_feats
            model.texts = texts
        else:
            # 如果没有 text_model，直接设置
            model.texts = texts


def inference_single_image(model, test_pipeline, rgb_path, class_texts, device='cuda:0'):
    """
    对单张图像进行推理（支持双模态 RGB + IR）
    
    Returns:
        pred_boxes: list of [x1, y1, x2, y2]
        pred_scores: list of float
    """
    # 准备数据
    data = {
        'img_path': rgb_path,
        'img_id': 0,
        'texts': class_texts,  # 添加文本类别
    }
    
    # 运行 pipeline
    data = test_pipeline(data)
    
    # 准备 batch（确保转换为 float32）
    inputs = data['inputs'].unsqueeze(0).to(device).float()
    inputs_ir = None
    if 'inputs_ir' in data:
        inputs_ir = data['inputs_ir'].unsqueeze(0).to(device).float()
    
    data_samples = [data['data_samples']]
    
    with torch.no_grad():
        # 推理 (text embeddings 已经在 setup_text_embeddings 中设置)
        results = model(
            inputs,
            data_samples,
            mode='predict',
            inputs_ir=inputs_ir,
        )
    
    # 解析结果
    pred_instances = results[0].pred_instances
    
    # 获取框和分数
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    
    # 调试：打印前几个预测的 scores（只打印一次）
    if not hasattr(inference_single_image, '_debug_printed'):
        print(f"[DEBUG] First image predictions:")
        print(f"  - Number of predictions: {len(scores)}")
        print(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]" if len(scores) > 0 else "  - No predictions")
        print(f"  - First 5 scores: {scores[:5].tolist() if len(scores) >= 5 else scores.tolist()}")
        inference_single_image._debug_printed = True
    
    # 转换为列表
    pred_boxes = [box.tolist() for box in bboxes]
    pred_scores = scores.tolist()
    
    return pred_boxes, pred_scores


def main():
    args = parse_args()
    
    print("=" * 70)
    print("mAP50 Error Analysis Tool")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Score Threshold: {args.score_thr}")
    print(f"IoU Threshold: {args.iou_thr}")
    print("=" * 70)
    
    # 加载类别文本
    class_text_path = 'data/llvip/texts/llvip_class_texts.json'
    if osp.exists(class_text_path):
        with open(class_text_path, 'r') as f:
            class_texts = json.load(f)
        print(f"Loaded class texts from: {class_text_path}")
    else:
        # LLVIP 默认只有 person 类
        class_texts = [['person']]
        print(f"Using default class texts: {class_texts}")
    
    # 构建模型 (这里会加载配置)
    print("\nBuilding model...")
    model, test_pipeline, cfg = build_model_and_pipeline(
        args.config, args.checkpoint, args.device, score_thr=args.score_thr
    )
    print(f"Model loaded successfully! (score_thr={args.score_thr})")
    
    # 设置文本 embeddings (只需一次)
    print("Setting up text embeddings...")
    setup_text_embeddings(model, args.device)
    print("Text embeddings ready!")
    
    # 获取数据路径
    data_root = cfg.get('data_root', 'data/LLVIP/')
    val_ann_file = cfg.get('val_ann_file', 'coco_annotations/test.json')
    val_data_prefix = cfg.get('val_data_prefix', 'visible/test/')
    
    ann_file = osp.join(data_root, val_ann_file)
    rgb_img_dir = osp.join(data_root, val_data_prefix)
    ir_img_dir = rgb_img_dir.replace('visible', 'infrared')
    
    print(f"\nAnnotation file: {ann_file}")
    print(f"RGB image dir: {rgb_img_dir}")
    print(f"IR image dir: {ir_img_dir}")
    
    # 加载 COCO 标注 (直接读取 JSON，避免 pycocotools 的 ID 类型问题)
    print("\nLoading annotations...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # 构建图像信息字典和标注字典
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # 构建每张图像的标注列表
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    img_ids = list(images_dict.keys())
    
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    
    print(f"Total images to process: {len(img_ids)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    missed_dir = osp.join(args.output_dir, 'missed')  # 漏检文件夹
    false_pos_dir = osp.join(args.output_dir, 'false_pos')  # 误检文件夹
    os.makedirs(missed_dir, exist_ok=True)
    os.makedirs(false_pos_dir, exist_ok=True)
    
    # 统计变量
    total_images = 0
    images_with_errors = 0
    total_missed_gt = 0
    total_false_pos = 0
    error_summary = []
    
    print("\nProcessing images...")
    for img_id in tqdm(img_ids):
        img_info = images_dict[img_id]
        img_name = img_info['file_name']
        
        # 图像路径
        rgb_path = osp.join(rgb_img_dir, img_name)
        ir_path = osp.join(ir_img_dir, img_name)
        
        if not osp.exists(rgb_path):
            print(f"Warning: RGB image not found: {rgb_path}")
            continue
        
        # 获取 GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        # 推理
        try:
            pred_boxes, pred_scores = inference_single_image(
                model, test_pipeline, rgb_path, class_texts, args.device
            )
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 过滤低置信度预测
        filtered_boxes = []
        filtered_scores = []
        for box, score in zip(pred_boxes, pred_scores):
            if score >= args.score_thr:
                filtered_boxes.append(box)
                filtered_scores.append(score)
        
        pred_boxes = filtered_boxes
        pred_scores = filtered_scores
        
        # 分析
        analysis = analyze_single_image(gt_boxes, pred_boxes, pred_scores, args.iou_thr)
        
        total_images += 1
        
        # 检查是否有错误
        has_missed = len(analysis['missed_gt']) > 0
        has_false_pos = len(analysis['false_pos']) > 0
        has_error = has_missed or has_false_pos
        
        if has_error:
            images_with_errors += 1
            total_missed_gt += len(analysis['missed_gt'])
            total_false_pos += len(analysis['false_pos'])
            
            # 记录错误摘要
            error_summary.append({
                'img_name': img_name,
                'num_gt': len(gt_boxes),
                'num_pred': len(pred_boxes),
                'num_missed': len(analysis['missed_gt']),
                'num_false_pos': len(analysis['false_pos']),
                'missed_details': [(gt_idx, float(max_iou)) for gt_idx, _, max_iou in analysis['missed_gt']],
                'fp_details': [(float(score), float(max_iou)) for _, _, score, max_iou in analysis['false_pos']],
            })
            
            # 读取图像
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path) if osp.exists(ir_path) else None
            
            # 绘制框
            rgb_vis = draw_boxes_on_image(rgb_img, analysis)
            ir_vis = draw_boxes_on_image(ir_img, analysis) if ir_img is not None else None
            
            base_name = osp.splitext(img_name)[0]
            
            # 保存到漏检文件夹
            if has_missed:
                rgb_out_path = osp.join(missed_dir, f'{base_name}.jpg')
                cv2.imwrite(rgb_out_path, rgb_vis)
                if ir_vis is not None:
                    ir_out_path = osp.join(missed_dir, f'{base_name}_ir.jpg')
                    cv2.imwrite(ir_out_path, ir_vis)
            
            # 保存到误检文件夹
            if has_false_pos:
                rgb_out_path = osp.join(false_pos_dir, f'{base_name}.jpg')
                cv2.imwrite(rgb_out_path, rgb_vis)
                if ir_vis is not None:
                    ir_out_path = osp.join(false_pos_dir, f'{base_name}_ir.jpg')
                    cv2.imwrite(ir_out_path, ir_vis)
    
    # 统计两个文件夹中的图片数量
    missed_images = len([f for f in os.listdir(missed_dir) if f.endswith('.jpg') and not f.endswith('_ir.jpg')])
    false_pos_images = len([f for f in os.listdir(false_pos_dir) if f.endswith('.jpg') and not f.endswith('_ir.jpg')])
    
    # 打印统计结果
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print(f"Images with errors: {images_with_errors}")
    print(f"Total missed GT (漏检): {total_missed_gt}")
    print(f"Total false positives (误检): {total_false_pos}")
    if total_images > 0:
        print(f"\nError rate: {images_with_errors / total_images * 100:.2f}%")
    print(f"\nResults saved to:")
    print(f"  漏检 (Missed GT):     {missed_dir} ({missed_images} images)")
    print(f"  误检 (False Positive): {false_pos_dir} ({false_pos_images} images)")
    
    # 保存错误摘要
    summary_path = osp.join(args.output_dir, 'error_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': total_images,
            'images_with_errors': images_with_errors,
            'total_missed_gt': total_missed_gt,
            'total_false_pos': total_false_pos,
            'score_threshold': args.score_thr,
            'iou_threshold': args.iou_thr,
            'errors': error_summary
        }, f, indent=2)
    
    print(f"Error summary saved to: {summary_path}")
    
    # 打印 Top 10 问题图像
    if error_summary:
        print("\n" + "-" * 70)
        print("Top 10 Problem Images (by total errors):")
        print("-" * 70)
        sorted_errors = sorted(error_summary, 
                               key=lambda x: x['num_missed'] + x['num_false_pos'],
                               reverse=True)
        for i, err in enumerate(sorted_errors[:10]):
            print(f"{i+1:2d}. {err['img_name']}")
            print(f"    GT: {err['num_gt']}, Pred: {err['num_pred']}")
            print(f"    Missed: {err['num_missed']}, False Pos: {err['num_false_pos']}")


if __name__ == '__main__':
    main()
