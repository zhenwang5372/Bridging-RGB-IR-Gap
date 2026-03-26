#!/usr/bin/env python
"""
可视化高置信度FP图片
- GT框：绿色，标注"GT"
- TP框：蓝色，标注置信度
- 高置信度FP框：红色，标注置信度
"""

import pickle
import json
import numpy as np
import cv2
import os
import argparse
from pathlib import Path


def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def coco_to_xyxy(bbox):
    """COCO 格式 [x, y, w, h] 转 [x1, y1, x2, y2]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def draw_box(img, bbox, color, label, thickness=2):
    """在图片上画框和标签"""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 计算标签位置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # 标签背景
    cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
    # 标签文字（白色）
    cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return img


def analyze_and_visualize(results, coco_data, rgb_dir, ir_dir, output_dir, fp_threshold=0.5, iou_thr=0.5):
    """分析并可视化高置信度FP"""
    
    # 构建映射
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色定义 (BGR)
    COLOR_GT = (0, 255, 0)      # 绿色 - GT
    COLOR_TP = (255, 0, 0)      # 蓝色 - TP
    COLOR_FP = (0, 0, 255)      # 红色 - FP
    
    fp_image_count = 0
    total_high_conf_fp = 0
    
    print(f"开始分析和可视化...")
    print(f"FP置信度阈值: {fp_threshold}")
    print(f"IoU阈值: {iou_thr}")
    print()
    
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
            pred = result['pred_instances']
        else:
            img_id = idx
            pred = result.pred_instances
        
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        # 获取预测
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
        
        # 获取GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        num_gt = len(gt_boxes)
        num_pred = len(bboxes)
        
        # 使用正确的匹配算法（按置信度排序）
        sorted_indices = np.argsort(scores)[::-1]
        gt_matched = [False] * num_gt
        pred_is_tp = [False] * num_pred
        pred_matched_gt = [-1] * num_pred  # 记录每个预测匹配的GT索引
        
        # 按置信度顺序匹配
        for pi in sorted_indices:
            pred_box = bboxes[pi]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(gt, pred_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            
            if best_iou >= iou_thr:
                gt_matched[best_gt_idx] = True
                pred_is_tp[pi] = True
                pred_matched_gt[pi] = best_gt_idx
        
        # 检查是否有高置信度FP
        high_conf_fps = []
        for pi in range(num_pred):
            if not pred_is_tp[pi] and scores[pi] >= fp_threshold:
                high_conf_fps.append((pi, scores[pi], bboxes[pi]))
        
        if len(high_conf_fps) == 0:
            continue
        
        fp_image_count += 1
        total_high_conf_fp += len(high_conf_fps)
        
        # 读取图片
        rgb_path = os.path.join(rgb_dir, img_name)
        ir_path = os.path.join(ir_dir, img_name)
        
        if not os.path.exists(rgb_path):
            print(f"警告: RGB图片不存在: {rgb_path}")
            continue
        
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path) if os.path.exists(ir_path) else None
        
        # 在图片上画框
        for img, suffix in [(rgb_img, ''), (ir_img, '_ir')]:
            if img is None:
                continue
            
            img_draw = img.copy()
            
            # 1. 画GT框（绿色）
            for gi, gt in enumerate(gt_boxes):
                draw_box(img_draw, gt, COLOR_GT, "GT", thickness=2)
            
            # 2. 画TP框（蓝色，显示置信度）
            for pi in range(num_pred):
                if pred_is_tp[pi]:
                    score = scores[pi]
                    bbox = bboxes[pi]
                    label = f"TP:{score:.2f}"
                    draw_box(img_draw, bbox, COLOR_TP, label, thickness=2)
            
            # 3. 画高置信度FP框（红色，显示置信度）
            for pi, score, bbox in high_conf_fps:
                label = f"FP:{score:.2f}"
                draw_box(img_draw, bbox, COLOR_FP, label, thickness=3)
            
            # 保存
            base_name = os.path.splitext(img_name)[0]
            output_name = f"{base_name}{suffix}.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, img_draw)
        
        if fp_image_count % 20 == 0:
            print(f"已处理 {fp_image_count} 张含高置信度FP的图片...")
    
    print()
    print("=" * 60)
    print(f"可视化完成!")
    print(f"  含高置信度FP的图片数: {fp_image_count}")
    print(f"  高置信度FP总数: {total_high_conf_fp}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)
    
    return fp_image_count, total_high_conf_fp


def main():
    parser = argparse.ArgumentParser(description='可视化高置信度FP')
    parser.add_argument('--pkl', type=str, required=True, help='预测结果pkl文件')
    parser.add_argument('--ann-file', type=str, required=True, help='COCO标注文件')
    parser.add_argument('--rgb-dir', type=str, default='data/LLVIP/visible/test', help='RGB图片目录')
    parser.add_argument('--ir-dir', type=str, default='data/LLVIP/infrared/test', help='IR图片目录')
    parser.add_argument('--output-dir', type=str, default='tools/analysis/with_text_update/vis_high_conf_fp', 
                        help='输出目录')
    parser.add_argument('--fp-threshold', type=float, default=0.5, help='FP置信度阈值')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("可视化高置信度FP")
    print("=" * 60)
    print(f"PKL文件: {args.pkl}")
    print(f"标注文件: {args.ann_file}")
    print(f"RGB目录: {args.rgb_dir}")
    print(f"IR目录: {args.ir_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"FP置信度阈值: {args.fp_threshold}")
    print()
    
    # 加载数据
    print("加载预测结果...")
    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)
    
    print("加载标注文件...")
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"预测结果数: {len(results)}")
    print(f"图片数: {len(coco_data['images'])}")
    print()
    
    # 分析并可视化
    analyze_and_visualize(
        results, coco_data, 
        args.rgb_dir, args.ir_dir, args.output_dir,
        fp_threshold=args.fp_threshold
    )


if __name__ == '__main__':
    main()
