#!/usr/bin/env python3
"""
从 test.py --out 保存的 pkl 结果文件分析 mAP50 错误

使用方法:
1. 先运行测试保存结果:
   python tools/test.py configs/xxx.py checkpoint.pth --out results.pkl
   
2. 然后运行分析:
   python tools/analysis/analyze_from_pkl.py \
       --results results.pkl \
       --ann-file data/LLVIP/coco_annotations/test.json \
       --img-dir data/LLVIP/visible/test/ \
       --output-dir data/LLVIP/map50 \
       --iou-thr 0.5
"""

import argparse
import os
import os.path as osp
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze mAP50 errors from pkl results')
    parser.add_argument('--results', required=True, help='Path to results.pkl from test.py')
    parser.add_argument('--ann-file', required=True, help='Path to COCO annotation file')
    parser.add_argument('--img-dir', required=True, help='Directory containing images')
    parser.add_argument('--output-dir', required=True, help='Output directory for visualizations')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold for filtering predictions')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process')
    return parser.parse_args()


def compute_iou(box1, box2):
    """计算两个框的 IoU (boxes in xyxy format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def coco_to_xyxy(bbox):
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def analyze_single_image(gt_boxes, pred_boxes, pred_scores, iou_thr=0.5):
    """分析单张图像的检测结果"""
    result = {
        'matched': [],      # (gt_idx, pred_idx, iou, score)
        'missed_gt': [],    # (gt_idx, gt_box, max_iou)
        'false_pos': [],    # (pred_idx, pred_box, score, max_iou)
    }
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return result
    
    # 计算 IoU 矩阵
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0:
        # 所有预测都是 FP
        for pred_idx, (pred_box, score) in enumerate(zip(pred_boxes, pred_scores)):
            result['false_pos'].append((pred_idx, pred_box, score, 0.0))
        return result
    
    if num_pred == 0:
        # 所有 GT 都是漏检
        for gt_idx, gt_box in enumerate(gt_boxes):
            result['missed_gt'].append((gt_idx, gt_box, 0.0))
        return result
    
    # 计算 IoU 矩阵
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)
    
    # 贪婪匹配
    gt_matched = [False] * num_gt
    pred_matched = [False] * num_pred
    
    # 按 IoU 降序排列
    indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
    
    for gt_idx, pred_idx in zip(indices[0], indices[1]):
        if gt_matched[gt_idx] or pred_matched[pred_idx]:
            continue
        
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_thr:
            gt_matched[gt_idx] = True
            pred_matched[pred_idx] = True
            result['matched'].append((gt_idx, pred_idx, iou, pred_scores[pred_idx]))
    
    # 找出漏检的 GT
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            max_iou = iou_matrix[gt_idx].max() if num_pred > 0 else 0.0
            result['missed_gt'].append((gt_idx, gt_boxes[gt_idx], max_iou))
    
    # 找出误检的预测
    for pred_idx, matched in enumerate(pred_matched):
        if not matched:
            max_iou = iou_matrix[:, pred_idx].max() if num_gt > 0 else 0.0
            result['false_pos'].append((pred_idx, pred_boxes[pred_idx], pred_scores[pred_idx], max_iou))
    
    return result


def draw_boxes_on_image(img, analysis_result, gt_boxes):
    """在图像上绘制检测结果"""
    if img is None:
        return None
    
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    # 绘制所有 GT 框 (蓝色)
    for gt_idx, gt_box in enumerate(gt_boxes):
        cv2.rectangle(img, 
                     (int(gt_box[0]), int(gt_box[1])), 
                     (int(gt_box[2]), int(gt_box[3])), 
                     (255, 0, 0), thickness)
    
    # 绘制匹配的预测框 (绿色)
    for gt_idx, pred_idx, iou, score in analysis_result['matched']:
        pred_box = analysis_result.get('_pred_boxes', [])[pred_idx] if '_pred_boxes' in analysis_result else None
        if pred_box is not None:
            cv2.rectangle(img, 
                         (int(pred_box[0]), int(pred_box[1])), 
                         (int(pred_box[2]), int(pred_box[3])), 
                         (0, 255, 0), thickness)
            label = f'IoU:{iou:.2f}'
            cv2.putText(img, label, 
                       (int(pred_box[0]), int(pred_box[1]) - 5),
                       font, font_scale, (0, 255, 0), 1)
    
    # 绘制漏检的 GT (红色框 + X)
    for gt_idx, gt_box, max_iou in analysis_result['missed_gt']:
        pt1 = (int(gt_box[0]), int(gt_box[1]))
        pt2 = (int(gt_box[2]), int(gt_box[3]))
        color = (0, 0, 255)
        cv2.rectangle(img, pt1, pt2, color, thickness + 1)
        cv2.line(img, pt1, pt2, color, 2)
        cv2.line(img, (pt1[0], pt2[1]), (pt2[0], pt1[1]), color, 2)
        label = f'MISS max_IoU:{max_iou:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 10), (pt1[0] + text_w, pt1[1]), (0, 0, 255), -1)
        cv2.putText(img, label, (pt1[0], pt1[1] - 5), font, font_scale, (255, 255, 255), 1)
    
    # 绘制误检 (黄色)
    for pred_idx, pred_box, score, max_iou in analysis_result['false_pos']:
        pt1 = (int(pred_box[0]), int(pred_box[1]))
        pt2 = (int(pred_box[2]), int(pred_box[3]))
        cv2.rectangle(img, pt1, pt2, (0, 255, 255), thickness + 1)
        label = f'FP IoU:{max_iou:.2f} conf:{score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 10), (pt1[0] + text_w, pt1[1]), (0, 255, 255), -1)
        cv2.putText(img, label, (pt1[0], pt1[1] - 5), font, font_scale, (0, 0, 0), 1)
    
    # 添加图例
    legend_y = 30
    cv2.rectangle(img, (5, 10), (250, 130), (0, 0, 0), -1)
    cv2.putText(img, 'Legend:', (10, legend_y), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, 'Blue = GT', (10, legend_y + 22), font, 0.5, (255, 0, 0), 1)
    cv2.putText(img, 'Green = Matched', (10, legend_y + 44), font, 0.5, (0, 255, 0), 1)
    cv2.putText(img, 'Red+X = Missed GT', (10, legend_y + 66), font, 0.5, (0, 0, 255), 1)
    cv2.putText(img, 'Yellow = False Positive', (10, legend_y + 88), font, 0.5, (0, 255, 255), 1)
    
    return img


def main():
    args = parse_args()
    
    print("=" * 70)
    print("mAP50 Error Analysis (from pkl results)")
    print("=" * 70)
    print(f"Results file: {args.results}")
    print(f"Annotation file: {args.ann_file}")
    print(f"Image directory: {args.img_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Score threshold: {args.score_thr}")
    print(f"IoU threshold: {args.iou_thr}")
    print("=" * 70)
    
    # 加载检测结果
    print("\nLoading detection results...")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} results")
    
    # 加载标注
    print("Loading annotations...")
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images_dict = {img['id']: img for img in coco_data['images']}
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    missed_dir = osp.join(args.output_dir, 'missed')
    false_pos_dir = osp.join(args.output_dir, 'false_pos')
    os.makedirs(missed_dir, exist_ok=True)
    os.makedirs(false_pos_dir, exist_ok=True)
    
    # 统计
    total_images = 0
    images_with_errors = 0
    total_missed_gt = 0
    total_false_pos = 0
    
    # 处理结果（按 results 顺序，从结果中获取 img_id）
    num_to_process = min(len(results), args.max_images) if args.max_images else len(results)
    
    print(f"\nProcessing {num_to_process} images...")
    
    for idx in tqdm(range(num_to_process)):
        result = results[idx]
        
        # 从结果中获取 img_id 和图像路径
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
            img_path = result.get('img_path', '')
            img_name = osp.basename(img_path) if img_path else f'{img_id}.jpg'
        else:
            img_id = idx
            img_name = f'{img_id}.jpg'
        
        # 尝试从 images_dict 获取信息
        if img_id in images_dict:
            img_info = images_dict[img_id]
            img_name = img_info['file_name']
        
        # 获取检测结果
        result = results[idx]
        
        # 处理不同格式的结果 (dict 或 object)
        if isinstance(result, dict):
            pred_instances = result['pred_instances']
            if isinstance(pred_instances, dict):
                bboxes = pred_instances['bboxes'].cpu().numpy()
                scores = pred_instances['scores'].cpu().numpy()
            else:
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
        else:
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
        
        # 过滤低分预测
        mask = scores >= args.score_thr
        pred_boxes = bboxes[mask].tolist()
        pred_scores = scores[mask].tolist()
        
        # 打印第一张图的调试信息
        if idx == 0:
            print(f"\n[DEBUG] First image: {img_name}")
            print(f"  - Total predictions: {len(scores)}")
            print(f"  - After score filter ({args.score_thr}): {len(pred_scores)}")
            if len(scores) > 0:
                print(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                print(f"  - First 5 scores: {scores[:5].tolist()}")
        
        # 获取 GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        # 分析
        analysis = analyze_single_image(gt_boxes, pred_boxes, pred_scores, args.iou_thr)
        analysis['_pred_boxes'] = pred_boxes  # 保存用于可视化
        
        total_images += 1
        
        has_missed = len(analysis['missed_gt']) > 0
        has_false_pos = len(analysis['false_pos']) > 0
        has_error = has_missed or has_false_pos
        
        if has_error:
            images_with_errors += 1
            total_missed_gt += len(analysis['missed_gt'])
            total_false_pos += len(analysis['false_pos'])
            
            # 读取图像
            rgb_path = osp.join(args.img_dir, img_name)
            ir_path = rgb_path.replace('visible', 'infrared')
            
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path) if osp.exists(ir_path) else None
            
            # 绘制
            rgb_vis = draw_boxes_on_image(rgb_img, analysis, gt_boxes)
            ir_vis = draw_boxes_on_image(ir_img, analysis, gt_boxes) if ir_img is not None else None
            
            base_name = osp.splitext(img_name)[0]
            
            # 保存
            if has_missed:
                cv2.imwrite(osp.join(missed_dir, f'{base_name}.jpg'), rgb_vis)
                if ir_vis is not None:
                    cv2.imwrite(osp.join(missed_dir, f'{base_name}_ir.jpg'), ir_vis)
            
            if has_false_pos:
                cv2.imwrite(osp.join(false_pos_dir, f'{base_name}.jpg'), rgb_vis)
                if ir_vis is not None:
                    cv2.imwrite(osp.join(false_pos_dir, f'{base_name}_ir.jpg'), ir_vis)
    
    # 打印统计
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print(f"Images with errors: {images_with_errors}")
    print(f"Total missed GT: {total_missed_gt}")
    print(f"Total false positives: {total_false_pos}")
    print(f"\nResults saved to:")
    print(f"  Missed: {missed_dir}")
    print(f"  False positives: {false_pos_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
