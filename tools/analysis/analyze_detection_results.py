#!/usr/bin/env python3
"""
检测结果分析工具

分析哪些图像的 mAP50 vs mAP75 差距大，找出定位不准的样本。

使用方法:
    python tools/analysis/analyze_detection_results.py \
        --config configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py \
        --checkpoint work_dirs/LLVIP/no_update/senet_fused_only/batch16_1280_2gpu/best_xxx.pth \
        --output-dir work_dirs/LLVIP/analysis
"""

import argparse
import os
import os.path as osp
import json
import numpy as np
from collections import defaultdict

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmdet.evaluation.functional import eval_map
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../..')))


def compute_per_image_metrics(coco_gt, coco_dt, iou_thrs=[0.5, 0.75]):
    """
    计算每张图像在不同 IoU 阈值下的 AP
    
    Returns:
        dict: {img_id: {'ap50': float, 'ap75': float, 'gap': float}}
    """
    results = {}
    
    # 获取所有图像 ID
    img_ids = coco_gt.getImgIds()
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        
        # 获取该图像的 GT 和预测
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        
        if len(gt_anns) == 0:
            continue
        
        # 计算每个 IoU 阈值下的指标
        metrics = {'img_id': img_id, 'img_name': img_name, 'num_gt': len(gt_anns), 'num_dt': len(dt_anns)}
        
        for iou_thr in iou_thrs:
            # 简化版：计算 TP 比例
            tp = 0
            for gt in gt_anns:
                gt_box = gt['bbox']  # [x, y, w, h]
                best_iou = 0
                for dt in dt_anns:
                    dt_box = dt['bbox']
                    iou = compute_iou(gt_box, dt_box)
                    best_iou = max(best_iou, iou)
                if best_iou >= iou_thr:
                    tp += 1
            
            recall = tp / len(gt_anns) if len(gt_anns) > 0 else 0
            metrics[f'recall@{int(iou_thr*100)}'] = recall
        
        # 计算 gap (mAP50 - mAP75 的差距)
        metrics['gap'] = metrics.get('recall@50', 0) - metrics.get('recall@75', 0)
        
        results[img_id] = metrics
    
    return results


def compute_iou(box1, box2):
    """计算两个框的 IoU (COCO 格式: [x, y, w, h])"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 转换为 [x1, y1, x2, y2]
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
    
    # 计算交集
    inter_x1 = max(box1_xyxy[0], box2_xyxy[0])
    inter_y1 = max(box1_xyxy[1], box2_xyxy[1])
    inter_x2 = min(box1_xyxy[2], box2_xyxy[2])
    inter_y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def analyze_localization_errors(coco_gt, coco_dt, output_dir):
    """分析定位误差，找出问题样本"""
    
    print("=" * 60)
    print("Analyzing Detection Results")
    print("=" * 60)
    
    # 计算每张图像的指标
    results = compute_per_image_metrics(coco_gt, coco_dt)
    
    # 按 gap (mAP50 - mAP75 差距) 排序
    sorted_results = sorted(results.values(), key=lambda x: -x['gap'])
    
    # 输出问题最大的图像
    print("\n" + "=" * 60)
    print("Top 20 Images with Largest Localization Gap (recall@50 - recall@75)")
    print("这些图像检测到了目标，但定位不够精确")
    print("=" * 60)
    
    problem_images = []
    for i, r in enumerate(sorted_results[:20]):
        print(f"{i+1:3d}. {r['img_name']}")
        print(f"     GT: {r['num_gt']}, DT: {r['num_dt']}")
        print(f"     Recall@50: {r['recall@50']:.3f}, Recall@75: {r['recall@75']:.3f}")
        print(f"     Gap: {r['gap']:.3f}")
        print()
        problem_images.append(r)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存问题图像列表
    with open(osp.join(output_dir, 'localization_problem_images.json'), 'w') as f:
        json.dump(problem_images, f, indent=2)
    
    # 保存所有结果
    with open(osp.join(output_dir, 'all_image_metrics.json'), 'w') as f:
        json.dump(list(results.values()), f, indent=2)
    
    # 统计分析
    gaps = [r['gap'] for r in results.values()]
    print("\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)
    print(f"Total images: {len(gaps)}")
    print(f"Mean gap: {np.mean(gaps):.4f}")
    print(f"Std gap: {np.std(gaps):.4f}")
    print(f"Max gap: {np.max(gaps):.4f}")
    print(f"Images with gap > 0.3: {sum(1 for g in gaps if g > 0.3)}")
    print(f"Images with gap > 0.5: {sum(1 for g in gaps if g > 0.5)}")
    
    # 按问题类型分类
    recall50_low = [r for r in results.values() if r['recall@50'] < 0.5]
    recall75_low = [r for r in results.values() if r['recall@75'] < 0.5 and r['recall@50'] >= 0.8]
    
    print(f"\nImages with low recall@50 (<0.5): {len(recall50_low)}")
    print("  → 模型难以检测这些目标 (漏检问题)")
    
    print(f"\nImages with high recall@50 but low recall@75: {len(recall75_low)}")
    print("  → 模型检测到了但定位不准 (定位问题)")
    
    # 保存分类结果
    with open(osp.join(output_dir, 'detection_problem_images.json'), 'w') as f:
        json.dump([r['img_name'] for r in recall50_low], f, indent=2)
    
    with open(osp.join(output_dir, 'localization_problem_images_list.txt'), 'w') as f:
        for r in sorted_results[:50]:
            f.write(f"{r['img_name']}\n")
    
    print(f"\n✅ Results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze detection results')
    parser.add_argument('--gt-ann', required=True, help='Ground truth annotation file (COCO format)')
    parser.add_argument('--dt-ann', required=True, help='Detection results file (COCO format)')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    # 加载 GT 和预测结果
    print(f"Loading GT: {args.gt_ann}")
    coco_gt = COCO(args.gt_ann)
    
    print(f"Loading DT: {args.dt_ann}")
    coco_dt = coco_gt.loadRes(args.dt_ann)
    
    # 分析
    analyze_localization_errors(coco_gt, coco_dt, args.output_dir)


if __name__ == '__main__':
    main()
