#!/usr/bin/env python
"""
使用正确的匹配算法（按置信度排序）重新分析并计算mAP
"""

import pickle
import json
import numpy as np
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def analyze_with_correct_matching(results, coco_data, iou_thr=0.5):
    """
    使用正确的匹配算法分析结果
    关键改动：按置信度从高到低排序后再匹配
    """
    # 构建映射
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # FN 图片集合
    fn_images = set()
    # 高置信度FP图片集合 (score >= 0.5)
    high_conf_fp_images = set()
    
    for idx, result in enumerate(results):
        # 获取图片信息
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
        
        # 获取 GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        num_gt = len(gt_boxes)
        num_pred = len(bboxes)
        
        if num_gt == 0:
            # 没有 GT，所有预测都是 FP
            for score in scores:
                if score >= 0.5:
                    high_conf_fp_images.add(img_name)
                    break
            continue
        
        if num_pred == 0:
            # 没有预测，所有 GT 都是 FN
            fn_images.add(img_name)
            continue
        
        # ========== 关键改动：按置信度从高到低排序 ==========
        sorted_indices = np.argsort(scores)[::-1]  # 按置信度降序
        
        gt_matched = [False] * num_gt
        
        # 按置信度顺序处理每个预测
        for pi in sorted_indices:
            pred_box = bboxes[pi]
            score = scores[pi]
            
            # 找到与该预测IoU最大且未匹配的GT
            best_iou = 0
            best_gt_idx = -1
            
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(gt, pred_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            
            # 判断是TP还是FP
            if best_iou >= iou_thr:
                # TP
                gt_matched[best_gt_idx] = True
            else:
                # FP
                if score >= 0.5:
                    high_conf_fp_images.add(img_name)
        
        # 检查是否有漏检
        for gi, matched in enumerate(gt_matched):
            if not matched:
                fn_images.add(img_name)
                break
    
    return fn_images, high_conf_fp_images


def filter_and_calculate_map(results, coco_data, exclude_images, img_id_to_name):
    """过滤图片并计算mAP"""
    
    # 创建新的COCO数据
    filtered_coco = {
        'images': [],
        'annotations': [],
        'categories': coco_data.get('categories', []),
    }
    
    img_id_mapping = {}
    new_img_id = 1
    
    for img in coco_data['images']:
        if img['file_name'] not in exclude_images:
            old_id = img['id']
            new_img = img.copy()
            new_img['id'] = new_img_id
            filtered_coco['images'].append(new_img)
            img_id_mapping[old_id] = new_img_id
            new_img_id += 1
    
    new_ann_id = 1
    for ann in coco_data.get('annotations', []):
        if ann['image_id'] in img_id_mapping:
            new_ann = ann.copy()
            new_ann['image_id'] = img_id_mapping[ann['image_id']]
            new_ann['id'] = new_ann_id
            filtered_coco['annotations'].append(new_ann)
            new_ann_id += 1
    
    # 转换预测结果为COCO格式
    coco_results = []
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
            pred = result['pred_instances']
        else:
            img_id = idx
            pred = result.pred_instances
        
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        if img_name in exclude_images or img_id not in img_id_mapping:
            continue
        
        new_img_id = img_id_mapping[img_id]
        
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
            labels = pred.get('labels', np.ones(len(bboxes)))
            if hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy() if hasattr(pred, 'labels') else np.ones(len(bboxes))
        
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            coco_results.append({
                'image_id': new_img_id,
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(score)
            })
    
    # 计算mAP
    if len(coco_results) == 0:
        return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0}
    
    coco_gt = COCO()
    coco_gt.dataset = filtered_coco
    coco_gt.createIndex()
    
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-score0', type=str, required=True)
    parser.add_argument('--pkl-score0001', type=str, required=True)
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--output', type=str, default='filtered_map_results_correct.txt')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("使用正确的匹配算法（按置信度排序）重新计算")
    print("=" * 80)
    
    # 加载标注
    print("\n加载标注文件...")
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 处理 score_thr=0
    print("\n" + "=" * 80)
    print("处理 score_thr=0")
    print("=" * 80)
    
    print("加载预测结果...")
    with open(args.pkl_score0, 'rb') as f:
        results_0 = pickle.load(f)
    
    print("使用正确的匹配算法分析...")
    fn_images_0, fp_images_0 = analyze_with_correct_matching(results_0, coco_data)
    
    exclude_0 = fn_images_0 | fp_images_0
    
    print(f"\n统计结果:")
    print(f"  漏检图片数: {len(fn_images_0)}")
    print(f"  高置信度FP图片数 (score>=0.5): {len(fp_images_0)}")
    print(f"  合并去重后排除: {len(exclude_0)}")
    
    print(f"\n计算排除后的mAP...")
    map_results_0 = filter_and_calculate_map(results_0, coco_data, exclude_0, img_id_to_name)
    
    print(f"\n结果:")
    print(f"  mAP:    {map_results_0['mAP']:.4f}")
    print(f"  mAP50:  {map_results_0['mAP_50']:.4f}")
    print(f"  mAP75:  {map_results_0['mAP_75']:.4f}")
    
    # 处理 score_thr=0.001
    print("\n" + "=" * 80)
    print("处理 score_thr=0.001")
    print("=" * 80)
    
    print("加载预测结果...")
    with open(args.pkl_score0001, 'rb') as f:
        results_0001 = pickle.load(f)
    
    print("使用正确的匹配算法分析...")
    fn_images_0001, fp_images_0001 = analyze_with_correct_matching(results_0001, coco_data)
    
    exclude_0001 = fn_images_0001 | fp_images_0001
    
    print(f"\n统计结果:")
    print(f"  漏检图片数: {len(fn_images_0001)}")
    print(f"  高置信度FP图片数 (score>=0.5): {len(fp_images_0001)}")
    print(f"  合并去重后排除: {len(exclude_0001)}")
    
    print(f"\n计算排除后的mAP...")
    map_results_0001 = filter_and_calculate_map(results_0001, coco_data, exclude_0001, img_id_to_name)
    
    print(f"\n结果:")
    print(f"  mAP:    {map_results_0001['mAP']:.4f}")
    print(f"  mAP50:  {map_results_0001['mAP_50']:.4f}")
    print(f"  mAP75:  {map_results_0001['mAP_75']:.4f}")
    
    # 验证高置信度FP是否一致
    print("\n" + "=" * 80)
    print("验证：高置信度FP应该一致")
    print("=" * 80)
    print(f"  score_thr=0 高置信度FP:     {len(fp_images_0)} 张")
    print(f"  score_thr=0.001 高置信度FP: {len(fp_images_0001)} 张")
    
    if fp_images_0 == fp_images_0001:
        print(f"  ✓ 完全一致！")
    else:
        diff = fp_images_0 ^ fp_images_0001  # 对称差集
        print(f"  差异: {len(diff)} 张")
        if len(fp_images_0 - fp_images_0001) > 0:
            print(f"    只在score_thr=0中: {len(fp_images_0 - fp_images_0001)} 张")
        if len(fp_images_0001 - fp_images_0) > 0:
            print(f"    只在score_thr=0.001中: {len(fp_images_0001 - fp_images_0)} 张")
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("使用正确匹配算法（按置信度排序）的结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【score_thr=0】\n")
        f.write("-" * 60 + "\n")
        f.write(f"漏检图片: {len(fn_images_0)} 张\n")
        f.write(f"高置信度FP图片 (score>=0.5): {len(fp_images_0)} 张\n")
        f.write(f"合并去重后排除: {len(exclude_0)} 张\n\n")
        f.write(f"mAP:    {map_results_0['mAP']:.4f}\n")
        f.write(f"mAP50:  {map_results_0['mAP_50']:.4f}\n")
        f.write(f"mAP75:  {map_results_0['mAP_75']:.4f}\n\n")
        
        f.write("【score_thr=0.001】\n")
        f.write("-" * 60 + "\n")
        f.write(f"漏检图片: {len(fn_images_0001)} 张\n")
        f.write(f"高置信度FP图片 (score>=0.5): {len(fp_images_0001)} 张\n")
        f.write(f"合并去重后排除: {len(exclude_0001)} 张\n\n")
        f.write(f"mAP:    {map_results_0001['mAP']:.4f}\n")
        f.write(f"mAP50:  {map_results_0001['mAP_50']:.4f}\n")
        f.write(f"mAP75:  {map_results_0001['mAP_75']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n结果已保存到: {args.output}")
    print("\n完成!")


if __name__ == '__main__':
    main()
