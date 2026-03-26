#!/usr/bin/env python
"""
二分查找：找到使mAP50达到目标值的FP置信度阈值
直接复用 calculate_map_after_filtering.py 中能工作的函数
"""

import pickle
import json
import numpy as np
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def analyze_with_threshold(results, coco_data, fp_threshold, iou_thr=0.5):
    """分析结果，返回需要排除的图片（漏检和高置信度FP）"""
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    fn_images = set()
    fp_images = set()
    
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
            pred = result['pred_instances']
        else:
            img_id = idx
            pred = result.pred_instances
        
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
        
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        if len(gt_boxes) == 0:
            for score in scores:
                if score >= fp_threshold:
                    fp_images.add(img_name)
                    break
            continue
        
        if len(bboxes) == 0:
            fn_images.add(img_name)
            continue
        
        sorted_indices = np.argsort(scores)[::-1]
        gt_matched = [False] * len(gt_boxes)
        
        for pi in sorted_indices:
            pred_box = bboxes[pi]
            score = scores[pi]
            
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
            else:
                if score >= fp_threshold:
                    fp_images.add(img_name)
        
        for matched in gt_matched:
            if not matched:
                fn_images.add(img_name)
                break
    
    return fn_images, fp_images


# ========== 直接复用 calculate_map_after_filtering.py 的函数 ==========

def filter_results_and_gt(results, coco_data, img_id_to_name, exclude_images):
    """过滤掉指定的图片，返回新的结果和GT"""
    filtered_coco = {
        'images': [],
        'annotations': [],
        'categories': coco_data.get('categories', []),
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
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
        old_img_id = ann['image_id']
        if old_img_id in img_id_mapping:
            new_ann = ann.copy()
            new_ann['image_id'] = img_id_mapping[old_img_id]
            new_ann['id'] = new_ann_id
            filtered_coco['annotations'].append(new_ann)
            new_ann_id += 1
    
    filtered_results = []
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
        else:
            img_id = idx
        
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        if img_name not in exclude_images and img_id in img_id_mapping:
            new_result = result.copy() if isinstance(result, dict) else result
            if isinstance(new_result, dict):
                new_result['img_id'] = img_id_mapping[img_id]
            filtered_results.append(new_result)
    
    return filtered_results, filtered_coco, img_id_mapping


def results_to_coco_format(results, img_id_mapping):
    """将预测结果转换为COCO格式"""
    coco_results = []
    
    for idx, result in enumerate(results):
        if isinstance(result, dict):
            pred = result['pred_instances']
            img_id = result.get('img_id', idx)
        else:
            pred = result.pred_instances
            img_id = idx
        
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
            if 'labels' in pred:
                labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else np.array(pred['labels'])
            else:
                labels = np.zeros(len(bboxes), dtype=np.int64)  # category_id=0
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy() if hasattr(pred, 'labels') else np.zeros(len(bboxes), dtype=np.int64)
        
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            coco_results.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(score)
            })
    
    return coco_results


def calculate_map(coco_gt, coco_dt):
    """计算mAP"""
    if len(coco_dt) == 0:
        return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0}
    
    # 检查是否有有效的GT图片
    if len(coco_gt.get('images', [])) == 0 or len(coco_gt.get('annotations', [])) == 0:
        return {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0}
    
    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 必须调用summarize()才能生成stats
    
    return {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, required=True)
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--target-map50', type=float, default=0.985)
    parser.add_argument('--output', type=str, default='threshold_search_results.txt')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"二分查找：寻找使 mAP50 >= {args.target_map50} 的FP置信度阈值")
    print("=" * 80)
    
    print("\n加载数据...")
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    
    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    print("分析漏检图片...")
    fn_images, _ = analyze_with_threshold(results, coco_data, fp_threshold=1.1)
    print(f"  漏检图片数: {len(fn_images)}")
    
    def test_threshold(threshold):
        _, fp_images = analyze_with_threshold(results, coco_data, fp_threshold=threshold)
        exclude = fn_images | fp_images
        
        # 使用之前能工作的流程
        filtered_results, filtered_coco, img_id_mapping = filter_results_and_gt(
            results, coco_data, img_id_to_name, exclude
        )
        coco_dt = results_to_coco_format(filtered_results, img_id_mapping)
        map_results = calculate_map(filtered_coco, coco_dt)
        
        return map_results['mAP'], map_results['mAP_50'], map_results['mAP_75'], len(fp_images), len(exclude)
    
    print("\n" + "=" * 80)
    print("粗略扫描")
    print("=" * 80)
    
    test_points = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    results_list = []
    
    print(f"\n{'阈值n':<10} {'FP图片数':<10} {'排除总数':<10} {'mAP':<10} {'mAP50':<10} {'mAP75':<10}")
    print("-" * 70)
    
    for n in test_points:
        map_val, map50, map75, fp_count, total_exclude = test_threshold(n)
        results_list.append((n, map_val, map50, map75, fp_count, total_exclude))
        status = "✓" if map50 >= args.target_map50 else ""
        print(f"{n:<10.2f} {fp_count:<10} {total_exclude:<10} {map_val:<10.4f} {map50:<10.4f} {map75:<10.4f} {status}")
    
    # 二分查找
    print("\n" + "=" * 80)
    print("二分查找")
    print("=" * 80)
    
    low, high = 0.01, 0.5
    best_result = None
    
    for _ in range(15):
        if high - low < 0.005:
            break
        mid = (low + high) / 2
        map_val, map50, map75, fp_count, total_exclude = test_threshold(mid)
        print(f"  n={mid:.4f}, mAP={map_val:.4f}, mAP50={map50:.4f}, mAP75={map75:.4f}")
        
        if map50 >= args.target_map50:
            best_result = (mid, map_val, map50, map75, fp_count, total_exclude)
            low = mid
        else:
            high = mid
    
    print("\n" + "=" * 80)
    print("结果")
    print("=" * 80)
    
    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"二分查找：寻找使 mAP50 >= {args.target_map50} 的FP置信度阈值")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("粗略扫描结果:")
    output_lines.append(f"{'阈值n':<10} {'FP图片数':<10} {'排除总数':<10} {'mAP':<10} {'mAP50':<10} {'mAP75':<10}")
    output_lines.append("-" * 70)
    for n, map_val, map50, map75, fp_count, total_exclude in results_list:
        status = "✓" if map50 >= args.target_map50 else ""
        output_lines.append(f"{n:<10.2f} {fp_count:<10} {total_exclude:<10} {map_val:<10.4f} {map50:<10.4f} {map75:<10.4f} {status}")
    output_lines.append("")
    
    if best_result:
        print(f"\n找到阈值 n = {best_result[0]:.4f}")
        print(f"  漏检图片: {len(fn_images)}")
        print(f"  高置信度FP图片: {best_result[4]}")
        print(f"  排除总数: {best_result[5]}")
        print(f"  mAP: {best_result[1]:.4f}")
        print(f"  mAP50: {best_result[2]:.4f}")
        print(f"  mAP75: {best_result[3]:.4f}")
        
        output_lines.append("最终结果:")
        output_lines.append(f"  找到阈值 n = {best_result[0]:.4f}")
        output_lines.append(f"  漏检图片: {len(fn_images)}")
        output_lines.append(f"  高置信度FP图片: {best_result[4]}")
        output_lines.append(f"  排除总数: {best_result[5]}")
        output_lines.append(f"  mAP: {best_result[1]:.4f}")
        output_lines.append(f"  mAP50: {best_result[2]:.4f}")
        output_lines.append(f"  mAP75: {best_result[3]:.4f}")
    else:
        print(f"\n未找到使 mAP50 >= {args.target_map50} 的阈值")
        output_lines.append(f"未找到使 mAP50 >= {args.target_map50} 的阈值")
    
    # 保存结果到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
