#!/usr/bin/env python
"""
从日志文件中提取需要排除的图片，然后重新计算mAP、mAP50、mAP75
"""

import pickle
import json
import numpy as np
import argparse
import re
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_log_file(log_path):
    """
    从日志文件中解析出需要排除的图片列表
    返回: {
        'score_thr_0': {
            'fn_images': set(),
            'fp_high_conf_images': set()
        },
        'score_thr_0001': {
            'fn_images': set(),
            'fp_high_conf_images': set()
        }
    }
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {
        'score_thr_0': {
            'fn_images': set(),
            'fp_high_conf_images': set()
        },
        'score_thr_0001': {
            'fn_images': set(),
            'fp_high_conf_images': set()
        }
    }
    
    # 解析 score_thr=0 的漏检图片
    fn_pattern_0 = re.compile(r'【漏检】score_thr=0.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fn_match_0 = fn_pattern_0.search(content)
    if fn_match_0:
        fn_section = fn_match_0.group(1)
        # 提取所有图片名
        img_names = re.findall(r'(\d+\.jpg)', fn_section)
        result['score_thr_0']['fn_images'] = set(img_names)
    
    # 解析 score_thr=0.001 的漏检图片
    fn_pattern_0001 = re.compile(r'【漏检】score_thr=0\.001.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fn_match_0001 = fn_pattern_0001.search(content)
    if fn_match_0001:
        fn_section = fn_match_0001.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fn_section)
        result['score_thr_0001']['fn_images'] = set(img_names)
    
    # 解析 score_thr=0 的高置信度FP图片 (score >= 0.5)
    fp_pattern_0 = re.compile(r'【高置信度误检】score_thr=0.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fp_match_0 = fp_pattern_0.search(content)
    if fp_match_0:
        fp_section = fp_match_0.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fp_section)
        result['score_thr_0']['fp_high_conf_images'] = set(img_names)
    
    # 解析 score_thr=0.001 的高置信度FP图片 (score >= 0.5)
    fp_pattern_0001 = re.compile(r'【高置信度误检】score_thr=0\.001.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fp_match_0001 = fp_pattern_0001.search(content)
    if fp_match_0001:
        fp_section = fp_match_0001.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fp_section)
        result['score_thr_0001']['fp_high_conf_images'] = set(img_names)
    
    return result


def load_pkl_and_ann(pkl_path, ann_path):
    """加载pkl结果和标注文件"""
    print(f"加载预测结果: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"加载标注文件: {ann_path}")
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建映射
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    img_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    return results, coco_data, img_id_to_name, img_name_to_id


def filter_results_and_gt(results, coco_data, img_id_to_name, exclude_images):
    """
    过滤掉指定的图片，返回新的结果和GT
    """
    # 创建新的COCO数据
    filtered_coco = {
        'images': [],
        'annotations': [],
        'categories': coco_data.get('categories', []),
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }
    
    # 过滤图片和标注
    img_id_mapping = {}  # 旧id -> 新id
    new_img_id = 1
    
    for img in coco_data['images']:
        if img['file_name'] not in exclude_images:
            old_id = img['id']
            new_img = img.copy()
            new_img['id'] = new_img_id
            filtered_coco['images'].append(new_img)
            img_id_mapping[old_id] = new_img_id
            new_img_id += 1
    
    # 过滤标注
    new_ann_id = 1
    for ann in coco_data.get('annotations', []):
        old_img_id = ann['image_id']
        if old_img_id in img_id_mapping:
            new_ann = ann.copy()
            new_ann['image_id'] = img_id_mapping[old_img_id]
            new_ann['id'] = new_ann_id
            filtered_coco['annotations'].append(new_ann)
            new_ann_id += 1
    
    # 过滤预测结果
    filtered_results = []
    for idx, result in enumerate(results):
        # 获取图片ID
        if isinstance(result, dict):
            img_id = result.get('img_id', idx)
        else:
            img_id = idx
        
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        if img_name not in exclude_images and img_id in img_id_mapping:
            # 创建新的结果
            new_result = result.copy() if isinstance(result, dict) else result
            if isinstance(new_result, dict):
                new_result['img_id'] = img_id_mapping[img_id]
            filtered_results.append(new_result)
    
    return filtered_results, filtered_coco, img_id_mapping


def results_to_coco_format(results, img_id_mapping):
    """
    将预测结果转换为COCO格式用于评估
    """
    coco_results = []
    
    for idx, result in enumerate(results):
        # 获取预测
        if isinstance(result, dict):
            pred = result['pred_instances']
            img_id = result.get('img_id', idx)
        else:
            pred = result.pred_instances
            img_id = idx
        
        # 获取boxes和scores
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
            if 'labels' in pred:
                labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else np.array(pred['labels'])
            else:
                labels = np.ones(len(bboxes), dtype=np.int64)
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy() if hasattr(pred, 'labels') else np.ones(len(bboxes), dtype=np.int64)
        
        # 转换格式
        for bbox, score, label in zip(bboxes, scores, labels):
            # xyxy -> xywh
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            coco_results.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            })
    
    return coco_results


def calculate_map(coco_gt, coco_dt, iou_thresholds=None):
    """
    计算mAP指标
    """
    if len(coco_dt) == 0:
        print("警告: 没有检测结果")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0
        }
    
    # 创建COCO对象
    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    
    # 加载检测结果
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    
    # 创建评估器
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    
    # 设置IoU阈值
    if iou_thresholds is not None:
        coco_eval.params.iouThrs = iou_thresholds
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取结果
    results = {
        'mAP': coco_eval.stats[0],       # mAP @ IoU=0.50:0.95
        'mAP_50': coco_eval.stats[1],    # mAP @ IoU=0.50
        'mAP_75': coco_eval.stats[2],    # mAP @ IoU=0.75
        'AR_1': coco_eval.stats[6],      # AR @ maxDets=1
        'AR_10': coco_eval.stats[7],     # AR @ maxDets=10
        'AR_100': coco_eval.stats[8],    # AR @ maxDets=100
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='计算排除特定图片后的mAP')
    parser.add_argument('--log-file', type=str, required=True,
                        help='日志文件路径')
    parser.add_argument('--pkl-score0', type=str, required=True,
                        help='score_thr=0 的 pkl 文件路径')
    parser.add_argument('--pkl-score0001', type=str, required=True,
                        help='score_thr=0.001 的 pkl 文件路径')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='COCO标注文件路径')
    parser.add_argument('--output', type=str, 
                        default='filtered_map_results.txt',
                        help='输出结果文件路径')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("计算排除特定图片后的mAP")
    print("=" * 80)
    
    # 1. 解析日志文件，获取需要排除的图片
    print("\n步骤 1: 解析日志文件...")
    excluded_imgs = parse_log_file(args.log_file)
    
    print("\n排除的图片统计:")
    print(f"  score_thr=0:")
    print(f"    漏检图片数: {len(excluded_imgs['score_thr_0']['fn_images'])}")
    print(f"    高置信度FP图片数: {len(excluded_imgs['score_thr_0']['fp_high_conf_images'])}")
    
    # 合并去重
    exclude_score0 = excluded_imgs['score_thr_0']['fn_images'] | excluded_imgs['score_thr_0']['fp_high_conf_images']
    print(f"    合并后排除图片数: {len(exclude_score0)}")
    
    print(f"  score_thr=0.001:")
    print(f"    漏检图片数: {len(excluded_imgs['score_thr_0001']['fn_images'])}")
    print(f"    高置信度FP图片数: {len(excluded_imgs['score_thr_0001']['fp_high_conf_images'])}")
    
    exclude_score0001 = excluded_imgs['score_thr_0001']['fn_images'] | excluded_imgs['score_thr_0001']['fp_high_conf_images']
    print(f"    合并后排除图片数: {len(exclude_score0001)}")
    
    # 2. 处理 score_thr=0
    print("\n" + "=" * 80)
    print("处理 score_thr=0")
    print("=" * 80)
    
    results_0, coco_data, img_id_to_name, img_name_to_id = load_pkl_and_ann(
        args.pkl_score0, args.ann_file
    )
    
    print(f"\n原始数据:")
    print(f"  图片总数: {len(coco_data['images'])}")
    print(f"  标注总数: {len(coco_data['annotations'])}")
    print(f"  预测结果数: {len(results_0)}")
    
    print(f"\n过滤数据...")
    filtered_results_0, filtered_coco_0, img_id_mapping_0 = filter_results_and_gt(
        results_0, coco_data, img_id_to_name, exclude_score0
    )
    
    print(f"\n过滤后数据:")
    print(f"  图片总数: {len(filtered_coco_0['images'])}")
    print(f"  标注总数: {len(filtered_coco_0['annotations'])}")
    print(f"  预测结果数: {len(filtered_results_0)}")
    print(f"  排除的图片数: {len(exclude_score0)}")
    
    print(f"\n转换为COCO格式...")
    coco_dt_0 = results_to_coco_format(filtered_results_0, img_id_mapping_0)
    print(f"  检测结果数: {len(coco_dt_0)}")
    
    print(f"\n计算mAP...")
    map_results_0 = calculate_map(filtered_coco_0, coco_dt_0)
    
    print(f"\n结果:")
    print(f"  mAP:    {map_results_0['mAP']:.4f}")
    print(f"  mAP50:  {map_results_0['mAP_50']:.4f}")
    print(f"  mAP75:  {map_results_0['mAP_75']:.4f}")
    
    # 3. 处理 score_thr=0.001
    print("\n" + "=" * 80)
    print("处理 score_thr=0.001")
    print("=" * 80)
    
    results_0001, _, _, _ = load_pkl_and_ann(
        args.pkl_score0001, args.ann_file
    )
    
    print(f"\n原始数据:")
    print(f"  图片总数: {len(coco_data['images'])}")
    print(f"  标注总数: {len(coco_data['annotations'])}")
    print(f"  预测结果数: {len(results_0001)}")
    
    print(f"\n过滤数据...")
    filtered_results_0001, filtered_coco_0001, img_id_mapping_0001 = filter_results_and_gt(
        results_0001, coco_data, img_id_to_name, exclude_score0001
    )
    
    print(f"\n过滤后数据:")
    print(f"  图片总数: {len(filtered_coco_0001['images'])}")
    print(f"  标注总数: {len(filtered_coco_0001['annotations'])}")
    print(f"  预测结果数: {len(filtered_results_0001)}")
    print(f"  排除的图片数: {len(exclude_score0001)}")
    
    print(f"\n转换为COCO格式...")
    coco_dt_0001 = results_to_coco_format(filtered_results_0001, img_id_mapping_0001)
    print(f"  检测结果数: {len(coco_dt_0001)}")
    
    print(f"\n计算mAP...")
    map_results_0001 = calculate_map(filtered_coco_0001, coco_dt_0001)
    
    print(f"\n结果:")
    print(f"  mAP:    {map_results_0001['mAP']:.4f}")
    print(f"  mAP50:  {map_results_0001['mAP_50']:.4f}")
    print(f"  mAP75:  {map_results_0001['mAP_75']:.4f}")
    
    # 4. 保存结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("排除特定图片后的mAP计算结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【score_thr=0】\n")
        f.write("-" * 60 + "\n")
        f.write(f"排除的图片类型:\n")
        f.write(f"  - 漏检图片: {len(excluded_imgs['score_thr_0']['fn_images'])} 张\n")
        f.write(f"  - 高置信度FP图片 (score>0.5): {len(excluded_imgs['score_thr_0']['fp_high_conf_images'])} 张\n")
        f.write(f"  - 合并去重后: {len(exclude_score0)} 张\n\n")
        
        f.write(f"原始数据:\n")
        f.write(f"  - 总图片数: {len(coco_data['images'])}\n")
        f.write(f"  - 总标注数: {len(coco_data['annotations'])}\n\n")
        
        f.write(f"过滤后数据:\n")
        f.write(f"  - 剩余图片数: {len(filtered_coco_0['images'])}\n")
        f.write(f"  - 剩余标注数: {len(filtered_coco_0['annotations'])}\n\n")
        
        f.write(f"mAP 指标:\n")
        f.write(f"  - mAP:    {map_results_0['mAP']:.4f}\n")
        f.write(f"  - mAP50:  {map_results_0['mAP_50']:.4f}\n")
        f.write(f"  - mAP75:  {map_results_0['mAP_75']:.4f}\n\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("【score_thr=0.001】\n")
        f.write("-" * 60 + "\n")
        f.write(f"排除的图片类型:\n")
        f.write(f"  - 漏检图片: {len(excluded_imgs['score_thr_0001']['fn_images'])} 张\n")
        f.write(f"  - 高置信度FP图片 (score>0.5): {len(excluded_imgs['score_thr_0001']['fp_high_conf_images'])} 张\n")
        f.write(f"  - 合并去重后: {len(exclude_score0001)} 张\n\n")
        
        f.write(f"原始数据:\n")
        f.write(f"  - 总图片数: {len(coco_data['images'])}\n")
        f.write(f"  - 总标注数: {len(coco_data['annotations'])}\n\n")
        
        f.write(f"过滤后数据:\n")
        f.write(f"  - 剩余图片数: {len(filtered_coco_0001['images'])}\n")
        f.write(f"  - 剩余标注数: {len(filtered_coco_0001['annotations'])}\n\n")
        
        f.write(f"mAP 指标:\n")
        f.write(f"  - mAP:    {map_results_0001['mAP']:.4f}\n")
        f.write(f"  - mAP50:  {map_results_0001['mAP_50']:.4f}\n")
        f.write(f"  - mAP75:  {map_results_0001['mAP_75']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("计算完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n结果已保存到: {args.output}")
    print("\n完成!")


if __name__ == '__main__':
    main()
