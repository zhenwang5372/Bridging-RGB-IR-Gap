#!/usr/bin/env python
"""
文本更新版本的 mAP50 分析脚本
对比 score_thr=0 和 score_thr=0.001 的结果
生成分析报告和可视化
"""

import pickle
import json
import numpy as np
import cv2
import os
import argparse
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='分析文本更新版本的检测结果')
    parser.add_argument('--pkl-score0', type=str, required=True,
                        help='score_thr=0 的 pkl 文件路径')
    parser.add_argument('--pkl-score0001', type=str, required=True,
                        help='score_thr=0.001 的 pkl 文件路径')
    parser.add_argument('--ann-file', type=str, 
                        default='data/LLVIP/coco_annotations/test.json',
                        help='COCO 标注文件路径')
    parser.add_argument('--img-dir', type=str,
                        default='data/LLVIP/visible/test',
                        help='RGB 图片目录')
    parser.add_argument('--ir-dir', type=str,
                        default='data/LLVIP/infrared/test',
                        help='IR 图片目录')
    parser.add_argument('--output-log', type=str,
                        default='tools/analysis/map50_analysis_report_text_update.log',
                        help='输出报告路径')
    parser.add_argument('--fp-dir', type=str,
                        default='tools/analysis/FP',
                        help='FP 可视化输出目录')
    parser.add_argument('--fn-dir', type=str,
                        default='tools/analysis/FN',
                        help='FN 可视化输出目录')
    parser.add_argument('--fp-vis-thr', type=float, default=0.3,
                        help='FP 可视化的置信度阈值')
    return parser.parse_args()


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


def load_data(pkl_path, ann_path):
    """加载 pkl 结果和标注"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # 构建映射
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    return results, coco_data, img_to_anns, img_id_to_name


def analyze_results(results, img_to_anns, img_id_to_name, iou_thr=0.5):
    """
    分析检测结果
    返回：统计数据、FP 详情、FN 详情
    """
    stats = {
        'total_gt': 0,
        'total_pred': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
    }
    
    # FP 详情: [(img_name, score, bbox), ...]
    fp_details = []
    # FN 详情: [(img_name, gt_bbox), ...]
    fn_details = []
    
    for idx, result in enumerate(results):
        img_id = result.get('img_id', idx) if isinstance(result, dict) else idx
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        # 获取预测
        pred = result['pred_instances'] if isinstance(result, dict) else result.pred_instances
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
        stats['total_gt'] += num_gt
        stats['total_pred'] += num_pred
        
        if num_gt == 0:
            # 没有 GT，所有预测都是 FP
            stats['fp'] += num_pred
            for score, bbox in zip(scores, bboxes):
                fp_details.append((img_name, float(score), bbox.tolist()))
            continue
        
        if num_pred == 0:
            # 没有预测，所有 GT 都是 FN
            stats['fn'] += num_gt
            for gt in gt_boxes:
                fn_details.append((img_name, gt))
            continue
        
        # 计算 IoU 矩阵
        iou_matrix = np.zeros((num_gt, num_pred))
        for i, gt in enumerate(gt_boxes):
            for j, pred_box in enumerate(bboxes):
                iou_matrix[i, j] = compute_iou(gt, pred_box.tolist())
        
        # 贪婪匹配
        gt_matched = [False] * num_gt
        pred_matched = [False] * num_pred
        
        indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
        for gi, pi in zip(indices[0], indices[1]):
            if gt_matched[gi] or pred_matched[pi]:
                continue
            if iou_matrix[gi, pi] >= iou_thr:
                gt_matched[gi] = True
                pred_matched[pi] = True
                stats['tp'] += 1
        
        # 统计 FP 和 FN
        for pi, (matched, score, bbox) in enumerate(zip(pred_matched, scores, bboxes)):
            if not matched:
                stats['fp'] += 1
                fp_details.append((img_name, float(score), bbox.tolist()))
        
        for gi, (matched, gt) in enumerate(zip(gt_matched, gt_boxes)):
            if not matched:
                stats['fn'] += 1
                fn_details.append((img_name, gt))
    
    # 计算 Precision 和 Recall
    stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
    stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
    
    return stats, fp_details, fn_details


def draw_boxes_on_image(img, gt_boxes, pred_boxes, pred_scores, pred_matched, 
                        missed_gt_indices, fp_indices):
    """在图像上绘制框"""
    img = img.copy()
    
    # 绘制所有 GT（绿色）
    for i, gt in enumerate(gt_boxes):
        color = (0, 255, 0)  # 绿色
        thickness = 2
        if i in missed_gt_indices:
            color = (0, 0, 255)  # 红色表示漏检
            thickness = 3
        cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), color, thickness)
        label = "GT" if i not in missed_gt_indices else "MISSED"
        cv2.putText(img, label, (int(gt[0]), int(gt[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 绘制预测框
    for i, (bbox, score, matched) in enumerate(zip(pred_boxes, pred_scores, pred_matched)):
        if matched:
            color = (255, 255, 0)  # 青色表示 TP
            label = f"TP:{score:.2f}"
        else:
            color = (0, 165, 255)  # 橙色表示 FP
            label = f"FP:{score:.2f}"
        
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(img, label, (int(bbox[0]), int(bbox[3]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def visualize_fp_fn(results, img_to_anns, img_id_to_name, img_dir, ir_dir, 
                    fp_dir, fn_dir, fp_vis_thr=0.3):
    """可视化 FP 和 FN"""
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    
    fp_images = set()
    fn_images = set()
    
    for idx, result in enumerate(results):
        img_id = result.get('img_id', idx) if isinstance(result, dict) else idx
        img_name = img_id_to_name.get(img_id, f'{img_id}.jpg')
        
        # 获取预测
        pred = result['pred_instances'] if isinstance(result, dict) else result.pred_instances
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy() if hasattr(pred['bboxes'], 'cpu') else np.array(pred['bboxes'])
            scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else np.array(pred['scores'])
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
        
        # 获取 GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        # 计算匹配
        pred_matched = [False] * len(bboxes)
        gt_matched = [False] * len(gt_boxes)
        
        if len(gt_boxes) > 0 and len(bboxes) > 0:
            iou_matrix = np.zeros((len(gt_boxes), len(bboxes)))
            for i, gt in enumerate(gt_boxes):
                for j, pred_box in enumerate(bboxes):
                    iou_matrix[i, j] = compute_iou(gt, pred_box.tolist())
            
            indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
            for gi, pi in zip(indices[0], indices[1]):
                if gt_matched[gi] or pred_matched[pi]:
                    continue
                if iou_matrix[gi, pi] >= 0.5:
                    gt_matched[gi] = True
                    pred_matched[pi] = True
        
        # 检查是否有高分 FP
        has_high_fp = False
        for pi, (matched, score) in enumerate(zip(pred_matched, scores)):
            if not matched and score >= fp_vis_thr:
                has_high_fp = True
                break
        
        # 检查是否有漏检
        missed_gt_indices = [i for i, m in enumerate(gt_matched) if not m]
        has_fn = len(missed_gt_indices) > 0
        
        # 读取图像
        rgb_path = os.path.join(img_dir, img_name)
        ir_path = os.path.join(ir_dir, img_name)
        
        if not os.path.exists(rgb_path):
            continue
        
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path) if os.path.exists(ir_path) else None
        
        # 可视化 FP（只显示高分 FP）
        if has_high_fp:
            fp_images.add(img_name)
            
            # 过滤只显示高分预测
            high_score_mask = scores >= fp_vis_thr
            vis_bboxes = bboxes[high_score_mask]
            vis_scores = scores[high_score_mask]
            vis_matched = [pred_matched[i] for i in range(len(pred_matched)) if high_score_mask[i]]
            
            # 绘制 RGB
            fp_indices = [i for i, m in enumerate(vis_matched) if not m]
            vis_rgb = draw_boxes_on_image(rgb_img, gt_boxes, vis_bboxes, vis_scores, 
                                          vis_matched, [], fp_indices)
            cv2.imwrite(os.path.join(fp_dir, img_name), vis_rgb)
            
            # 绘制 IR
            if ir_img is not None:
                vis_ir = draw_boxes_on_image(ir_img, gt_boxes, vis_bboxes, vis_scores,
                                             vis_matched, [], fp_indices)
                ir_name = img_name.replace('.jpg', '_ir.jpg')
                cv2.imwrite(os.path.join(fp_dir, ir_name), vis_ir)
        
        # 可视化 FN
        if has_fn:
            fn_images.add(img_name)
            
            # 绘制所有预测和漏检的 GT
            vis_rgb = draw_boxes_on_image(rgb_img, gt_boxes, bboxes, scores,
                                          pred_matched, missed_gt_indices, [])
            cv2.imwrite(os.path.join(fn_dir, img_name), vis_rgb)
            
            if ir_img is not None:
                vis_ir = draw_boxes_on_image(ir_img, gt_boxes, bboxes, scores,
                                             pred_matched, missed_gt_indices, [])
                ir_name = img_name.replace('.jpg', '_ir.jpg')
                cv2.imwrite(os.path.join(fn_dir, ir_name), vis_ir)
    
    return fp_images, fn_images


def generate_report(stats_0, fp_details_0, fn_details_0,
                    stats_0001, fp_details_0001, fn_details_0001,
                    output_path, fp_images, fn_images):
    """生成分析报告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                    mAP50 分析报告 (文本更新版本)\n")
        f.write(f"                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【配置信息】\n")
        f.write("-" * 60 + "\n")
        f.write("  基础配置 score_thr: 0.001\n")
        f.write("  对比测试 score_thr: 0 (完全无过滤)\n")
        f.write("  IoU 阈值: 0.5\n\n")
        
        # ================== 整体统计 ==================
        f.write("=" * 80 + "\n")
        f.write("【整体统计】score_thr=0 (完全无过滤)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Total GT:          {stats_0['total_gt']}\n")
        f.write(f"  Total Predictions: {stats_0['total_pred']}\n")
        f.write(f"  TP:                {stats_0['tp']}\n")
        f.write(f"  FP:                {stats_0['fp']}\n")
        f.write(f"  FN:                {stats_0['fn']}\n")
        f.write(f"  Precision:         {stats_0['precision']:.4f}\n")
        f.write(f"  Recall:            {stats_0['recall']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("【整体统计】score_thr=0.001 (配置文件默认)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Total GT:          {stats_0001['total_gt']}\n")
        f.write(f"  Total Predictions: {stats_0001['total_pred']}\n")
        f.write(f"  TP:                {stats_0001['tp']}\n")
        f.write(f"  FP:                {stats_0001['fp']}\n")
        f.write(f"  FN:                {stats_0001['fn']}\n")
        f.write(f"  Precision:         {stats_0001['precision']:.4f}\n")
        f.write(f"  Recall:            {stats_0001['recall']:.4f}\n\n")
        
        # ================== 对比 ==================
        f.write("=" * 80 + "\n")
        f.write("【对比】score_thr=0 vs score_thr=0.001\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Predictions 差异: {stats_0['total_pred'] - stats_0001['total_pred']}\n")
        f.write(f"  TP 差异:          {stats_0['tp'] - stats_0001['tp']}\n")
        f.write(f"  FP 差异:          {stats_0['fp'] - stats_0001['fp']}\n")
        f.write(f"  FN 差异:          {stats_0['fn'] - stats_0001['fn']}\n\n")
        
        # ================== 高置信度误检 ==================
        def write_fp_section(f, fp_details, score_thr_name):
            f.write("=" * 80 + "\n")
            f.write(f"【高置信度误检】{score_thr_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # 按置信度区间分组
            bins = [
                (0.5, 1.0, "score >= 0.5", True),
                (0.4, 0.5, "0.4 <= score < 0.5", True),
                (0.3, 0.4, "0.3 <= score < 0.4", True),
                (0.2, 0.3, "0.2 <= score < 0.3", False),
                (0.1, 0.2, "0.1 <= score < 0.2", False),
                (0.0, 0.1, "0 <= score < 0.1", False),
            ]
            
            for low, high, label, show_detail in bins:
                fps_in_range = [(name, score, bbox) for name, score, bbox in fp_details 
                                if low <= score < high]
                
                # 按图片名排序
                fps_in_range.sort(key=lambda x: x[0])
                
                # 统计图片数
                img_names = set(name for name, _, _ in fps_in_range)
                
                f.write(f"【{label}】\n")
                f.write("-" * 60 + "\n")
                f.write(f"  FP 总数: {len(fps_in_range)}\n")
                f.write(f"  涉及图片数: {len(img_names)}\n")
                
                if show_detail and len(fps_in_range) > 0:
                    f.write("\n  详情:\n")
                    for name, score, bbox in fps_in_range:
                        bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                        f.write(f"    {name} | score={score:.4f} | bbox={bbox_str}\n")
                
                f.write("\n")
        
        write_fp_section(f, fp_details_0, "score_thr=0")
        write_fp_section(f, fp_details_0001, "score_thr=0.001")
        
        # ================== 漏检 ==================
        def write_fn_section(f, fn_details, score_thr_name):
            f.write("=" * 80 + "\n")
            f.write(f"【漏检】{score_thr_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # 按图片名排序
            fn_details_sorted = sorted(fn_details, key=lambda x: x[0])
            
            # 统计图片数
            img_names = set(name for name, _ in fn_details_sorted)
            
            f.write(f"  漏检总数: {len(fn_details_sorted)}\n")
            f.write(f"  涉及图片数: {len(img_names)}\n\n")
            
            if len(fn_details_sorted) > 0:
                f.write("  详情:\n")
                for name, gt in fn_details_sorted:
                    gt_str = f"[{gt[0]:.1f}, {gt[1]:.1f}, {gt[2]:.1f}, {gt[3]:.1f}]"
                    f.write(f"    {name} | GT bbox={gt_str}\n")
            
            f.write("\n")
        
        write_fn_section(f, fn_details_0, "score_thr=0")
        write_fn_section(f, fn_details_0001, "score_thr=0.001")
        
        # ================== 可视化统计 ==================
        f.write("=" * 80 + "\n")
        f.write("【可视化统计】\n")
        f.write("=" * 80 + "\n")
        f.write(f"  FP 可视化图片数 (score >= 0.3): {len(fp_images)}\n")
        f.write(f"  FN 可视化图片数: {len(fn_images)}\n")
        f.write(f"  FP 保存目录: tools/analysis/FP/\n")
        f.write(f"  FN 保存目录: tools/analysis/FN/\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"报告已保存到: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("文本更新版本 mAP50 分析")
    print("=" * 60)
    
    # 加载 score_thr=0 的结果
    print(f"\n加载 score_thr=0 结果: {args.pkl_score0}")
    results_0, coco_data, img_to_anns, img_id_to_name = load_data(args.pkl_score0, args.ann_file)
    
    # 加载 score_thr=0.001 的结果
    print(f"加载 score_thr=0.001 结果: {args.pkl_score0001}")
    results_0001, _, _, _ = load_data(args.pkl_score0001, args.ann_file)
    
    # 分析 score_thr=0
    print("\n分析 score_thr=0 结果...")
    stats_0, fp_details_0, fn_details_0 = analyze_results(results_0, img_to_anns, img_id_to_name)
    print(f"  TP: {stats_0['tp']}, FP: {stats_0['fp']}, FN: {stats_0['fn']}")
    
    # 分析 score_thr=0.001
    print("分析 score_thr=0.001 结果...")
    stats_0001, fp_details_0001, fn_details_0001 = analyze_results(results_0001, img_to_anns, img_id_to_name)
    print(f"  TP: {stats_0001['tp']}, FP: {stats_0001['fp']}, FN: {stats_0001['fn']}")
    
    # 可视化（使用 score_thr=0 的结果）
    print(f"\n生成可视化 (FP阈值: {args.fp_vis_thr})...")
    fp_images, fn_images = visualize_fp_fn(
        results_0, img_to_anns, img_id_to_name,
        args.img_dir, args.ir_dir,
        args.fp_dir, args.fn_dir,
        args.fp_vis_thr
    )
    print(f"  FP 可视化图片数: {len(fp_images)}")
    print(f"  FN 可视化图片数: {len(fn_images)}")
    
    # 生成报告
    print("\n生成报告...")
    generate_report(
        stats_0, fp_details_0, fn_details_0,
        stats_0001, fp_details_0001, fn_details_0001,
        args.output_log, fp_images, fn_images
    )
    
    print("\n完成!")


if __name__ == '__main__':
    main()
