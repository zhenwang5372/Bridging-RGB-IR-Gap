#!/usr/bin/env python
"""
生成 mAP50 分析报告
输出到 log 文件，包含问答形式的解释和详细数据
"""

import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def analyze_results(results, img_to_anns, img_id_to_name, score_thr=None, iou_thr=0.5):
    """分析所有结果，返回统计数据和详细信息"""
    stats = {
        'total_gt': 0,
        'total_pred': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'fp_images': [],  # (img_id, img_name, fp_count, max_fp_score)
        'fn_images': [],  # (img_id, img_name, fn_count)
        'high_conf_fps': [],  # (img_id, img_name, score, bbox)
    }
    
    for idx, result in enumerate(results):
        img_id = result.get('img_id', idx) if isinstance(result, dict) else idx
        img_name = img_id_to_name.get(img_id, f'{img_id}')
        
        # 获取预测
        pred = result['pred_instances'] if isinstance(result, dict) else result.pred_instances
        if isinstance(pred, dict):
            bboxes = pred['bboxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
        else:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
        
        # 过滤
        if score_thr is not None:
            mask = scores >= score_thr
            bboxes = bboxes[mask]
            scores = scores[mask]
        
        # 获取 GT
        anns = img_to_anns.get(img_id, [])
        gt_boxes = [coco_to_xyxy(ann['bbox']) for ann in anns]
        
        num_gt = len(gt_boxes)
        num_pred = len(bboxes)
        stats['total_gt'] += num_gt
        stats['total_pred'] += num_pred
        
        if num_gt == 0:
            stats['fp'] += num_pred
            if num_pred > 0:
                max_score = float(scores.max())
                stats['fp_images'].append((img_id, img_name, num_pred, max_score))
                for i, (score, bbox) in enumerate(zip(scores, bboxes)):
                    if score >= 0.5:  # 高置信度误检
                        stats['high_conf_fps'].append((img_id, img_name, float(score), bbox.tolist()))
            continue
            
        if num_pred == 0:
            stats['fn'] += num_gt
            stats['fn_images'].append((img_id, img_name, num_gt))
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
        
        # 统计 FN 和 FP
        fn_count = sum(1 for m in gt_matched if not m)
        fp_count = sum(1 for m in pred_matched if not m)
        
        stats['fn'] += fn_count
        stats['fp'] += fp_count
        
        if fn_count > 0:
            stats['fn_images'].append((img_id, img_name, fn_count))
        
        if fp_count > 0:
            fp_scores = [scores[i] for i, m in enumerate(pred_matched) if not m]
            max_fp_score = max(fp_scores) if fp_scores else 0
            stats['fp_images'].append((img_id, img_name, fp_count, float(max_fp_score)))
            
            # 记录高置信度误检
            for i, (m, score, bbox) in enumerate(zip(pred_matched, scores, bboxes)):
                if not m and score >= 0.5:
                    stats['high_conf_fps'].append((img_id, img_name, float(score), bbox.tolist()))
    
    return stats


def main():
    # 路径配置
    pkl_path = 'work_dirs/analysis_results.pkl'
    ann_path = 'data/LLVIP/coco_annotations/test.json'
    output_path = 'tools/analysis/map50_analysis_report.log'
    
    # 加载数据
    print("加载数据...")
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
    
    # 分析两种情况
    print("分析中...")
    stats_no_filter = analyze_results(results, img_to_anns, img_id_to_name, score_thr=None)
    stats_filtered = analyze_results(results, img_to_anns, img_id_to_name, score_thr=0.3)
    
    # 生成报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                    mAP50 分析报告\n")
        f.write(f"                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # ===== 问答1 =====
        f.write("=" * 80 + "\n")
        f.write("Q1: TP、FP、FN 在两种情况下分别是多少？\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【情况1】mAP 验证时使用（无 score_thr 过滤，只有 NMS）\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total GT:          {stats_no_filter['total_gt']}\n")
        f.write(f"  Total Predictions: {stats_no_filter['total_pred']}\n")
        f.write(f"  TP: {stats_no_filter['tp']}\n")
        f.write(f"  FP: {stats_no_filter['fp']}\n")
        f.write(f"  FN: {stats_no_filter['fn']}\n")
        tp, fp, fn = stats_no_filter['tp'], stats_no_filter['fp'], stats_no_filter['fn']
        f.write(f"  Precision: {tp/(tp+fp):.4f}\n" if (tp+fp) > 0 else "  Precision: N/A\n")
        f.write(f"  Recall:    {tp/(tp+fn):.4f}\n\n" if (tp+fn) > 0 else "  Recall: N/A\n\n")
        
        f.write("【情况2】图片可视化时使用（score_thr=0.3）\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total GT:          {stats_filtered['total_gt']}\n")
        f.write(f"  Total Predictions: {stats_filtered['total_pred']}\n")
        f.write(f"  TP: {stats_filtered['tp']}\n")
        f.write(f"  FP: {stats_filtered['fp']}\n")
        f.write(f"  FN: {stats_filtered['fn']}\n")
        tp, fp, fn = stats_filtered['tp'], stats_filtered['fp'], stats_filtered['fn']
        f.write(f"  Precision: {tp/(tp+fp):.4f}\n" if (tp+fp) > 0 else "  Precision: N/A\n")
        f.write(f"  Recall:    {tp/(tp+fn):.4f}\n\n" if (tp+fn) > 0 else "  Recall: N/A\n\n")
        
        # ===== 问答2 =====
        f.write("\n" + "=" * 80 + "\n")
        f.write("Q2: 为什么两种情况下 TP/FP/FN 有差别？\n")
        f.write("=" * 80 + "\n\n")
        
        pred_diff = stats_no_filter['total_pred'] - stats_filtered['total_pred']
        fp_diff = stats_no_filter['fp'] - stats_filtered['fp']
        fn_diff = stats_filtered['fn'] - stats_no_filter['fn']
        
        f.write(f"  过滤掉的低分预测数: {pred_diff}\n")
        f.write(f"  过滤掉的 FP 数:     {fp_diff}\n")
        f.write(f"  增加的 FN 数:       {fn_diff}\n\n")
        
        f.write("  解释：\n")
        f.write("  - score_thr=0.3 过滤掉了大量低置信度预测\n")
        f.write("  - 这些低分预测大部分是误检（FP），所以 FP 大幅减少\n")
        f.write("  - 少部分是正确检测但置信度低，被过滤后变成漏检（FN）\n\n")
        
        # ===== 问答3 =====
        f.write("\n" + "=" * 80 + "\n")
        f.write("Q3: 验证时用的是哪个？\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("  答：mAP 计算使用【情况1】（无 score_thr 过滤）\n\n")
        f.write("  原因：\n")
        f.write("  - mAP 需要完整的 PR 曲线\n")
        f.write("  - 按置信度从高到低排序，逐个累积计算 Precision/Recall\n")
        f.write("  - 高分检测在前，此时 Precision 高\n")
        f.write("  - 低分检测在后，PR 曲线下面积受影响小\n")
        f.write("  - 最终 mAP50 = PR 曲线下面积 ≈ 96%\n\n")
        
        # ===== 问答4 =====
        f.write("\n" + "=" * 80 + "\n")
        f.write("Q4: 为什么低分 FP 对 mAP 曲线影响小？\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("  因为 mAP 计算是按置信度排序后逐个累积的：\n\n")
        f.write("  检测按置信度从高到低排序：\n")
        f.write("  检测1 (conf=0.99) → TP → P=1/1=100%, R=1%\n")
        f.write("  检测2 (conf=0.98) → TP → P=2/2=100%, R=2%\n")
        f.write("  ...\n")
        f.write("  检测50 (conf=0.50) → TP → P=50/50=100%, R=50%\n")
        f.write("  检测51 (conf=0.49) → FP → P=50/51=98%, R=50%\n")
        f.write("  ...\n")
        f.write("  检测1000 (conf=0.01) → FP → P下降但R不变\n\n")
        f.write("  关键点：\n")
        f.write("  - 低分 FP 出现在排序末尾\n")
        f.write("  - 此时 Recall 已经接近最大值\n")
        f.write("  - mAP = PR曲线下面积，高 Recall 区域的宽度小\n")
        f.write("  - 低分 FP 只影响曲线尾部，对总面积贡献微小\n\n")
        
        # ===== 问答5 =====
        f.write("\n" + "=" * 80 + "\n")
        f.write("Q5: NMS 和 score_thr 的区别是什么？\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("  ┌─────────────┬────────────────────┬──────────────┐\n")
        f.write("  │   过滤方式  │        作用        │   何时生效   │\n")
        f.write("  ├─────────────┼────────────────────┼──────────────┤\n")
        f.write("  │     NMS     │ 去除重叠的冗余框   │  ✅ 始终生效 │\n")
        f.write("  │  score_thr  │ 过滤低置信度框     │  取决于配置  │\n")
        f.write("  └─────────────┴────────────────────┴──────────────┘\n\n")
        
        f.write("  你的模型配置：\n")
        f.write("  test_cfg = dict(\n")
        f.write("      score_thr=0.001,   # 几乎不过滤！\n")
        f.write("      nms=dict(type='nms', iou_threshold=0.7),  # NMS 生效\n")
        f.write("      max_per_img=300\n")
        f.write("  )\n\n")
        f.write("  131,327 个预测 = NMS 后 + score_thr=0.001 过滤后的结果\n")
        f.write("  - NMS 去除了重叠框\n")
        f.write("  - score_thr=0.001 保留了几乎所有框（mAP 计算需要）\n\n")
        
        # ===== 高置信度误检 =====
        f.write("\n" + "=" * 80 + "\n")
        f.write("高置信度误检 (FP, score >= 0.5)\n")
        f.write("=" * 80 + "\n\n")
        
        high_conf_fps = stats_filtered['high_conf_fps']
        high_conf_fps.sort(key=lambda x: x[2], reverse=True)  # 按置信度排序
        
        f.write(f"  总数: {len(high_conf_fps)} 个高置信度误检\n\n")
        
        # 按图片分组
        img_fp_map = {}
        for img_id, img_name, score, bbox in high_conf_fps:
            if img_name not in img_fp_map:
                img_fp_map[img_name] = []
            img_fp_map[img_name].append((score, bbox))
        
        f.write("  按图片汇总 (共 {} 张图片有高置信度误检):\n".format(len(img_fp_map)))
        f.write("-" * 60 + "\n")
        
        # 按最高误检分数排序
        sorted_imgs = sorted(img_fp_map.items(), key=lambda x: max(s for s, _ in x[1]), reverse=True)
        
        for img_name, fps in sorted_imgs[:50]:  # 显示前50张
            max_score = max(s for s, _ in fps)
            f.write(f"  {img_name}: {len(fps)} 个 FP, 最高分={max_score:.4f}\n")
        
        if len(sorted_imgs) > 50:
            f.write(f"  ... 还有 {len(sorted_imgs) - 50} 张图片\n")
        
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("  Top 30 高置信度误检详情:\n")
        f.write("-" * 60 + "\n")
        
        for i, (img_id, img_name, score, bbox) in enumerate(high_conf_fps[:30]):
            f.write(f"  {i+1:2d}. {img_name} | score={score:.4f} | bbox={bbox}\n")
        
        # ===== 漏检统计 =====
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("漏检 (FN) 统计 (score_thr=0.3)\n")
        f.write("=" * 80 + "\n\n")
        
        fn_images = stats_filtered['fn_images']
        fn_images.sort(key=lambda x: x[2], reverse=True)  # 按漏检数排序
        
        f.write(f"  有漏检的图片数: {len(fn_images)}\n")
        f.write(f"  总漏检数: {stats_filtered['fn']}\n\n")
        
        f.write("  漏检最多的 Top 30 图片:\n")
        f.write("-" * 60 + "\n")
        for i, (img_id, img_name, fn_count) in enumerate(fn_images[:30]):
            f.write(f"  {i+1:2d}. {img_name}: {fn_count} 个漏检\n")
        
        # ===== 图片ID列表 =====
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("便于复制的图片 ID 列表\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【高置信度误检图片】(score >= 0.5)\n")
        f.write("-" * 60 + "\n")
        high_fp_names = [name.replace('.jpg', '') for name in img_fp_map.keys()]
        f.write(", ".join(high_fp_names[:100]) + "\n")
        
        f.write("\n【漏检图片】\n")
        f.write("-" * 60 + "\n")
        fn_names = [item[1].replace('.jpg', '') for item in fn_images[:100]]
        f.write(", ".join(fn_names) + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"报告已保存到: {output_path}")


if __name__ == '__main__':
    main()
