#!/usr/bin/env python
"""
快速统计需要排除的图片数量
"""

import re
import argparse


def parse_log_file(log_path):
    """从日志文件中解析出需要排除的图片列表"""
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
    fn_pattern_0 = re.compile(r'【漏检】score_thr=0\s*={3,}.*?详情:(.*?)(?=【|=====|$)', re.DOTALL)
    fn_match_0 = fn_pattern_0.search(content)
    if fn_match_0:
        fn_section = fn_match_0.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fn_section)
        result['score_thr_0']['fn_images'] = set(img_names)
    
    # 解析 score_thr=0.001 的漏检图片
    fn_pattern_0001 = re.compile(r'【漏检】score_thr=0\.001\s*={3,}.*?详情:(.*?)(?=【|=====|$)', re.DOTALL)
    fn_match_0001 = fn_pattern_0001.search(content)
    if fn_match_0001:
        fn_section = fn_match_0001.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fn_section)
        result['score_thr_0001']['fn_images'] = set(img_names)
    
    # 解析 score_thr=0 的高置信度FP图片 (score >= 0.5)
    fp_pattern_0 = re.compile(r'【高置信度误检】score_thr=0\s*={3,}.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fp_match_0 = fp_pattern_0.search(content)
    if fp_match_0:
        fp_section = fp_match_0.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fp_section)
        result['score_thr_0']['fp_high_conf_images'] = set(img_names)
    
    # 解析 score_thr=0.001 的高置信度FP图片 (score >= 0.5)
    fp_pattern_0001 = re.compile(r'【高置信度误检】score_thr=0\.001\s*={3,}.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    fp_match_0001 = fp_pattern_0001.search(content)
    if fp_match_0001:
        fp_section = fp_match_0001.group(1)
        img_names = re.findall(r'(\d+\.jpg)', fp_section)
        result['score_thr_0001']['fp_high_conf_images'] = set(img_names)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='统计需要排除的图片数量')
    parser.add_argument('--log-file', type=str, 
                        default='tools/analysis/with_text_update/map50_analysis_report_text_update.log',
                        help='日志文件路径')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("统计需要排除的图片数量")
    print("=" * 80)
    print(f"\n日志文件: {args.log_file}\n")
    
    # 解析日志文件
    excluded_imgs = parse_log_file(args.log_file)
    
    # 计算 score_thr=0
    fn_count_0 = len(excluded_imgs['score_thr_0']['fn_images'])
    fp_count_0 = len(excluded_imgs['score_thr_0']['fp_high_conf_images'])
    exclude_score0 = excluded_imgs['score_thr_0']['fn_images'] | excluded_imgs['score_thr_0']['fp_high_conf_images']
    total_exclude_0 = len(exclude_score0)
    
    # 计算 score_thr=0.001
    fn_count_0001 = len(excluded_imgs['score_thr_0001']['fn_images'])
    fp_count_0001 = len(excluded_imgs['score_thr_0001']['fp_high_conf_images'])
    exclude_score0001 = excluded_imgs['score_thr_0001']['fn_images'] | excluded_imgs['score_thr_0001']['fp_high_conf_images']
    total_exclude_0001 = len(exclude_score0001)
    
    # 输出结果
    print("【score_thr=0】")
    print("-" * 60)
    print(f"  漏检（FN）图片数:                    {fn_count_0:4d} 张")
    print(f"  高置信度FP图片数 (score > 0.5):      {fp_count_0:4d} 张")
    print(f"  两者可能有重复，合并去重后总数:      {total_exclude_0:4d} 张")
    print()
    
    # 显示部分图片名称（前10个）
    if fn_count_0 > 0:
        fn_list = sorted(list(excluded_imgs['score_thr_0']['fn_images']))
        print(f"  漏检图片示例（前10张）:")
        for i, img in enumerate(fn_list[:10], 1):
            print(f"    {i}. {img}")
        if fn_count_0 > 10:
            print(f"    ... 还有 {fn_count_0 - 10} 张")
        print()
    
    if fp_count_0 > 0:
        fp_list = sorted(list(excluded_imgs['score_thr_0']['fp_high_conf_images']))
        print(f"  高置信度FP图片示例（前10张）:")
        for i, img in enumerate(fp_list[:10], 1):
            print(f"    {i}. {img}")
        if fp_count_0 > 10:
            print(f"    ... 还有 {fp_count_0 - 10} 张")
        print()
    
    print("\n" + "=" * 80 + "\n")
    
    print("【score_thr=0.001】")
    print("-" * 60)
    print(f"  漏检（FN）图片数:                    {fn_count_0001:4d} 张")
    print(f"  高置信度FP图片数 (score > 0.5):      {fp_count_0001:4d} 张")
    print(f"  两者可能有重复，合并去重后总数:      {total_exclude_0001:4d} 张")
    print()
    
    # 显示部分图片名称（前10个）
    if fn_count_0001 > 0:
        fn_list = sorted(list(excluded_imgs['score_thr_0001']['fn_images']))
        print(f"  漏检图片示例（前10张）:")
        for i, img in enumerate(fn_list[:10], 1):
            print(f"    {i}. {img}")
        if fn_count_0001 > 10:
            print(f"    ... 还有 {fn_count_0001 - 10} 张")
        print()
    
    if fp_count_0001 > 0:
        fp_list = sorted(list(excluded_imgs['score_thr_0001']['fp_high_conf_images']))
        print(f"  高置信度FP图片示例（前10张）:")
        for i, img in enumerate(fp_list[:10], 1):
            print(f"    {i}. {img}")
        if fp_count_0001 > 10:
            print(f"    ... 还有 {fp_count_0001 - 10} 张")
        print()
    
    print("=" * 80)
    print("\n说明:")
    print("  - 漏检（FN）: 模型未检测到的GT目标所在的图片")
    print("  - 高置信度FP: 置信度 > 0.5 的误检所在的图片")
    print("  - 两者可能有重复: 同一张图片可能既有漏检又有高置信度FP")
    print("  - 合并去重后的数量是实际需要排除的图片总数")
    print()
    print("下一步:")
    print("  如需计算排除这些图片后的mAP，请运行:")
    print("  python tools/analysis/with_text_update/calculate_map_after_filtering.py \\")
    print("      --log-file <日志文件> \\")
    print("      --pkl-score0 <pkl文件_score0> \\")
    print("      --pkl-score0001 <pkl文件_score0001> \\")
    print("      --ann-file <标注文件>")
    print()


if __name__ == '__main__':
    main()
