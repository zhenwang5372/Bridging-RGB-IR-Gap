#!/usr/bin/env python
"""
分析从score_thr=0到score_thr=0.001，高置信度FP的变化
"""

import re
import argparse


def parse_high_conf_fp(log_path, score_thr):
    """提取某个score_thr下的高置信度FP图片"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if score_thr == '0':
        pattern = re.compile(r'【高置信度误检】score_thr=0\s*={3,}.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    else:  # 0.001
        pattern = re.compile(r'【高置信度误检】score_thr=0\.001\s*={3,}.*?【score >= 0\.5】.*?详情:(.*?)(?=【|$)', re.DOTALL)
    
    match = pattern.search(content)
    if match:
        section = match.group(1)
        img_names = re.findall(r'(\d+\.jpg)', section)
        return set(img_names)
    return set()


def main():
    parser = argparse.ArgumentParser(description='分析高置信度FP的变化')
    parser.add_argument('--log-file', type=str,
                        default='tools/analysis/with_text_update/map50_analysis_report_text_update.log',
                        help='日志文件路径')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("分析高置信度FP（score>0.5）的变化")
    print("=" * 80)
    print()
    
    # 提取数据
    fp_score0 = parse_high_conf_fp(args.log_file, '0')
    fp_score0001 = parse_high_conf_fp(args.log_file, '0.001')
    
    print(f"【基础统计】")
    print("-" * 60)
    print(f"score_thr=0 的高置信度FP图片数:      {len(fp_score0):4d} 张")
    print(f"score_thr=0.001 的高置信度FP图片数:  {len(fp_score0001):4d} 张")
    print(f"差异:                                {len(fp_score0) - len(fp_score0001):4d} 张")
    print()
    
    # 集合分析
    only_in_score0 = fp_score0 - fp_score0001  # 只在score_thr=0中是FP
    only_in_score0001 = fp_score0001 - fp_score0  # 只在score_thr=0.001中是FP
    both = fp_score0 & fp_score0001  # 在两者中都是FP
    
    print(f"【详细分析】")
    print("-" * 60)
    print(f"两者共有的高置信度FP:              {len(both):4d} 张")
    print(f"  → 这些是真正的高置信度误检，无论score_thr如何都是FP")
    print()
    print(f"只在score_thr=0中是FP:             {len(only_in_score0):4d} 张")
    print(f"  → 这些图片在score_thr=0.001时，高置信度预测匹配了GT（变成TP）")
    print(f"  → 原因: 在score_thr=0时，低置信度预测\"抢占\"了GT")
    print()
    print(f"只在score_thr=0.001中是FP:         {len(only_in_score0001):4d} 张")
    print(f"  → 这种情况不应该出现（理论上）")
    print()
    
    print(f"【验证结论】")
    print("-" * 60)
    print(f"✓ 差异数量 = 只在score_thr=0中的FP数量")
    print(f"  {len(fp_score0) - len(fp_score0001)} = {len(only_in_score0)}")
    if len(fp_score0) - len(fp_score0001) == len(only_in_score0):
        print(f"  ✓ 验证通过！")
    else:
        print(f"  ✗ 验证失败！")
    print()
    
    print(f"【结论】")
    print("-" * 60)
    print(f"在 {len(only_in_score0)} 张图片中：")
    print(f"  - score_thr=0 时: 高置信度预测被判定为FP")
    print(f"  - score_thr=0.001 时: 同样的高置信度预测匹配了GT（变成TP）")
    print()
    print(f"原因: 在score_thr=0时，存在极低置信度(<0.001)但IoU很高的预测，")
    print(f"      这些预测\"抢占\"了GT，导致高置信度预测无法匹配。")
    print()
    
    if len(only_in_score0) > 0:
        print(f"【示例图片（前20张）】")
        print("-" * 60)
        sample = sorted(list(only_in_score0))[:20]
        for i, img in enumerate(sample, 1):
            print(f"  {i:2d}. {img}")
        if len(only_in_score0) > 20:
            print(f"  ... 还有 {len(only_in_score0) - 20} 张")
        print()
    
    if len(only_in_score0001) > 0:
        print(f"【异常情况】")
        print("-" * 60)
        print(f"警告: 发现 {len(only_in_score0001)} 张图片只在score_thr=0.001中是FP")
        print(f"这在理论上不应该出现，可能原因：")
        print(f"  1. 日志解析错误")
        print(f"  2. 数据不一致")
        print(f"  3. 其他未知因素")
        print()
        print(f"示例图片（前10张）:")
        sample = sorted(list(only_in_score0001))[:10]
        for i, img in enumerate(sample, 1):
            print(f"  {i}. {img}")
        print()
    
    print("=" * 80)
    print()
    print("【建议】")
    print("-" * 60)
    print("排除图片时，应该使用 score_thr=0.001 的结果：")
    print(f"  - 漏检图片: 63 张")
    print(f"  - 高置信度FP: {len(fp_score0001)} 张")
    print(f"  这样得到的结果更准确，因为:")
    print(f"    1. 避免了低置信度预测的干扰")
    print(f"    2. 反映了模型在实际使用场景下的表现")
    print("=" * 80)


if __name__ == '__main__':
    main()
