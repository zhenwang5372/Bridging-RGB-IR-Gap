#!/usr/bin/env python
"""
Merge grid search results from multiple GPUs into a single summary.
"""
import argparse
import os
import os.path as osp
import re
from datetime import datetime
from glob import glob


def parse_log_file(log_path):
    """Parse a single GPU log file and extract results."""
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all combination blocks
    # Pattern to match: Combination: A=X.XX, B=Y.YY followed by table with mAP values
    pattern = r"Combination: A=([0-9.]+), B=([0-9.]+).*?\| person\s+\| ([0-9.nan]+)\s+\| ([0-9.nan]+)\s+\| ([0-9.nan]+)\s+\| ([0-9.nan]+)\s+\| ([0-9.nan]+)\s+\| ([0-9.nan]+)\s+\|"
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        A, B, mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l = match
        
        def parse_float(s):
            try:
                return float(s)
            except:
                return float('nan')
        
        results.append({
            'A': float(A),
            'B': float(B),
            'mAP': parse_float(mAP),
            'mAP_50': parse_float(mAP_50),
            'mAP_75': parse_float(mAP_75),
            'mAP_s': parse_float(mAP_s),
            'mAP_m': parse_float(mAP_m),
            'mAP_l': parse_float(mAP_l)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Merge grid search results')
    parser.add_argument('--work-dir', default='work_dirs/grid_search_results',
                        help='Directory containing GPU log files')
    args = parser.parse_args()
    
    # Find all GPU log files
    log_pattern = osp.join(args.work_dir, 'grid_search_gpu*.log')
    log_files = sorted(glob(log_pattern))
    
    if not log_files:
        print(f"No log files found matching {log_pattern}")
        return
    
    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {f}")
    
    # Collect all results
    all_results = []
    for log_file in log_files:
        results = parse_log_file(log_file)
        print(f"  Parsed {len(results)} results from {osp.basename(log_file)}")
        all_results.extend(results)
    
    print(f"\nTotal results: {len(all_results)}")
    
    # Sort by mAP_50 descending
    valid_results = [r for r in all_results if r['mAP_50'] == r['mAP_50']]  # filter nan
    valid_results.sort(key=lambda x: x['mAP_50'], reverse=True)
    
    # Generate merged summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = osp.join(args.work_dir, f'MERGED_SUMMARY_{timestamp}.log')
    
    lines = []
    lines.append("=" * 80)
    lines.append("  MERGED GRID SEARCH RESULTS - ALL GPUs")
    lines.append("=" * 80)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total combinations tested: {len(all_results)}")
    lines.append(f"  Valid results: {len(valid_results)}")
    lines.append("=" * 80)
    lines.append("")
    
    # Top 20 results
    lines.append("  TOP 20 RESULTS (sorted by mAP_50)")
    lines.append("-" * 80)
    lines.append("| Rank |  A   |  B   |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |")
    lines.append("|------|------|------|-------|--------|--------|-------|-------|-------|")
    
    def fmt(v):
        if v != v:  # nan
            return " nan "
        return f"{v:.3f}"
    
    for rank, r in enumerate(valid_results[:20], 1):
        lines.append(f"| {rank:4d} | {r['A']:.2f} | {r['B']:.2f} | {fmt(r['mAP'])} | {fmt(r['mAP_50'])} | {fmt(r['mAP_75'])} | {fmt(r['mAP_s'])} | {fmt(r['mAP_m'])} | {fmt(r['mAP_l'])} |")
    
    lines.append("")
    lines.append("=" * 80)
    
    if valid_results:
        best = valid_results[0]
        lines.append("  BEST COMBINATION:")
        lines.append(f"    A = {best['A']:.2f}")
        lines.append(f"    B = {best['B']:.2f}")
        lines.append(f"    mAP_50 = {best['mAP_50']:.4f}")
        lines.append(f"    mAP = {best['mAP']:.4f}")
        lines.append("")
        lines.append(f"  Formula: output = {best['A']:.2f} * x_rgb + {best['B']:.2f} * self.gamma * fused")
    
    lines.append("=" * 80)
    lines.append("")
    
    # Full results table
    lines.append("")
    lines.append("  FULL RESULTS (all combinations, sorted by mAP_50)")
    lines.append("-" * 80)
    lines.append("| Rank |  A   |  B   |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |")
    lines.append("|------|------|------|-------|--------|--------|-------|-------|-------|")
    
    for rank, r in enumerate(valid_results, 1):
        lines.append(f"| {rank:4d} | {r['A']:.2f} | {r['B']:.2f} | {fmt(r['mAP'])} | {fmt(r['mAP_50'])} | {fmt(r['mAP_75'])} | {fmt(r['mAP_s'])} | {fmt(r['mAP_m'])} | {fmt(r['mAP_l'])} |")
    
    lines.append("")
    lines.append("=" * 80)
    
    summary_text = "\n".join(lines)
    
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nMerged summary saved to: {summary_file}")
    
    # Also save as CSV for easy analysis
    csv_file = osp.join(args.work_dir, f'MERGED_RESULTS_{timestamp}.csv')
    with open(csv_file, 'w') as f:
        f.write("A,B,mAP,mAP_50,mAP_75,mAP_s,mAP_m,mAP_l\n")
        for r in valid_results:
            f.write(f"{r['A']:.2f},{r['B']:.2f},{r['mAP']},{r['mAP_50']},{r['mAP_75']},{r['mAP_s']},{r['mAP_m']},{r['mAP_l']}\n")
    
    print(f"CSV results saved to: {csv_file}")


if __name__ == '__main__':
    main()
