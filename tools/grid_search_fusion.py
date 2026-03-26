# Copyright (c) OpenMMLab. All rights reserved.
"""
Grid search script for finding optimal fusion coefficients.
Formula: output = A * x_rgb + B * self.gamma * fused

Supports multi-GPU parallel execution by specifying A range per GPU.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python tools/grid_search_fusion.py \
        configs/custom_llvip/yolow_v2_rgb_ir_llvip_no_update.py \
        work_dirs/LLVIP/no_update/batch16_1280_2gpu/best_coco_bbox_mAP_50_epoch_50.pth \
        --work-dir work_dirs/grid_search_results \
        --gpu-id 0 --a-start 0.0 --a-end 0.25
"""
import argparse
import os
import os.path as osp
import sys
from datetime import datetime
from itertools import product

import torch
import torch.nn.functional as F

# Monkey-patch torch.load to use weights_only=False by default
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from mmengine.config import Config
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


# Global variables for current coefficients
CURRENT_A = 1.0
CURRENT_B = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid search for optimal fusion coefficients')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/grid_search_results',
        help='the directory to save results')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID for logging purposes')
    parser.add_argument(
        '--a-start',
        type=float,
        default=0.0,
        help='Start value for A (inclusive)')
    parser.add_argument(
        '--a-end',
        type=float,
        default=1.0,
        help='End value for A (inclusive)')
    parser.add_argument(
        '--a-step',
        type=float,
        default=0.05,
        help='Step size for A')
    parser.add_argument(
        '--b-start',
        type=float,
        default=1.0,
        help='Start value for B (inclusive)')
    parser.add_argument(
        '--b-end',
        type=float,
        default=2.0,
        help='End value for B (inclusive)')
    parser.add_argument(
        '--b-step',
        type=float,
        default=0.05,
        help='Step size for B')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def patch_fusion_module():
    """
    Patch the LightweightCrossFusion forward method to use global A, B coefficients.
    """
    from yolo_world.models.necks.rgb_ir_fusion import LightweightCrossFusion
    
    def patched_forward(self, x_rgb, x_ir):
        global CURRENT_A, CURRENT_B
        
        # Align IR channels to RGB
        x_ir_aligned = self.ir_align(x_ir)
        
        # Resize IR features if spatial dimensions don't match
        if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
            x_ir_aligned = F.interpolate(x_ir_aligned, size=x_rgb.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        # Generate spatial attention from IR features
        attention_map = self.attention_gen(x_ir_aligned)
        
        # Apply attention to RGB features
        x_rgb_attended = x_rgb * attention_map
        
        # Cross-modality fusion
        combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
        fused = self.cross_conv(combined)
        
        # Use global coefficients: output = A * x_rgb + B * self.gamma * fused
        output = CURRENT_A * x_rgb + CURRENT_B * self.gamma * fused
        
        return output
    
    LightweightCrossFusion.forward = patched_forward


def format_table_row(category, mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l):
    """Format a single table row."""
    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):  # nan check
            return "nan  "
        return f"{v:.3f}"
    
    return f"| {category:<8} | {fmt(mAP)} | {fmt(mAP_50)} | {fmt(mAP_75)} | {fmt(mAP_s)} | {fmt(mAP_m)} | {fmt(mAP_l)} |"


def format_result_block(A, B, metrics):
    """Format a result block for one combination."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Combination: A={A:.2f}, B={B:.2f}")
    lines.append(f"  Formula: output = {A:.2f} * x_rgb + {B:.2f} * self.gamma * fused")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append("| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |")
    lines.append("|----------|-------|--------|--------|-------|-------|-------|")
    
    # Extract metrics
    mAP = metrics.get('coco/bbox_mAP', metrics.get('bbox_mAP', float('nan')))
    mAP_50 = metrics.get('coco/bbox_mAP_50', metrics.get('bbox_mAP_50', float('nan')))
    mAP_75 = metrics.get('coco/bbox_mAP_75', metrics.get('bbox_mAP_75', float('nan')))
    mAP_s = metrics.get('coco/bbox_mAP_s', metrics.get('bbox_mAP_s', float('nan')))
    mAP_m = metrics.get('coco/bbox_mAP_m', metrics.get('bbox_mAP_m', float('nan')))
    mAP_l = metrics.get('coco/bbox_mAP_l', metrics.get('bbox_mAP_l', float('nan')))
    
    lines.append(format_table_row("person", mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l))
    lines.append("")
    
    return "\n".join(lines), mAP_50


def generate_values(start, end, step):
    """Generate a list of values from start to end with given step."""
    values = []
    current = start
    while current <= end + 1e-9:  # small epsilon for float comparison
        values.append(round(current, 2))
        current += step
    return values


def main():
    global CURRENT_A, CURRENT_B
    
    args = parse_args()
    
    # Generate A and B values based on arguments
    A_values = generate_values(args.a_start, args.a_end, args.a_step)
    B_values = generate_values(args.b_start, args.b_end, args.b_step)
    
    total_combinations = len(A_values) * len(B_values)
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Log file (per GPU)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = osp.join(args.work_dir, f'grid_search_gpu{args.gpu_id}_{timestamp}.log')
    
    # Write header
    with open(log_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  Grid Search - GPU {args.gpu_id}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Config: {args.config}\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Formula: output = A * x_rgb + B * self.gamma * fused\n")
        f.write(f"  A range: {A_values[0]:.2f} to {A_values[-1]:.2f} (step {args.a_step})\n")
        f.write(f"  B range: {B_values[0]:.2f} to {B_values[-1]:.2f} (step {args.b_step})\n")
        f.write(f"  A values: {A_values}\n")
        f.write(f"  Combinations for this GPU: {total_combinations}\n")
        f.write("=" * 70 + "\n")
    
    print("=" * 70)
    print(f"  Grid Search - GPU {args.gpu_id}")
    print("=" * 70)
    print(f"  A values: {A_values}")
    print(f"  B values: {B_values}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Log file: {log_file}")
    print("=" * 70)
    
    # Patch the fusion module
    patch_fusion_module()
    
    # Results storage for summary
    all_results = []
    
    # Run grid search
    combo_idx = 0
    for A in A_values:
        for B in B_values:
            combo_idx += 1
            
            print(f"\n[GPU{args.gpu_id}] [{combo_idx}/{total_combinations}] Testing A={A:.2f}, B={B:.2f}")
            
            # Set global coefficients
            CURRENT_A = A
            CURRENT_B = B
            
            try:
                # Load fresh config
                cfg = Config.fromfile(args.config)
                cfg.launcher = args.launcher
                cfg.work_dir = osp.join(args.work_dir, f'gpu{args.gpu_id}_A{A:.2f}_B{B:.2f}')
                cfg.load_from = args.checkpoint
                
                # Suppress verbose output
                cfg.log_level = 'WARNING'
                
                is_metainfo_lower(cfg)
                
                # Build runner
                if 'runner_type' not in cfg:
                    runner = Runner.from_cfg(cfg)
                else:
                    runner = RUNNERS.build(cfg)
                
                # Run test
                metrics = runner.test()
                
                # Format and save result
                result_block, mAP_50 = format_result_block(A, B, metrics)
                
                with open(log_file, 'a') as f:
                    f.write(result_block)
                
                print(result_block)
                
                # Store for summary
                all_results.append({
                    'A': A,
                    'B': B,
                    'mAP_50': mAP_50,
                    'metrics': metrics
                })
                
            except Exception as e:
                error_msg = f"\n[ERROR] A={A:.2f}, B={B:.2f}: {str(e)}\n"
                print(error_msg)
                with open(log_file, 'a') as f:
                    f.write(error_msg)
                all_results.append({
                    'A': A,
                    'B': B,
                    'mAP_50': float('nan'),
                    'metrics': {}
                })
    
    # Write summary for this GPU
    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append(f"  GPU {args.gpu_id} SUMMARY - Results Sorted by mAP_50")
    summary_lines.append("=" * 70 + "\n")
    
    # Sort by mAP_50 descending
    valid_results = [r for r in all_results if r['mAP_50'] == r['mAP_50']]  # filter nan
    valid_results.sort(key=lambda x: x['mAP_50'], reverse=True)
    
    summary_lines.append("| Rank | A    | B    | mAP_50 |")
    summary_lines.append("|------|------|------|--------|")
    
    for rank, r in enumerate(valid_results, 1):
        summary_lines.append(f"| {rank:4d} | {r['A']:.2f} | {r['B']:.2f} | {r['mAP_50']:.4f} |")
    
    summary_lines.append("")
    summary_lines.append("=" * 70)
    
    if valid_results:
        best = valid_results[0]
        summary_lines.append(f"  GPU {args.gpu_id} BEST COMBINATION:")
        summary_lines.append(f"    A = {best['A']:.2f}")
        summary_lines.append(f"    B = {best['B']:.2f}")
        summary_lines.append(f"    mAP_50 = {best['mAP_50']:.4f}")
    
    summary_lines.append("=" * 70)
    
    summary_text = "\n".join(summary_lines)
    
    with open(log_file, 'a') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n[GPU{args.gpu_id}] Results saved to: {log_file}")


if __name__ == '__main__':
    main()
