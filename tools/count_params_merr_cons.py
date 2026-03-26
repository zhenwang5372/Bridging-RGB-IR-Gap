#!/usr/bin/env python3
"""Count parameters for each module in the IR_RGB_Merr_Cons model (excluding frozen CLIP)."""

import sys
sys.path.insert(0, '/home/enhao02/wangzhen')

import torch
from mmengine.registry import DefaultScope

with DefaultScope.overwrite_default_scope('mmyolo'):
    import yolo_world
    from mmyolo.registry import MODELS

    def fmt(n):
        if n >= 1e6:
            return f'{n/1e6:.2f}M'
        elif n >= 1e3:
            return f'{n/1e3:.2f}K'
        return str(n)

    def count(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def print_submodules(module, prefix='', depth=1):
        for name, child in module.named_children():
            t, tr = count(child)
            indent = '  ' * depth
            print(f'{indent}{prefix}{name:<35} {fmt(t):>12}  ({t:>10,})')
            if depth < 2:
                print_submodules(child, prefix=f'{name}.', depth=depth+1)

    rgb_out_channels = [128, 256, 512]
    ir_out_channels = [64, 128, 256]
    text_channels = 512
    num_classes = 4

    modules = {
        'RGB Backbone (YOLOv8s-CSPDarknet)': dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=1024,
            deepen_factor=0.33,
            widen_factor=0.5,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        'IR Backbone (LiteFFTIR)': dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        'IR Correction (IR_RGB_Merr_Cons)': dict(
            type='IR_RGB_Merr_Cons',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            correction_alpha=-0.5,
            enhancement_beta=0.5,
            d_k=128,
        ),
        'Fusion (MultiLevelRGBIRFusion)': dict(
            type='MultiLevelRGBIRFusion',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            reduction=4,
        ),
        'RGB Enhancement (TextGuidedV2)': dict(
            type='TextGuidedRGBEnhancementV2',
            rgb_channels=rgb_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            d_k=128,
        ),
        'Text Update (MultiScaleV4)': dict(
            type='MultiScaleTextUpdateV4',
            in_channels=rgb_out_channels,
            text_dim=text_channels,
            embed_channels=256,
            num_heads=8,
            with_scale=True,
        ),
        'Neck (SimpleChannelAlign)': dict(
            type='SimpleChannelAlign',
            in_channels=rgb_out_channels,
            out_channels=rgb_out_channels,
        ),
        'Aggregator (ClassDimAgg)': dict(
            type='ClassDimensionAggregator',
            in_channels=rgb_out_channels,
            num_classes=num_classes,
            aggregation_method='conv',
            fusion_type='concat',
        ),
        'Head (YOLOWorldHead)': dict(
            type='YOLOWorldHead',
            head_module=dict(
                type='YOLOWorldHeadModule',
                in_channels=rgb_out_channels,
                widen_factor=1.0,
                embed_dims=text_channels,
                num_classes=num_classes,
                reg_max=16,
                use_bn_head=True,
                use_einsum=True,
            ),
        ),
    }

    print('=' * 75)
    print('  Model Parameter Count (excluding frozen CLIP text encoder)')
    print('  Config: yolow_v2_rgb_ir_flir_ircorrection_IR_RGB_Merr_Cons')
    print('  Input:  640x640, num_classes=4')
    print('=' * 75)

    grand_total = 0
    grand_trainable = 0
    module_summary = []

    for name, cfg in modules.items():
        try:
            m = MODELS.build(cfg)
            m.eval()
            t, tr = count(m)
            grand_total += t
            grand_trainable += tr
            module_summary.append((name, t, tr))

            print(f'\n  {name}')
            print(f'  {"─" * 60}')
            print(f'  {"Total":>50}: {fmt(t):>10}  ({t:>10,})')
            print_submodules(m)
        except Exception as e:
            print(f'\n  {name}')
            print(f'    ⚠ Build failed: {e}')

    print('\n' + '=' * 75)
    print('  SUMMARY (excluding frozen CLIP)')
    print('=' * 75)
    print(f'\n  {"Module":<45} {"Params":>12} {"Ratio":>8}')
    print(f'  {"─" * 45} {"─" * 12} {"─" * 8}')
    for name, t, tr in module_summary:
        ratio = t / grand_total * 100 if grand_total > 0 else 0
        print(f'  {name:<45} {fmt(t):>12} {ratio:>7.1f}%')

    print(f'  {"─" * 45} {"─" * 12} {"─" * 8}')
    print(f'  {"TOTAL (non-CLIP)":<45} {fmt(grand_total):>12} {100.0:>7.1f}%')
    print(f'  {"Total params":>50}: {grand_total:>10,}')
    print(f'  {"Total trainable":>50}: {grand_trainable:>10,}')
    print()
    print(f'  CLIP text encoder (frozen, ref):  ~63.19M (not included above)')
    print(f'  Grand total (with CLIP):          ~{fmt(grand_total + 63_190_000)}')
    print('=' * 75)
