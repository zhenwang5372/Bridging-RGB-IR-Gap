#!/usr/bin/env python3
"""Profile model: parameter count (total and per-module), GFLOPs, and FPS."""

import argparse
import time
import numpy as np
import torch
from mmengine.config import Config


def count_parameters(model):
    info = {}
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters()
                        if p.requires_grad)
        info[name] = (total, trainable)
    return info


def fmt(n):
    if n >= 1e6:
        return f'{n/1e6:.2f}M'
    elif n >= 1e3:
        return f'{n/1e3:.2f}K'
    return str(n)


def build_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        ci = cfg.custom_imports
        import_modules_from_strings(
            ci.get('imports', []),
            ci.get('allow_failed_imports', False))

    from mmengine.registry import DefaultScope
    DefaultScope.get_instance('profile', scope_name='mmyolo')

    from mmyolo.registry import MODELS
    model = MODELS.build(cfg.model)
    model = model.to(device)
    model.eval()
    if checkpoint_path:
        import mmengine
        ckpt = mmengine.load(checkpoint_path, map_location=device)
        sd = ckpt.get('state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
    return model, cfg


def make_dummy_data_samples(h, w, texts):
    """Build a list containing one DetDataSample with text info."""
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    ds = DetDataSample()
    ds.set_metainfo({
        'img_shape': (h, w),
        'ori_shape': (h, w),
        'scale_factor': (1.0, 1.0),
        'pad_param': np.array([0, 0, 0, 0], dtype=np.float32),
        'batch_input_shape': (h, w),
    })
    ds.texts = texts
    ds.gt_instances = InstanceData()
    ds.gt_instances.bboxes = torch.zeros((0, 4))
    ds.gt_instances.labels = torch.zeros((0,), dtype=torch.long)
    return [ds]


def measure_fps(model, device, h, w, warmup, repeats):
    rgb = torch.randn(1, 3, h, w, device=device)
    ir = torch.randn(1, 3, h, w, device=device)
    texts = ['person', 'car', 'bicycle', 'dog']
    data_samples = make_dummy_data_samples(h, w, texts)

    with torch.no_grad():
        for i in range(warmup):
            try:
                model.forward(rgb, data_samples, mode='predict',
                              inputs_ir=ir)
            except Exception as e:
                print(f'  Forward failed at warmup iter {i}: {e}')
                return -1

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model.forward(rgb, data_samples, mode='predict',
                          inputs_ir=ir)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return repeats / (t1 - t0)


def estimate_gflops(model, device, h, w):
    rgb = torch.randn(1, 3, h, w, device=device)
    ir = torch.randn(1, 3, h, w, device=device)
    texts = ['person', 'car', 'bicycle', 'dog']

    try:
        from fvcore.nn import FlopCountAnalysis

        class Wrap(torch.nn.Module):
            def __init__(self, m, h, w, texts):
                super().__init__()
                self.m = m
                self.h = h
                self.w = w
                self.texts = texts

            def forward(self, rgb_in, ir_in):
                ds = make_dummy_data_samples(self.h, self.w, self.texts)
                return self.m.forward(rgb_in, ds, mode='predict',
                                     inputs_ir=ir_in)

        wrapped = Wrap(model, h, w, texts)
        flops = FlopCountAnalysis(wrapped, (rgb, ir))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception as e:
        print(f'  fvcore failed: {e}')

    try:
        from thop import profile as thop_profile

        class Wrap2(torch.nn.Module):
            def __init__(self, m, h, w, texts):
                super().__init__()
                self.m = m
                self.h = h
                self.w = w
                self.texts = texts

            def forward(self, rgb_in, ir_in):
                ds = make_dummy_data_samples(self.h, self.w, self.texts)
                return self.m.forward(rgb_in, ds, mode='predict',
                                     inputs_ir=ir_in)

        wrapped2 = Wrap2(model, h, w, texts)
        macs, _ = thop_profile(wrapped2, inputs=(rgb, ir), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f'  thop also failed: {e}')

    return None


def main():
    parser = argparse.ArgumentParser(description='Profile model')
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--img-scale', nargs=2, type=int, default=[640, 640])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=200)
    args = parser.parse_args()

    h, w = args.img_scale
    print('=' * 70)
    print('Model Profiling')
    print('=' * 70)
    print(f'Config: {args.config}')
    print(f'Device: {args.device}')
    print(f'Input:  {h}x{w}')

    model, cfg = build_model(args.config, args.checkpoint, args.device)

    # ---- Parameters ----
    print('\n' + '=' * 70)
    print('Parameter Count')
    print('=' * 70)

    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    print(f'\n  {"Module":<40} {"Total":>12} {"Trainable":>12}')
    print(f'  {"-"*40} {"-"*12} {"-"*12}')

    module_info = count_parameters(model)
    for name, (total, trainable) in sorted(module_info.items()):
        print(f'  {name:<40} {fmt(total):>12} {fmt(trainable):>12}')

    if hasattr(model, 'backbone'):
        print(f'\n  --- Backbone Sub-modules ---')
        for name, module in model.backbone.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad)
            print(f'    backbone.{name:<30} {fmt(total):>12} '
                  f'{fmt(trainable):>12}')

    print(f'\n  {"TOTAL":<40} {fmt(total_all):>12} '
          f'{fmt(trainable_all):>12}')

    # ---- GFLOPs ----
    print('\n' + '=' * 70)
    print('GFLOPs Estimation')
    print('=' * 70)
    gflops = estimate_gflops(model, args.device, h, w)
    if gflops is not None:
        print(f'\n  GFLOPs: {gflops:.2f}')
    else:
        print('\n  Could not estimate GFLOPs (install fvcore or thop)')

    # ---- FPS ----
    print('\n' + '=' * 70)
    print('FPS Measurement')
    print('=' * 70)
    print(f'  Warmup: {args.warmup}, Repeats: {args.repeats}')
    fps = measure_fps(model, args.device, h, w, args.warmup, args.repeats)
    if fps > 0:
        print(f'  FPS: {fps:.2f}')
        print(f'  Latency: {1000/fps:.2f} ms/image')
    else:
        print('  FPS measurement failed')

    # ---- Summary ----
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'  Total Params:     {fmt(total_all)} ({total_all:,})')
    print(f'  Trainable Params: {fmt(trainable_all)} ({trainable_all:,})')
    if gflops is not None:
        print(f'  GFLOPs:           {gflops:.2f}')
    if fps > 0:
        print(f'  FPS:              {fps:.2f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
