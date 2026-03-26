# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR Simplified Trimodal Configuration (Tuned)
# 优化版：降低学习率，增加warmup，调整权重

_base_ = './yolow_v2_rgb_ir_flir_simplified.py'

# ======================== 关键超参数调整 ========================

# 1. 降低基础学习率（原来2e-3太大）
base_lr = 1e-3  # 降低50%

# 2. 增加warmup
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=298,  # max_epochs - close_mosaic_epochs
        switch_pipeline=_base_.train_pipeline_stage2
    )
]

# 3. 优化学习率调度
param_scheduler = [
    # Warmup阶段（前5个epoch）
    dict(
        type='LinearLR',
        start_factor=0.001,  # 从0.001*base_lr开始
        by_epoch=True,
        begin=0,
        end=5,  # 增加warmup时长
    ),
    # 主训练阶段
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,  # 最小学习率
        begin=5,
        end=300,
        T_max=295,
        by_epoch=True,
    )
]

# 4. 优化器配置：新模块使用更小的学习率
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=0.025,  # 降低weight decay
        batch_size_per_gpu=16
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # Text encoder保持低学习率
            'backbone.ir_model': dict(lr_mult=0.5),     # IR backbone降低
            'neck.rgb_enhance': dict(lr_mult=0.1),      # 新模块：降低10倍
            'neck.text_update': dict(lr_mult=0.1),      # 新模块：降低10倍
            'logit_scale': dict(weight_decay=0.0),
            # Gamma参数特殊处理
            'gamma': dict(lr_mult=0.01, weight_decay=0.0),  # 非常小的学习率
        }
    ),
    constructor='YOLOWv5OptimizerConstructor',
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# 5. 调整训练策略
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=1,
    dynamic_intervals=[(280, 1)]  # 最后20个epoch每个epoch都验证
)

# 6. 数据增强调整（降低强度，避免过拟合）
train_pipeline_stage1 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFile',
         ir_suffix='_PreviewData.jpeg',
         rgb_suffix='_RGB.jpg'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='SyncMosaic',
         img_scale=(640, 640),
         pad_val=114.0,
         prob=0.8,  # 降低mosaic概率
         pre_transform=[
             dict(type='LoadImageFromFile', backend_args=None),
             dict(type='LoadIRImageFromFile',
                  ir_suffix='_PreviewData.jpeg',
                  rgb_suffix='_RGB.jpg'),
             dict(type='LoadAnnotations', with_bbox=True),
         ]),
    dict(type='SyncRandomAffine',
         scaling_ratio_range=(0.7, 1.3),  # 降低缩放范围
         border=(-320, -320),
         border_val=(114, 114, 114),
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         max_translate_ratio=0.1),
    dict(type='SyncLetterResize',
         scale=(640, 640),
         pad_val=dict(img=114, img_ir=114),
         allow_scale_up=True),
    dict(type='SyncRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='DualModalityPhotometricDistortion',
         brightness_delta=20,  # 降低亮度变化
         contrast_range=(0.7, 1.3),  # 降低对比度变化
         saturation_range=(0.7, 1.3),
         hue_delta=10,
         ir_brightness_delta=15,
         ir_contrast_range=(0.8, 1.2),
         prob=0.5),
    dict(type='ThermalSpecificAugmentation',
         prob=0.2,  # 降低热成像增强概率
         fpa_noise_level=0.01,
         scale_range=(0.98, 1.02),
         shift_range=5,
         crossover_prob=0.05),
    dict(type='RandomLoadText',
         num_neg_samples=(4, 4),
         max_num_samples=4,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDualModalInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'flip', 'flip_direction', 'texts', 'scale_factor',
                    'pad_param', 'img_ir_path', 'img_ir_shape'))
]

# 7. 评估指标调整（使用完整路径）
data_root = '/root/autodl-tmp/data/FLIR_V1_aligned/align/'
val_ann_file = 'annotations_fixed/val_fixed.json'

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False,
    classwise=True,  # 输出每个类别的指标
)

