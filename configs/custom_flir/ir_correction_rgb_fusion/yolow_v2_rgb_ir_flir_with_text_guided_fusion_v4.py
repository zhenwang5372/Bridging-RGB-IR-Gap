# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR with Text-Guided Fusion V4 for FLIR Dataset
#
# ==================== V4 版本核心改进 ====================
#
# 相比 V3 版本的新增功能：
# 1. param_constraint: β和γ参数约束方式（解决学习到负值的问题）
# 2. mask_center: mask 零中心化方式（解决全正值问题，增加抑制能力）
#
# ==================== V4 新增接口 ====================
#
# 【接口1】param_constraint - β和γ参数约束方式
# ─────────────────────────────────────────────────────────
#
#   'softplus' ⭐ 默认:
#     β_pos = softplus(β), γ_pos = softplus(γ)
#     确保参数始终 > 0，目标区域 mask 高，背景区域 mask 低
#
#   'abs':
#     β_pos = |β|, γ_pos = |γ|
#     简单绝对值约束
#
#   'residual_alpha':
#     不约束 β/γ，改用 x_fused = x_rgb + α * (x_rgb * mask)
#     α 可学习补偿方向
#
#   'none':
#     不约束（与 V3 行为一致）
#
# 【接口2】mask_center - Mask 零中心化方式
# ─────────────────────────────────────────────────────────
#
#   'spatial_mean' ⭐ 默认:
#     mask_centered = mask - mean(mask)
#     高于均值→正值→增强，低于均值→负值→抑制
#
#   'tanh':
#     用 tanh 代替 sigmoid，输出范围 (-1, 1)
#
#   'smap_center':
#     S_map 阶段零中心化：S_map = S_map - mean(S_map)
#
#   'none':
#     不零中心化（与 V3 行为一致）
#
# ==================== 继承自 V3 的接口 ====================
#
# 【接口3】gap_method - GAP 计算方式
#   'logits' ⭐ 默认 | 'max' | 'entropy'
#
# 【接口4】smap_method - S_map 归一化方式
#   'sigmoid' | 'sigmoid_temp' | 'normalized' ⭐ 默认
#
# 【接口5】smap_order - S_map 计算顺序
#   'sum_first' ⭐ 默认 | 'multiply_first'
#
# 【接口6】mask_method - Mask 生成方式
#   'conv_gen' ⭐ 默认 | 'residual' | 'dual_branch' | 'se_spatial'
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (V4 默认推荐):
#   param_constraint='softplus', mask_center='spatial_mean'
#   适用: 大多数场景，解决 V3 的两个核心问题
#
# 配置2 (tanh 方案):
#   param_constraint='softplus', mask_center='tanh'
#   适用: 需要更大的负值范围 (-1, 1)
#
# 配置3 (残差补偿):
#   param_constraint='residual_alpha', mask_center='spatial_mean'
#   适用: 让模型自己学习增强/抑制方向
#
# 配置4 (V3 兼容):
#   param_constraint='none', mask_center='none'
#   适用: 与 V3 完全一致的行为

_base_ = ('../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
load_from = 'checkpoints/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

# ======================== Hyper-parameters ========================
num_classes = 4  # FLIR: car, person, bicycle, dog
num_training_classes = 4
max_epochs = 300
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 512
base_lr = 1.5e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
img_scale = (640, 640)

# ==================== V2/V3 参数配置 ====================
#
# GAP 计算方式
gap_method = 'logits'  # 'logits' | 'max' | 'entropy'

# S_map 归一化方式
smap_method = 'normalized'  # 'sigmoid' | 'sigmoid_temp' | 'normalized'

# S_map 计算顺序 (V3)
smap_order = 'sum_first'  # 'sum_first' | 'multiply_first'

# Mask 生成方式 (V3)
mask_method = 'residual'  # 'conv_gen' | 'residual' | 'dual_branch' | 'se_spatial'

# Mask 生成器的通道缩减比例 (V3)
mask_reduction = 8

# 温度参数（仅当 smap_method='sigmoid_temp' 时生效）
temperature = 1.0

# 融合参数初始值
fusion_beta = 1.0
fusion_gamma = 1.0
fusion_alpha = 0.1  # 仅 param_constraint='residual_alpha' 时使用

# ==================== V4 新增参数配置 ====================
#
# ⭐ β和γ参数约束方式
# 'softplus': 使用 softplus 确保 > 0（推荐）
# 'abs': 使用绝对值确保 > 0
# 'residual_alpha': 使用残差形式融合
# 'none': 不约束（V3 行为）
param_constraint = 'softplus'

# ⭐ Mask 零中心化方式
# 'spatial_mean': mask 减去空间均值（推荐）
# 'tanh': 使用 tanh 激活（范围 -1 到 1）
# 'smap_center': S_map 零中心化
# 'none': 不零中心化（V3 行为）
mask_center = 'tanh'

# ======================== Model Definition ========================
rgb_out_channels = [128, 256, 512]
ir_out_channels = [64, 128, 256]

model = dict(
    type='DualStreamYOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='FLIRDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        mean_ir=[0., 0., 0.],
        std_ir=[255., 255., 255.],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _delete_=True,
        type='DualStreamMultiModalYOLOBackboneWithTextGuidedFusion',
        
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=1024,
            deepen_factor=0.33,
            widen_factor=0.5,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        
        ir_model=dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        
        # ⭐ V4 版本：文本引导融合模块
        #
        # 【V4 新增参数说明】
        # - param_constraint: 控制 β和γ 的约束方式
        # - mask_center: 控制 mask 的零中心化方式
        #
        # 【切换配置示例】
        #
        # 示例1：V4 默认配置（推荐）
        #   param_constraint='softplus', mask_center='spatial_mean'
        #
        # 示例2：tanh 方案
        #   param_constraint='softplus', mask_center='tanh'
        #
        # 示例3：残差补偿方案
        #   param_constraint='residual_alpha', mask_center='spatial_mean'
        #
        # 示例4：V3 兼容模式
        #   param_constraint='none', mask_center='none'
        #
        text_guided_fusion=dict(
            type='TextGuidedRGBIRFusionV4',  # ⭐ 使用 V4 版本
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            beta=fusion_beta,
            gamma=fusion_gamma,
            alpha=fusion_alpha,
            # V2 参数
            gap_method=gap_method,
            smap_method=smap_method,
            temperature=temperature,
            # V3 参数
            smap_order=smap_order,
            mask_method=mask_method,
            mask_reduction=mask_reduction,
            # ⭐ V4 新增参数
            param_constraint=param_constraint,
            mask_center=mask_center,
        ),
        
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        
        with_text_model=True,
        frozen_stages=-1,
    ),

    neck=dict(
        _delete_=True,
        type='SimpleChannelAlign',
        in_channels=rgb_out_channels,
        out_channels=rgb_out_channels,
    ),

    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_training_classes)
    ),
)

# ======================== Data Settings ========================
data_root = '/home/ssd1/users/wangzhen01/YOLO-World-master_2/data/flir/root/autodl-tmp/data/FLIR_V1_aligned/align/'
train_ann_file = 'annotations_fixed/train_fixed.json'
val_ann_file = 'annotations_fixed/val_fixed.json'
train_data_prefix = 'JPEGImages/'
val_data_prefix = 'JPEGImages/'

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFile',
         ir_suffix='_PreviewData.jpeg',
         rgb_suffix='_RGB.jpg'),
    dict(type='LoadAnnotations', with_bbox=True),
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDualModalInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'flip', 'flip_direction', 'texts', 'scale_factor',
                    'pad_param', 'img_ir_path', 'img_ir_shape'))
]

train_pipeline = [
    *pre_transform,
    dict(type='SyncMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=pre_transform,
         prob=1.0),
    dict(type='SyncRandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         max_translate_ratio=0.1,
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
         border=(-img_scale[0] // 2, -img_scale[1] // 2),
         border_val=(114, 114, 114)),
    dict(type='SyncLetterResize',
         scale=img_scale,
         pad_val=dict(img=114, img_ir=114),
         allow_scale_up=True),
    dict(type='SyncRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='DualModalityPhotometricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
         ir_brightness_delta=20,
         ir_contrast_range=(0.8, 1.2),
         prob=0.5),
    dict(type='ThermalSpecificAugmentation',
         fpa_noise_level=0.02,
         crossover_prob=0.1,
         scale_range=(0.95, 1.05),
         shift_range=10,
         prob=0.3),
    *text_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='SyncLetterResize',
         scale=img_scale,
         pad_val=dict(img=114, img_ir=114),
         allow_scale_up=True),
    dict(type='SyncRandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         max_translate_ratio=0.05,
         scaling_ratio_range=(0.5, 1.5),
         border=(0, 0),
         border_val=(114, 114, 114)),
    dict(type='SyncRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='DualModalityPhotometricDistortion', prob=0.3),
    *text_transform,
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFile',
         ir_suffix='_PreviewData.jpeg',
         rgb_suffix='_RGB.jpg'),
    dict(type='SyncLetterResize',
         scale=img_scale,
         pad_val=dict(img=114, img_ir=114),
         allow_scale_up=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(type='PackDualModalInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts',
                    'img_ir_path', 'img_ir_shape'))
]

# ======================== Dataset Configuration ========================
train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='FLIRDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        ir_suffix='_PreviewData.jpeg',
        rgb_suffix='_RGB.jpg',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path='data/flir/texts/flir_class_texts.json',
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='FLIRDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        ir_suffix='_PreviewData.jpeg',
        rgb_suffix='_RGB.jpg',
        test_mode=True,
    ),
    class_text_path='data/flir/texts/flir_class_texts.json',
    pipeline=test_pipeline,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=train_dataset,
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)

test_dataloader = val_dataloader

# ======================== Evaluator ========================
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox',
    classwise=True,
)
test_evaluator = val_evaluator

# ======================== Training Settings ========================
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP_50',
        rule='greater',
        max_keep_ckpts=3
    )
)

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2),
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)]
)

# ======================== Optimizer Configuration ========================
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'backbone.ir_model': dict(lr_mult=1.0),
            'backbone.text_guided_fusion': dict(lr_mult=1.0),
            'logit_scale': dict(weight_decay=0.0)
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)
