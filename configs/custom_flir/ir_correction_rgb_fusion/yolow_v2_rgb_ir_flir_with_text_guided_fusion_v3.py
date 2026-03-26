# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR with Text-Guided Fusion V3 (Scheme 2 进阶版) for FLIR Dataset
#
# ==================== V3 版本新增接口 ====================
#
# 【接口1】smap_order - S_map 计算顺序
# ─────────────────────────────────────────────────────────
#
#   'multiply_first' (先乘后加):
#     S_map = Σ_c [ (w_c × A_rgb^c) ⊙ (w_c × A_ir^c) ]
#     特点: 保留类别级别的细粒度交互
#
#   'sum_first' (先加后乘) ⭐ 默认:
#     A_rgb_agg = Σ_c (w_c × A_rgb^c)
#     A_ir_agg = Σ_c (w_c × A_ir^c)
#     S_map = A_rgb_agg ⊙ A_ir_agg
#     特点: 先聚合类别信息，再计算一致性
#
# 【接口2】mask_method - Mask 生成方式
# ─────────────────────────────────────────────────────────
#
#   'conv_gen' (方案A: 轻量级卷积生成器) ⭐ 默认:
#     步骤1: mask_raw = σ(β·X_ir + γ·S_map) → [B, C, H, W]
#     步骤2: mask = Conv(C→mid) → BN → ReLU → Conv(mid→C) → Sigmoid
#     输出: mask [B, C, H, W]
#     特点: 先用原始公式，再用卷积学习空间特征
#
#   'residual' (方案B: 残差 Mask 细化):
#     步骤: mask_raw = σ(β·X_ir + γ·S_map)
#           delta = refine_conv(mask_raw)
#           mask = clamp(mask_raw + 0.1×delta, 0, 1)
#     特点: 保留原有逻辑，用残差学习增量
#
#   'dual_branch' (方案C: 双分支 Mask 生成):
#     空间分支: mask_spatial = spatial_conv(S_map) → [B, 1, H, W]
#     通道分支: mask_channel = channel_conv(X_ir) → [B, C, H, W]
#     融合: mask = mask_spatial × mask_channel
#     特点: 解耦空间和通道信息
#
#   'se_spatial' (方案D: SE通道注意力 + 空间卷积):
#     SE块: se_weight = SE(X_ir) → [B, C, 1, 1]
#     空间: S_refined = spatial_conv(S_map)
#     组合: mask = σ(β × se_weight × X_ir + γ × S_refined)
#     特点: 通道注意力增强重要通道，空间卷积细化S_map
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (默认推荐):
#   smap_order='sum_first', mask_method='conv_gen'
#   适用: 一般场景，卷积学习能力强
#
# 配置2 (轻量级):
#   smap_order='sum_first', mask_method='residual'
#   适用: 参数量敏感场景
#
# 配置3 (高精度):
#   smap_order='multiply_first', mask_method='dual_branch'
#   适用: 需要细粒度类别交互
#
# 配置4 (通道注意力增强):
#   smap_order='sum_first', mask_method='se_spatial'
#   适用: 通道特征重要性差异大的场景

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

# ==================== V3 核心参数配置 ====================
#
# GAP 计算方式（与 V2 相同）
gap_method = 'logits'  # 'logits' | 'max' | 'entropy'

# S_map 归一化方式（与 V2 相同）
smap_method = 'normalized'  # 'sigmoid' | 'sigmoid_temp' | 'normalized'

# ⭐ V3 新增：S_map 计算顺序
# 'sum_first': 先按类别求和，再做 Hadamard 积（默认）
# 'multiply_first': 先做 Hadamard 积，再按类别求和
smap_order = 'sum_first'

# ⭐ V3 新增：Mask 生成方式
# 'conv_gen': 轻量级卷积生成器（默认）
# 'residual': 残差 Mask 细化
# 'dual_branch': 双分支 Mask 生成
# 'se_spatial': SE通道注意力 + 空间卷积
mask_method = 'conv_gen'

# ⭐ V3 新增：Mask 生成器的通道缩减比例（仅 conv_gen/residual/dual_branch 有效）
mask_reduction = 8

# 温度参数（仅当 smap_method='sigmoid_temp' 时生效）
temperature = 1.0

# 融合参数
fusion_beta = 1.0
fusion_gamma = 0.5

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
        
        # ⭐ V3 版本：文本引导融合模块（进阶版）
        #
        # 新增参数说明：
        # - smap_order: 控制 S_map 的计算顺序
        # - mask_method: 控制 Mask 的生成方式
        # - mask_reduction: Mask 生成器的通道缩减比例
        #
        # 【切换配置示例】
        #
        # 示例1：默认配置（推荐）
        #   smap_order='sum_first', mask_method='conv_gen'
        #
        # 示例2：保留 V2 逻辑 + 卷积 Mask
        #   smap_order='multiply_first', mask_method='conv_gen'
        #
        # 示例3：轻量级残差方案
        #   smap_order='sum_first', mask_method='residual'
        #
        # 示例4：双分支方案
        #   smap_order='sum_first', mask_method='dual_branch'
        #
        # 示例5：SE注意力方案
        #   smap_order='sum_first', mask_method='se_spatial'
        #
        text_guided_fusion=dict(
            type='TextGuidedRGBIRFusionV3',  # ⭐ 使用 V3 版本
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            beta=fusion_beta,
            gamma=fusion_gamma,
            # V2 参数
            gap_method=gap_method,
            smap_method=smap_method,
            temperature=temperature,
            # ⭐ V3 新增参数
            smap_order=smap_order,
            mask_method=mask_method,
            mask_reduction=mask_reduction,
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

find_unused_parameters = True
