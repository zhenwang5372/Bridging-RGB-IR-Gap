# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR with Text-Guided Fusion V5 for FLIR Dataset
#
# ==================== V5 版本核心改进 ====================
#
# 相比 V4 的主要改进：
# 1. 去除 x_ir_aligned 对 S_map 的干扰，直接使用 S_map 引导融合
# 2. 使用归一化 sigmoid 替代 softmax/普通 sigmoid，解决值过小/分布偏移问题
# 3. 去除类别权重 w（V4 中 w 全退化为 1.0），直接求和
# 4. α 使用 softplus 约束，保证 > 0
# 5. 支持三种融合模式，可通过配置切换
# 6. 支持 mask 监督损失（可选）
#
# ==================== V5 核心接口 ====================
#
# 【接口1】fusion_mode - 融合模式（三选一）
# ─────────────────────────────────────────────────────────
#
#   'smap_direct' ⭐ 默认推荐:
#     mask = S_map - mean(S_map)
#     x_fused = x_rgb + α * x_rgb * mask
#     特点: 最简单直接，S_map 信息无损传递
#
#   'channel_spatial':
#     spatial_mask = S_map
#     channel_attn = sigmoid(pool(ir_align(x_ir)))
#     mask = channel_attn * spatial_mask - mean(...)
#     x_fused = x_rgb + α * x_rgb * mask
#     特点: 空间和通道分离，IR 只提供通道注意力
#
#   'dual_gate':
#     enhance_gate = sigmoid(γ_e * S_map)
#     suppress_gate = sigmoid(-γ_s * S_map)
#     x_fused = x_rgb + α * x_rgb * enhance_gate - β * x_rgb * suppress_gate
#     特点: 显式的增强/抑制机制
#
# 【接口2】alpha_init / alpha_constraint - α 参数
# ─────────────────────────────────────────────────────────
#   alpha_init: α 初始值（默认 0.1）
#   alpha_constraint: α 约束方式
#     'softplus' ⭐ 默认: α_pos = softplus(α)，保证 > 0
#     'abs': α_pos = |α|
#     'sigmoid': α_pos = sigmoid(α)，范围 (0, 1)
#     'none': 不约束
#
# 【接口3】smap_normalize - S_map 归一化方式
# ─────────────────────────────────────────────────────────
#   'sigmoid_centered' ⭐ 默认:
#     S_map = sigmoid(S_map_raw - mean(S_map_raw))
#     先减均值再 sigmoid，确保有高有低
#
#   'sigmoid':
#     S_map = sigmoid(S_map_raw)
#     简单 sigmoid，可能全高或全低
#
#   'none':
#     不归一化，保持原始值
#
# 【接口4】use_class_weight - 是否使用类别权重 w
# ─────────────────────────────────────────────────────────
#   False ⭐ 默认: 不使用（V4 中 w 全退化为 1.0，无意义）
#   True: 使用类别权重
#
# 【接口5】use_refine_conv - 是否使用细化卷积
# ─────────────────────────────────────────────────────────
#   False ⭐ 默认: 不使用（简化路径）
#   True: 使用卷积细化 mask
#
# 【接口6】mask_center - mask 零中心化方式
# ─────────────────────────────────────────────────────────
#   'spatial_mean' ⭐ 默认:
#     mask_centered = mask - mean(mask)
#     高于均值→正值→增强，低于均值→负值→抑制
#
#   'none':
#     不零中心化
#
# 【接口7】use_mask_supervision - 是否使用 mask 监督损失
# ─────────────────────────────────────────────────────────
#   False ⭐ 默认: 不使用
#   True: 使用 GT boxes 监督 S_map，需配合 DualStreamYOLOWorldDetectorV2
#
# ==================== 推荐配置组合 ====================
#
# 配置1 (V5 默认推荐 - 最简单):
#   fusion_mode='smap_direct'
#   alpha_constraint='softplus', alpha_init=0.1
#   smap_normalize='sigmoid_centered'
#   use_class_weight=False
#   use_refine_conv=False
#   mask_center='spatial_mean'
#
# 配置2 (通道-空间分离):
#   fusion_mode='channel_spatial'
#   其他同配置1
#
# 配置3 (双向门控):
#   fusion_mode='dual_gate'
#   其他同配置1
#
# 配置4 (带 mask 监督):
#   配置1 + use_mask_supervision=True, mask_loss_weight=0.1
#

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

# ==================== V5 参数配置 ====================
#
# ⭐ 融合模式（三选一）
# 'smap_direct': S_map 直接引导（推荐）
# 'channel_spatial': 通道-空间分离注意力
# 'dual_gate': 双向门控
fusion_mode = 'smap_direct'

# ⭐ α 参数
alpha_init = 0.1  # α 初始值
alpha_constraint = 'softplus'  # 'softplus' | 'abs' | 'sigmoid' | 'none'

# ⭐ S_map 归一化方式
smap_normalize = 'sigmoid_centered'  # 'sigmoid_centered' | 'sigmoid' | 'none'

# ⭐ 是否使用类别权重 w
use_class_weight = False  # False（推荐） | True

# ⭐ 是否使用细化卷积
use_refine_conv = False  # False（推荐） | True
refine_conv_kernel = 1   # 卷积核大小，仅当 use_refine_conv=True 时生效

# ⭐ mask 零中心化方式
mask_center = 'spatial_mean'  # 'spatial_mean' | 'none'

# ⭐ 是否使用 mask 监督损失
use_mask_supervision = True  # False | True
mask_loss_weight = 0.1        # mask 损失权重

# ======================== Model Definition ========================
rgb_out_channels = [128, 256, 512]
ir_out_channels = [64, 128, 256]

model = dict(
    # ⭐ V5 使用 V2 Detector（支持 mask 监督损失）
    type='DualStreamYOLOWorldDetectorV2',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    # ⭐ V5 新增：mask 监督配置
    use_mask_supervision=use_mask_supervision,
    mask_loss_weight=mask_loss_weight,
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
        # ⭐ V5 使用 V2 Backbone（支持返回 S_map）
        type='DualStreamMultiModalYOLOBackboneWithTextGuidedFusionV2',
        
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
        
        # ⭐ V5 版本：文本引导融合模块
        #
        # 【配置说明】
        # - fusion_mode: 融合模式，三选一
        # - alpha_init/alpha_constraint: α 参数配置
        # - smap_normalize: S_map 归一化方式
        # - use_class_weight: 是否使用类别权重
        # - use_refine_conv: 是否使用细化卷积
        # - mask_center: mask 零中心化方式
        #
        text_guided_fusion=dict(
            type='TextGuidedRGBIRFusionV5',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            # V5 参数
            fusion_mode=fusion_mode,
            alpha_init=alpha_init,
            alpha_constraint=alpha_constraint,
            smap_normalize=smap_normalize,
            use_class_weight=use_class_weight,
            use_refine_conv=use_refine_conv,
            refine_conv_kernel=refine_conv_kernel,
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

# ======================== Evaluation Settings ========================
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann_file,
    metric=['bbox'],
    proposal_nums=(100, 1, 10),
)
test_evaluator = val_evaluator

# ======================== Training Settings ========================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP_50',
        rule='greater',
        max_keep_ckpts=3
    ),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50),
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
         switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    constructor='YOLOWv5OptimizerConstructor'
)
