# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR with Stable Training (延长warmup + 梯度裁剪)
#
# 改进点：
# 1. 延长warmup期到10个epoch（原先5个epoch warmup太快导致振荡）
# 2. 增强梯度裁剪（确保max_norm=10.0）
# 3. 降低初始学习率增长速度（从lr*1e-5开始增长到lr，更平滑）
#
# 基于 yolow_v2_rgb_ir_flir_class_specific_v2.py
# 核心架构保持不变，仅优化训练稳定性

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
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
base_lr = 1.5e-3  # 保持与阶段1相同的学习率
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
img_scale = (640, 640)

# IR Correction参数（保留）
correction_alpha = 0.3
temperature = 0.07  # 保持阶段1的temperature

# ⭐ RGB Enhancement V2参数
d_k = 128  # Attention的key维度

# ⭐ Text Update V2参数
text_hidden_dim = 256  # Cross-Attention隐藏维度
text_scale_init = 0.0  # 残差缩放初始值
text_fusion_method = 'learned_weight'  # 或 'equal'

# ⭐ Head V2参数
use_bn_head = True  # 使用BN版本的ContrastiveHead
use_einsum = True  # 使用einsum计算相似度

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
        type='DualStreamMultiModalYOLOBackboneWithClassSpecificV2',  # ⭐ V2 Backbone
        
        # RGB backbone（保持不变）
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=1024,
            deepen_factor=0.33,
            widen_factor=0.5,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        
        # IR backbone（保持不变）
        ir_model=dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        
        # Text model（保持不变）
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        
        # IR纠错模块（保持不变）
        ir_correction=dict(
            type='TextGuidedIRCorrectionV4',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            correction_alpha=correction_alpha,
        ),
        
        # RGB-IR fusion（保持不变，必须保留）
        fusion_module=dict(
            type='MultiLevelRGBIRFusion',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            reduction=4,
        ),
        
        # ⭐ RGB Enhancement V2（阶段4）
        rgb_enhancement=dict(
            type='TextGuidedRGBEnhancementV2',
            rgb_channels=rgb_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            d_k=d_k,
        ),
        
        # ⭐ Text Update V2（阶段5）
        text_update=dict(
            type='MultiScaleTextUpdateV2',
            in_channels=rgb_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            hidden_dim=text_hidden_dim,
            scale_init=text_scale_init,
            fusion_method=text_fusion_method,
        ),
        
        with_text_model=True,
        frozen_stages=-1,
    ),
    
    # Neck: 可能不需要，但保留以防万一
    neck=dict(
        _delete_=True,
        type='SimpleChannelAlign',
        in_channels=rgb_out_channels,
        out_channels=rgb_out_channels,
    ),
    
    # ⭐ 新增：类别维度聚合器（支持与Fused特征融合）
    # 将 [B, num_cls, C, H, W] 聚合为 [B, C, H, W]，并与Fused特征融合
    # 方案1: Concat + Conv 融合，保留完整视觉信息
    aggregator=dict(
        type='ClassDimensionAggregator',
        in_channels=rgb_out_channels,  # [128, 256, 512]
        num_classes=num_training_classes,  # 4
        aggregation_method='conv',  # 使用1x1卷积聚合类别维度
        fusion_type='concat',  # ⭐ 新增：使用Concat+Conv与Fused特征融合
    ),
    
    # ⭐ Head: 使用传统的YOLOWorldHeadModule（不带V2后缀）
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',  # 传统head，接收4D输入
            embed_dims=text_channels,
            num_classes=num_training_classes,
            use_bn_head=use_bn_head,
            use_einsum=use_einsum,
        )
    ),
    
    train_cfg=dict(
        assigner=dict(num_classes=num_training_classes)
    ),
)

# ======================== Data Settings（完全保持不变）========================
data_root = '/home/ssd1/users/wangzhen01/YOLO-World-master_2/data/flir/root/autodl-tmp/data/FLIR_V1_aligned/align/'
train_ann_file = 'annotations_fixed/train_fixed.json'
val_ann_file = 'annotations_fixed/val_fixed.json'
train_data_prefix = 'JPEGImages/'
val_data_prefix = 'JPEGImages/'

# Pre-transform
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFile',
         ir_suffix='_PreviewData.jpeg',
         rgb_suffix='_RGB.jpg'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# Text transform
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

# Training pipeline
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

# Training pipeline stage 2
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

# Test/Validation pipeline
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

# ======================== Dataset Configuration（保持不变）========================
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

# ======================== Evaluator（保持不变）========================
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
         switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)]
)

# ======================== 优化器配置（核心改进）========================
# ⚠️ 改进1: 添加显式的梯度裁剪
# ⚠️ 改进2: 确保优化器稳定性
optim_wrapper = dict(
    _delete_=True,  # 完全覆盖base配置
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0, norm_type=2),  # ✅ 增强梯度裁剪
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # 标准Adam参数
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'backbone.ir_model': dict(lr_mult=1.0),
            'backbone.ir_correction': dict(lr_mult=1.0),
            'backbone.rgb_enhancement': dict(lr_mult=1.0),  # V2模块
            'backbone.text_update': dict(lr_mult=1.0),      # V2模块
            'logit_scale': dict(weight_decay=0.0),
            'bias': dict(weight_decay=0.0)
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)

# ======================== 学习率调度（核心改进）========================
# ⚠️ 改进: 延长warmup期到10个epoch，使学习率增长更平滑
# 
# 原问题：5个epoch的warmup太快，导致前期振荡
# - Epoch 1: lr = 7e-5  → 3e-4
# - Epoch 2: lr = 4e-4  → 6e-4  (大幅跳升导致性能下降)
# - Epoch 3: lr = 7e-4  → 1e-3
# - Epoch 4: lr = 1.1e-3 → 1.3e-3 (大幅跳升导致性能下降)
# - Epoch 5: lr = 1.45e-3 → 1.5e-3
#
# 解决方案：10个epoch的warmup，每个epoch学习率增长更小
# - 从 lr * 0.01 (1.5e-5) 平滑增长到 lr (1.5e-3)
# - 每个epoch增长约10%，而非原来的50%+

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.01,  # ✅ 从 lr*0.01 = 1.5e-5 开始（比原来1e-5/1.5e-3更平滑）
#         by_epoch=True,
#         begin=100,
#         end=150,  # ✅ 延长到10个epoch（原来是5个epoch）
#     ),
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.05,  # 最小学习率 = 7.5e-5
#         begin=150,
#         end=max_epochs,
#         by_epoch=True,
#     )
# ]

# ======================== Visualizer ========================
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

find_unused_parameters = True

