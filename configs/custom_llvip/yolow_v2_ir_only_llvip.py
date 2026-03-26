# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 IR-Only Baseline Configuration for LLVIP Dataset
#
# ============================================================================
# 配置文件说明 (Configuration Description):
# ============================================================================
# 本配置文件用于在 LLVIP 数据集上训练 **IR-Only** 单模态目标检测模型。
# 这是一个 **Baseline 配置**，用于与 RGB-IR 双模态检测器进行对比实验。
#
# 使用的核心模块:
#   - Detector:          IROnlyYOLOWorldDetector
#   - Backbone:          IROnlyBackbone
#   - IR Backbone:       LiteFFTIRBackbone (base_channels=32, 输出: [64, 128, 256])
#   - Channel Align:     Conv1x1 + BN + SiLU ([64,128,256] → [128,256,512])
#   - Text Model:        HuggingCLIPLanguageBackbone
#   - Neck:              SimpleChannelAlign (无实际操作，只做 pass-through)
#   - Head:              YOLOWorldHead
#
# 与 RGB-IR 双模态配置的区别:
#   - 不使用 RGB Backbone (YOLOv8CSPDarknet)
#   - 不使用 RGB-IR Fusion Module
#   - 只使用 IR 图像作为视觉输入
#   - 在 Backbone 内部添加 Channel Align 层对齐到 Head 需要的通道数
#
# 数据流程:
#   IR Image (1280x1280) ──→ LiteFFTIRBackbone ──→ [64, 128, 256] (空间尺寸由 IR backbone 决定)
#                                                        │
#                                                        ▼
#                                           Channel Align (Conv1x1 + BN + SiLU)
#                                           + Spatial Interpolate (bilinear)
#                                                        │
#                                                        ▼
#                                           [128, 256, 512] @ [160x160, 80x80, 40x40]
#                                           (对齐到 YOLOv8 的标准下采样率 8x, 16x, 32x)
#                                                        │
#                                                        ▼
#                                           SimpleChannelAlign (pass-through) ──→ YOLOWorldHead
#   
#   Text ─────→ CLIP Encoder ──────────────────────────────────────────────────→ (cls contrast)
#
# 数据集信息:
#   - 数据集: LLVIP (Low-Light Visible-Infrared Pair)
#   - 原始尺寸: 1280x1024 (IR 为 3 通道)
#   - 训练尺寸: 1280x1280 (长边保持，短边 padding)
#   - 类别数: 1 (person)
# ============================================================================

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')

custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

# ======================== Hyper-parameters ========================
num_classes = 1  # LLVIP: only person
num_training_classes = 1
max_epochs = 300
close_mosaic_epochs = 5
save_epoch_intervals = 1
text_channels = 512
base_lr = 2e-4
weight_decay = 0.1
train_batch_size_per_gpu = 8

img_scale = (1280, 1280)  # LLVIP 使用 1280x1280 (长边 padding)

# ======================== Model Definition ========================
# IR backbone 输出通道 (LiteFFTIRBackbone with base_channels=32)
ir_out_channels = [64, 128, 256]
# 对齐后的输出通道 (与标准 YOLOv8s 一致)
aligned_out_channels = [128, 256, 512]

model = dict(
    type='IROnlyYOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='DualModalDataPreprocessor',  # 复用双模态预处理器 (但只使用 IR)
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        mean_ir=[0., 0., 0.],
        std_ir=[255., 255., 255.],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _delete_=True,
        type='IROnlyBackbone',
        # IR backbone
        ir_model=dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        # IR channels → Aligned channels
        ir_channels=ir_out_channels,      # [64, 128, 256]
        out_channels=aligned_out_channels,  # [128, 256, 512]
        # Text model (CLIP)
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        with_text_model=True,
        frozen_stages=-1,
    ),
    # Neck: 通道对齐 (实际上 IROnlyBackbone 已经对齐了，这里只是 pass-through)
    neck=dict(
        _delete_=True,
        type='SimpleChannelAlign',
        in_channels=aligned_out_channels,   # [128, 256, 512]
        out_channels=aligned_out_channels,  # [128, 256, 512]
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
    )
)

# ======================== Data Settings ========================
data_root = 'data/LLVIP/'
train_ann_file = 'coco_annotations/train.json'
val_ann_file = 'coco_annotations/test.json'
train_data_prefix = 'visible/train/'
val_data_prefix = 'visible/test/'

# Pre-transform: 复用 LLVIP 的数据加载 (虽然加载了 RGB，但模型不使用)
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFileLLVIP',  # LLVIP 专用
         rgb_dir='visible',
         ir_dir='infrared'),
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
    # IR-Only: 只对 IR 做增强
    dict(type='ThermalSpecificAugmentation',
         fpa_noise_level=0.02,
         crossover_prob=0.1,
         scale_range=(0.95, 1.05),
         shift_range=10,
         prob=0.3),
    *text_transform,
]

# Training pipeline stage 2 (close mosaic)
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
    *text_transform,
]

# Test/Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFileLLVIP',  # LLVIP 专用
         rgb_dir='visible',
         ir_dir='infrared'),
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
        type='LLVIPDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        rgb_dir='visible',
        ir_dir='infrared',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path='data/llvip/texts/llvip_class_texts.json',
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='LLVIPDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        rgb_dir='visible',
        ir_dir='infrared',
        test_mode=True,
    ),
    class_text_path='data/llvip/texts/llvip_class_texts.json',
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
         switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)]
)

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
