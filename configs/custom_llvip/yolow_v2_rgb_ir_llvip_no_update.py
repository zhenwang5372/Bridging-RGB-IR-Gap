# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR No-Update Baseline Configuration for LLVIP Dataset
#
# ============================================================================
# 配置文件说明 (Configuration Description):
# ============================================================================
# 本配置文件用于在 LLVIP 数据集上训练 RGB-IR 双模态目标检测模型。
# 这是一个 **Baseline 配置**，不包含 IR Correction 模块。
#
# 使用的核心模块:
#   - Dataset:           LLVIPDataset (yolo_world/datasets/llvip_dataset.py)
#   - IR Image Loader:   LoadIRImageFromFileLLVIP (yolo_world/datasets/transformers/llvip_transforms.py)
#   - Data Preprocessor: DualModalDataPreprocessor (通用归一化参数)
#   - RGB Backbone:      YOLOv8CSPDarknet
#   - IR Backbone:       LiteFFTIRBackbone (base_channels=32, 输出: [64, 128, 256])
#   - Fusion:            MultiLevelRGBIRFusion (Backbone 级别融合)
#   - Text Model:        HuggingCLIPLanguageBackbone
#   - Neck:              SimpleChannelAlign (只做通道对齐，无 IR Correction)
#   - Head:              YOLOWorldHead
#
# 与 Text Correction V4 配置的区别:
#   - 无 TextGuidedIRCorrectionV4 模块
#   - 使用 DualStreamMultiModalYOLOBackbone (非 V4 版本)
#   - 更简单的模型结构，用于 Baseline 对比
#
# 数据流程:
#   RGB Image ──→ YOLOv8CSPDarknet ──→ [P3, P4, P5] ─┐
#                                                     ├─→ MultiLevelRGBIRFusion ──→ SimpleChannelAlign ──→ YOLOWorldHead
#   IR Image ───→ LiteFFTIRBackbone ──→ [P3, P4, P5] ─┘
#   Text ───────→ CLIP Encoder ──────────────────────────────────────────────────────────────────────────→ (cls contrast)
#
# 数据集信息:
#   - 数据集: LLVIP (Low-Light Visible-Infrared Pair)
#   - 原始尺寸: 1280x1024 (RGB 和 IR 均为 3 通道)
#   - 训练尺寸: 1280x1280 (长边保持，短边 padding)
#   - 类别数: 1 (person)
# ============================================================================

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')

# load_from = 'checkpoints/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'

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
rgb_out_channels = [128, 256, 512]
ir_out_channels = [64, 128, 256]

model = dict(
    type='DualStreamYOLOWorldDetector',
    mm_neck=False,  # 不使用多模态Neck
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='DualModalDataPreprocessor',  # 使用通用预处理器
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        mean_ir=[0., 0., 0.],
        std_ir=[255., 255., 255.],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _delete_=True,
        type='DualStreamMultiModalYOLOBackbone',  # 无 Correction 版本
        # RGB backbone (YOLOv8s)
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        # IR backbone
        ir_model=dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        # RGB-IR fusion (在Backbone中完成融合)
        fusion_module=dict(
            type='MultiLevelRGBIRFusionV3',
            rgb_channels=rgb_out_channels,  # [128, 256, 512]
            ir_channels=ir_out_channels,    # [64, 128, 256]
            reduction=4,
            fusion_type='senet',
            output_type='fused_only',
        ),
        # Text model (CLIP)
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        with_text_model=True,
        frozen_stages=-1,
    ),
    # ⭐ No-Update Neck: 只做通道对齐，无 IR Correction
    neck=dict(
        _delete_=True,
        type='SimpleChannelAlign',
        in_channels=[128, 256, 512],   # Fusion输出: [128, 256, 512]
        out_channels=[128, 256, 512],
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

# Pre-transform: 使用 LLVIP 专用的 IR 加载器
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
    dict(type='DualModalityPhotometricDistortion', prob=0.3),
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
        type='LLVIPDataset',  # LLVIP 专用数据集
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
        type='LLVIPDataset',  # LLVIP 专用数据集
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
