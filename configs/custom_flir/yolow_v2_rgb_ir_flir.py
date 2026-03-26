# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR Dual-Modal Configuration for FLIR Dataset

# from tkinter.constants import FALSE

# from numpy import False_


_base_ = ('../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
# 加载 YOLO-World v2 S 预训练权重
# 匹配的部分: backbone.image_model, backbone.text_model, neck, bbox_head
# 随机初始化: backbone.ir_model, backbone.fusion_module
load_from = 'checkpoints/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

# ======================== Pretrained Weights (Optional) ========================
# Uncomment to load pretrained YOLO-World weights for the RGB backbone
# Note: IR backbone has no pretrained weights and will be trained from scratch
# load_from = 'pretrained_models/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-XXX.pth'

# ======================== Hyper-parameters ========================
num_classes = 4  # FLIR classes: person, car, bicycle, dog
num_training_classes = 4
max_epochs = 300
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 512
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16  # Reduced due to dual-stream memory

# Image settings
img_scale = (640, 640)

# ======================== Model Definition ========================
# YOLOv8s backbone output channels (with widen_factor=0.5)
# P3: 128, P4: 256, P5: 512
rgb_out_channels = [128, 256, 512]

# LiteFFTIRBackbone output channels (with base_channels=32)
# P3: 64, P4: 128, P5: 256
ir_out_channels = [64, 128, 256]

# Neck configuration
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

model = dict(
    type='DualStreamYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='FLIRDataPreprocessor',
        # Standard YOLO normalization: scale to [0, 1] range
        # This is compatible with pretrained weights if loaded
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        # IR uses same normalization for consistency
        mean_ir=[0., 0., 0.],
        std_ir=[255., 255., 255.],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _delete_=True,
        type='DualStreamMultiModalYOLOBackbone',
        # RGB backbone (YOLOv8s CSPDarknet)
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        # IR backbone (lightweight FFT-based)
        ir_model=dict(
            type='LiteFFTIRBackbone',
            in_channels=3,
            base_channels=32,  # Outputs [64, 128, 256]
            # use_se=False,        # ← 开启 SE-Block
            # se_reduction=16,    # ← reduction ratio
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            freq_ratio=0.5,
        ),
        # RGB-IR fusion moduleD
        fusion_module=dict(
            type='MultiLevelRGBIRFusion',
            rgb_channels=rgb_out_channels,  # [128, 256, 512]
            ir_channels=ir_out_channels,    # [64, 128, 256]
            reduction=4,
        ),
        # Text model (CLIP)
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        with_text_model=True,
    ),
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        # Use base channel values; widen_factor=0.5 from _base_ will scale them
        # Expected actual channels: [256*0.5, 512*0.5, 1024*0.5] = [128, 256, 512]
        in_channels=[256, 512, _base_.last_stage_out_channels],
        out_channels=[256, 512, _base_.last_stage_out_channels],
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
# FLIR dataset paths
data_root = '/root/autodl-tmp/data/FLIR_V1_aligned/align/'
train_ann_file = 'annotations_fixed/train_fixed.json'
val_ann_file = 'annotations_fixed/val_fixed.json'
train_data_prefix = 'JPEGImages/'
val_data_prefix = 'JPEGImages/'

# Pre-transform: Load RGB and IR images  
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFile',
         ir_suffix='_PreviewData.jpeg',
         rgb_suffix='_RGB.jpg'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# Text transform for YOLO-World (use PackDualModalInputs for dual-modal)
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDualModalInputs',  # Custom packer for RGB+IR
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                    'flip', 'flip_direction', 'texts', 'scale_factor',
                    'pad_param', 'img_ir_path', 'img_ir_shape'))
]

# Training pipeline with synchronized RGB-IR augmentations
# Follows reference config structure: pre_transform -> Mosaic -> Affine -> last_transforms
train_pipeline = [
    *pre_transform,
    # Synchronized Mosaic (for 4 images with RGB-IR pairs)
    dict(type='SyncMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=pre_transform,
         prob=1.0),
    # Synchronized random affine (border for 2x canvas from mosaic)
    dict(type='SyncRandomAffine',
         max_rotate_degree=0.0,  # Same as reference: no rotation
         max_shear_degree=0.0,   # Same as reference: no shear
         max_translate_ratio=0.1,
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
         border=(-img_scale[0] // 2, -img_scale[1] // 2),  # Crop from 2x canvas
         border_val=(114, 114, 114)),
    # Synchronized letter resize back to target scale
    dict(type='SyncLetterResize',
         scale=img_scale,
         pad_val=dict(img=114, img_ir=114),
         allow_scale_up=True),
    # Synchronized random flip (same as reference: 0.5 prob)
    dict(type='SyncRandomFlip', prob=0.5, direction='horizontal'),
    # Modality-specific photometric distortion
    # RGB: full HSV augmentation, IR: brightness/contrast only
    dict(type='DualModalityPhotometricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
         ir_brightness_delta=20,
         ir_contrast_range=(0.8, 1.2),
         prob=0.5),
    # Thermal-specific augmentation for IR
    dict(type='ThermalSpecificAugmentation',
         fpa_noise_level=0.02,
         crossover_prob=0.1,
         scale_range=(0.95, 1.05),
         shift_range=10,
         prob=0.3),
    # Text transform and packing
    *text_transform,
]

# Training pipeline stage 2 (without mosaic, for close_mosaic_epochs)
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
         border=(0, 0),  # No mosaic, no border expansion
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
    dict(type='PackDualModalInputs',  # Use custom packer for inference too
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts',
                    'img_ir_path', 'img_ir_shape'))
]

# ======================== Dataset Configuration ========================
# Use MultiModalDataset wrapper to provide text information to RandomLoadText
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
    _delete_=True,  # Delete base config to avoid parameter merge
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=train_dataset,
)

val_dataloader = dict(
    _delete_=True,  # Delete base config to avoid parameter merge
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
    classwise=True,  # 打印每个类别的 AP 详细表格
)
test_evaluator = val_evaluator

# ======================== Training Settings ========================
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP_50',  # Use standard COCO metric
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
            'backbone.ir_model': dict(lr_mult=1.0),  # Train IR backbone normally
            'logit_scale': dict(weight_decay=0.0)
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)

# ======================== Runtime Settings ========================
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)
