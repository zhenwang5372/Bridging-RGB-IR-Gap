# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR with Class-Specific Features for FLIR Dataset V2

# 核心改进（V2版本）：
# - Backbone: 双流RGB-IR + Text-guided IR Correction（保留）
# - Fusion: MultiLevelRGBIRFusion（保留，必须使用）
# - 阶段4: Text-guided RGB Enhancement V2（改进）
#   - Q=Text, K=Fused, V=Fused（标准QKV Attention）
#   - 用Fused features作为Value，包含完整RGB+IR信息
# - 阶段5: Multi-scale Text Update V2（改进）
#   - 完全独立的Cross-Attention
#   - 直接从Fused提取，不依赖阶段4
# - Head: YOLO-World Head V2（改进）
#   - Max Pooling聚合类别维度
#   - Region-Text相似度分类（YOLO-World风格）
#   - 标准YOLO回归

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
base_lr = 1.5e-3  # 恢复阶段1的学习率
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
img_scale = (640, 640)

# IR Correction参数（保留）
correction_alpha = 0.3
temperature = 0.07  # 恢复阶段1的temperature

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
            type='TextGuidedIRCorrection',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            correction_alpha=correction_alpha,
            temperature=temperature,
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
    
    # ⭐ 新增：类别维度聚合器
    # 将 [B, num_cls, C, H, W] 聚合为 [B, C, H, W]，以便使用传统YOLO Head
    aggregator=dict(
        type='ClassDimensionAggregator',
        in_channels=rgb_out_channels,  # [256, 512, 512]
        num_classes=num_training_classes,  # 4
        aggregation_method='conv',  # 使用1x1卷积聚合
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

# ======================== Training Settings（保持不变）========================
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
            'backbone.ir_correction': dict(lr_mult=1.0),
            'backbone.rgb_enhancement': dict(lr_mult=1.0),  # V2模块
            'backbone.text_update': dict(lr_mult=1.0),      # V2模块
            'logit_scale': dict(weight_decay=0.0),
            'bias': dict(weight_decay=0.0)
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

