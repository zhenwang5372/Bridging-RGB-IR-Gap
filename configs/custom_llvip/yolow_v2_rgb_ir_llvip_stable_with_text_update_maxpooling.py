# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR Stable (With Text Update) for LLVIP Dataset
#
# ============================================================================
# 配置说明：启用 Text Update，启用 Aggregator Fusion (concat)
# ============================================================================
#
# 数据流：
#   RGB → Image_Model ──┐
#                       ├─→ IR_Correction → Fusion → RGB_Enhancement → Aggregator → Head
#   IR  → IR_Model   ───┘         ↑                        ↑              ↓
#                                 │                        │        aggregated_feats
#   Text → Text_Model ────────────┴────────────────────────┼──→ Text_Update ───┘
#                                                          │         ↓
#                                                          └── text_updated (更新后)
#
# 与基础版的区别：
#   1. ✅ 启用 Text Update 模块 - 文本特征被视觉信息更新
#   2. ✅ Aggregator 与 fused_feats 融合 (fusion_type='concat')
#
# 设计理念：
#   文本语义不仅指导视觉处理，还能从视觉信息中学习并更新
#
# 使用了yoloworld类似的I-pooling,没有跨batch聚合，没有使用置信度门控
#
# 数据集信息:
#   - 数据集: LLVIP (Low-Light Visible-Infrared Pair)
#   - 原始尺寸: 1280x1024 (RGB 和 IR 均为 3 通道)
#   - 训练尺寸: 1280x1280 (长边保持，短边 padding)
#   - 类别数: 1 (person)
#   - 数据结构:
#       data/LLVIP/visible/train/*.jpg      (RGB)
#       data/LLVIP/visible/test/*.jpg       (RGB)
#       data/LLVIP/infrared/train/*.jpg     (IR)
#       data/LLVIP/infrared/test/*.jpg      (IR)
#       data/LLVIP/coco_annotations/train.json
#       data/LLVIP/coco_annotations/test.json
#
# 与 FLIR 配置的主要区别:
#   - 数据集类: LLVIPDataset (目录替换) vs FLIRDataset (后缀替换)
#   - IR Loader: LoadIRImageFromFileLLVIP vs LoadIRImageFromFile
#   - 类别数: 1 vs 4
# ============================================================================

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
load_from = 'checkpoints/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

# ======================== Hyper-parameters ========================
num_classes = 1  # LLVIP: only person
num_training_classes = 1
max_epochs = 300
close_mosaic_epochs = 5  # 与其他LLVIP配置一致
save_epoch_intervals = 1
text_channels = 512
base_lr = 2e-4  # 与其他LLVIP配置一致 (较低学习率适合更大图像)
weight_decay = 0.1  # 与其他LLVIP配置一致
train_batch_size_per_gpu = 8  # 降低 batch_size 适应更大的图像尺寸
img_scale = (1280, 1280)  # LLVIP 使用 1280x1280 (长边 padding)

# IR Correction参数
correction_alpha = 0.3
temperature = 0.07

# RGB Enhancement V2参数
d_k = 128  # Attention的key维度

# ⭐ Text Update V2参数
text_hidden_dim = 256  # Cross-Attention隐藏维度
text_scale_init = 0.0  # 残差缩放初始值
text_fusion_method = 'learned_weight'  # 可选: 'learned_weight', 'equal'

# Head参数
use_bn_head = True
use_einsum = True

# ======================== Model Definition ========================
rgb_out_channels = [128, 256, 512]
ir_out_channels = [64, 128, 256]

model = dict(
    type='DualStreamYOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='DualModalDataPreprocessor',  # 使用通用预处理器 (适用于LLVIP)
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        mean_ir=[0., 0., 0.],
        std_ir=[255., 255., 255.],
        bgr_to_rgb=True,
    ),
    
    backbone=dict(
        _delete_=True,
        type='DualStreamMultiModalYOLOBackboneWithClassSpecificV2',
        
        # RGB backbone
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=1024,
            deepen_factor=0.33,
            widen_factor=0.5,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
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
        
        # Text model
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        ),
        
        # IR纠错模块
        ir_correction=dict(
            type='TextGuidedIRCorrectionV4',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            correction_alpha=correction_alpha,
        ),
        
        # RGB-IR fusion
        fusion_module=dict(
            type='MultiLevelRGBIRFusion',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            reduction=4,
        ),
        
        # RGB Enhancement V2
        rgb_enhancement=dict(
            type='TextGuidedRGBEnhancementV2',
            rgb_channels=rgb_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            d_k=d_k,
        ),
        
        # ⭐ Text Update V3 - 启用（带置信度门控）
        text_update=dict(
            type='MultiScaleTextUpdateV4',
            in_channels=rgb_out_channels,
            text_dim=text_channels,
            embed_channels=256,
            num_heads=8,
            with_scale=True,
    
        ),
        
        with_text_model=True,
        frozen_stages=-1,
    ),
    
    # Neck
    neck=dict(
        _delete_=True,
        type='SimpleChannelAlign',
        in_channels=rgb_out_channels,
        out_channels=rgb_out_channels,
    ),
    
    # Aggregator - 不融合 fused_feats
    aggregator=dict(
        type='ClassDimensionAggregator',
        in_channels=rgb_out_channels,
        num_classes=num_training_classes,
        aggregation_method='conv',
        fusion_type='concat',  # 与 fused_feats 融合
    ),
    
    # Head
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
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

# ======================== 优化器配置 ========================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0, norm_type=2),
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'backbone.ir_model': dict(lr_mult=1.0),
            'backbone.ir_correction': dict(lr_mult=1.0),
            'backbone.rgb_enhancement': dict(lr_mult=1.0),
            'backbone.text_update': dict(lr_mult=1.0),  # Text Update 学习率
            'logit_scale': dict(weight_decay=0.0),
            'bias': dict(weight_decay=0.0)
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)

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

# ⭐ 关闭确定性算法（AdaptivePool 的 CUDA 反向传播不支持确定性）
randomness = dict(seed=None, deterministic=False)
