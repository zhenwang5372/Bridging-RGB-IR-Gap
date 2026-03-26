# Copyright (c) Tencent Inc. All rights reserved.
# YOLO-World v2 RGB-IR DroneVehicle - Remove White Borders Version
#
# ============================================================================
# 数据集说明:
# ============================================================================
# DroneVehicle 数据集 - 无人机视角车辆检测 (去除白边版本)
# - 图像分辨率: 640×512 (裁切后，已去除白边) → 640×640 (训练，带灰边padding)
# - 类别数: 5 (bus, car, freight_car, truck, van)
# - 类别ID映射 (与with_white_borders版本不同):
#     ID=1: car, ID=2: freight_car, ID=3: truck, ID=4: bus, ID=5: van
# - RGB/IR 图像通过文件夹区分: trainimg/ vs trainimgr/
# - 图片路径: images/ (640x512裁切后的图片)
# - 标注路径: annotations/remove_white_borders/ (四个版本可选)
# - 测试集配置: 复用验证集
# ============================================================================

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
load_from = 'checkpoints/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

# ======================== Hyper-parameters ========================
num_classes = 5  # DroneVehicle: bus, car, freight_car, truck, van
num_training_classes = 5
max_epochs = 300
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 512
base_lr = 1.5e-3
# base_lr = 3e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
img_scale = (640, 640)  # 裁切后640×512 → 640×640 (带padding)

# RGB Enhancement V2参数
d_k = 128  # Attention的key维度

# Text Update V2参数
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
        type='FLIRDataPreprocessor',
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

        ir_correction=dict(
            type='IR_RGB_Merr_Cons',
            rgb_channels=rgb_out_channels,
            ir_channels=ir_out_channels,
            text_dim=text_channels,
            num_classes=num_classes,
            correction_alpha=-0.5,
            enhancement_beta=0.5,
            d_k=128,
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
        
        # Text Update V4
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
    
    # Aggregator
    aggregator=dict(
        type='ClassDimensionAggregator',
        in_channels=rgb_out_channels,
        num_classes=num_training_classes,
        aggregation_method='conv',
        fusion_type='concat',
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
data_root = '/home/ssd1/users/wangzhen01/YOLO-World-master_2/data/dronevehicle/'

# ============================================================================
# 标注文件选择 (四个版本可选，默认使用remove_clip_less_5px版本)
# ============================================================================
# 版本1: remove_clip_less_5px - Clip后移除<5px的bbox（推荐）⭐
#   - 负值clip到0，超界clip到边界，再移除宽或高<5px的极小bbox
#   - 最干净的版本，适合训练
#   - Train: 315,660个标注, Val: 24,489个标注, Test: 159,610个标注
train_ann_file = 'annotations/remove_white_borders/remove_clip_less_5px/DV_train_r_clip_less5.json'
val_ann_file = 'annotations/remove_white_borders/remove_clip_less_5px/DV_val_r_clip_less5.json'
test_ann_file = 'annotations/remove_white_borders/remove_clip_less_5px/DV_test_r_clip_less5.json'

# 版本2: clip - Clip版本（包含<5px的bbox）
#   - 负值clip到0，超界clip到边界
#   - 包含所有clip后有效的bbox（含<5px的极小bbox）
#   - Train: 315,665个标注, Val: 24,490个标注, Test: 159,614个标注
# train_ann_file = 'annotations/remove_white_borders/clip/DV_train_r_clip.json'
# val_ann_file = 'annotations/remove_white_borders/clip/DV_val_r_clip.json'
# test_ann_file = 'annotations/remove_white_borders/clip/DV_test_r_clip.json'

# 版本3: minus - 原始版本（可能有负值和宽高=0的bbox）
#   - 原始DV_*_r.json文件，bbox可能有负值或宽高=0
#   - Train: 315,666个标注, Val: 24,490个标注, Test: 159,616个标注
# train_ann_file = 'annotations/remove_white_borders/minus/DV_train_r.json'
# val_ann_file = 'annotations/remove_white_borders/minus/DV_val_r.json'
# test_ann_file = 'annotations/remove_white_borders/minus/DV_test_r.json'

# 版本4: remove_beyond_borders - 删除超界版本
#   - 完全删除超出边界的bbox（损失约14%标注）
#   - Train: 271,262个标注, Val: 21,120个标注, Test: 137,450个标注
# train_ann_file = 'annotations/remove_white_borders/remove_beyond_borders/DV_train_r_no_beyond.json'
# val_ann_file = 'annotations/remove_white_borders/remove_beyond_borders/DV_val_r_no_beyond.json'
# test_ann_file = 'annotations/remove_white_borders/remove_beyond_borders/DV_test_r_no_beyond.json'

# 图片路径 (640x512裁切后的图片，所有版本共用)
train_data_prefix = 'images/train/trainimg/'
train_ir_prefix = 'images/train/trainimgr/'
val_data_prefix = 'images/val/valimg/'
val_ir_prefix = 'images/val/valimgr/'
test_data_prefix = 'images/test/testimg/'
test_ir_prefix = 'images/test/testimgr/'

# Pre-transform
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFileDroneVehicle',
         rgb_dir='trainimg',
         ir_dir='trainimgr'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# Pre-transform for validation (different directory names)
pre_transform_val = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFileDroneVehicle',
         rgb_dir='valimg',
         ir_dir='valimgr'),
    dict(type='LoadAnnotations', with_bbox=True),
]

# Pre-transform for test
pre_transform_test = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadIRImageFromFileDroneVehicle',
         rgb_dir='testimg',
         ir_dir='testimgr'),
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
    dict(type='LoadIRImageFromFileDroneVehicle',
         rgb_dir='valimg',
         ir_dir='valimgr'),
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
# 使用 DroneVehicleRWBDataset (Remove White Borders版本)
# category_id顺序: car(1), freight_car(2), truck(3), bus(4), van(5)
train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='DroneVehicleRWBDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        rgb_dir='trainimg',
        ir_dir='trainimgr',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ),
    class_text_path='data/dronevehicle/texts/dronevehicle_class_texts_remove_wb.json',
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='DroneVehicleRWBDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        rgb_dir='valimg',
        ir_dir='valimgr',
        test_mode=True,
    ),
    class_text_path='data/dronevehicle/texts/dronevehicle_class_texts_remove_wb.json',
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

# 测试集复用验证集
test_dataloader = val_dataloader

# ======================== Evaluator ========================
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox',
    classwise=True,
)
# 测试评估器复用验证评估器
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
            'backbone.text_update': dict(lr_mult=1.0),
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

# 关闭确定性算法
randomness = dict(seed=None, deterministic=False)
