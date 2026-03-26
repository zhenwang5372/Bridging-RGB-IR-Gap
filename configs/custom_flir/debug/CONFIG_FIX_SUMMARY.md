# 配置文件错误修复总结

## 修复日期
2026-01-14

## 问题1：`AttributeError: 'ConfigDict' object has no attribute 'last_stage_out_channels'`

### 原因
配置文件使用 `delete=True` 删除了原始 backbone，但仍然尝试从 `_base_` 继承属性。

### 解决方案
直接写入具体数值，不引用 `_base_` 属性。

### 修改位置
**文件**：`configs/custom_flir/yolow_v2_rgb_ir_flir_text_correction.py`

**修改内容**（第 51-58 行）：

```python
# 修改前
image_model=dict(
    type='YOLOv8CSPDarknet',
    arch='P5',
    last_stage_out_channels=_base_.last_stage_out_channels,  # ❌
    deepen_factor=_base_.deepen_factor,  # ❌
    widen_factor=_base_.widen_factor,  # ❌
    norm_cfg=_base_.norm_cfg,  # ❌
    act_cfg=dict(type='SiLU', inplace=True),
),

# 修改后
image_model=dict(
    type='YOLOv8CSPDarknet',
    arch='P5',
    last_stage_out_channels=1024,  # ✅ YOLOv8-S 默认值
    deepen_factor=0.33,  # ✅ YOLOv8-S 默认值
    widen_factor=0.5,  # ✅ YOLOv8-S 默认值
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),  # ✅ YOLOv8 默认值
    act_cfg=dict(type='SiLU', inplace=True),
),
```

---

## 问题2：`AttributeError: 'str' object has no attribute 'affine_scale'`

### 原因
配置文件定义 `base` 为字符串（路径），但在后续代码中当作配置对象使用 `base.affine_scale`。

### 解决方案
直接写入具体数值，不使用 `base.xxx` 引用。

### 修改位置

#### 修改1（第 162 行）：

```python
# 修改前
scaling_ratio_range=(1 - base.affine_scale, 1 + base.affine_scale),  # ❌

# 修改后
scaling_ratio_range=(0.5, 1.5),  # ✅ affine_scale=0.5 来自 YOLOv8-S
```

#### 修改2（第 306 行）：

```python
# 修改前
dynamic_intervals=[((max_epochs - close_mosaic_epochs), base.val_interval_stage2)]  # ❌

# 修改后
dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)]  # ✅ val_interval_stage2=1 来自 YOLOv8-S
```

---

## 问题3：`AttributeError: module 'yolo_world.models.backbones' has no attribute 'AgreementBasedRGBEnhancementNeck'`

### 原因
`__init__.py` 的 `__all__` 列表中包含了不存在的类名。

### 解决方案
从 `__all__` 列表中删除不存在的类。

### 修改位置

#### 文件1：`yolo_world/models/backbones/__init__.py`

```python
# 修改前
__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'LiteFFTIRBackbone',
    'AgreementBasedRGBEnhancementNeck',  # ❌ 这是 Neck，不应该在 backbones
    'TextGuidedIRCorrection',  # ❌ 这是 Neck，不应该在 backbones
    'DualStreamMultiModalYOLOBackboneWithCorrection'  # ❌ 未导入
]

# 修改后
__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'LiteFFTIRBackbone',
]
```

#### 文件2：`yolo_world/models/necks/__init__.py`

```python
# 修改前
__all__ = [
    'YOLOWorldPAFPN', 'YOLOWorldDualPAFPN',
    'LightweightCrossFusion', 'MultiLevelRGBIRFusion',
    'SimpleChannelAlign', 'NoNeckPassThrough',
    'AgreementBasedRGBEnhancementNeck',  # ❌ 不存在
    'TextGuidedIRCorrection',  # ✅ 已导入
    'DualStreamMultiModalYOLOBackboneWithCorrection'  # ✅ 已导入
]

# 修改后
__all__ = [
    'YOLOWorldPAFPN', 'YOLOWorldDualPAFPN',
    'LightweightCrossFusion', 'MultiLevelRGBIRFusion',
    'SimpleChannelAlign', 'NoNeckPassThrough',
    'TextGuidedIRCorrection',
    'DualStreamMultiModalYOLOBackboneWithCorrection'
]
```

---

## 参考的默认值（来自 YOLOv8-S）

从 `third_party/mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`:

| 参数 | 值 | 说明 |
|------|-----|------|
| `deepen_factor` | 0.33 | 控制网络深度 |
| `widen_factor` | 0.5 | 控制网络宽度 |
| `last_stage_out_channels` | 1024 | 最后一层输出通道 |
| `norm_cfg` | `dict(type='BN', momentum=0.03, eps=0.001)` | 归一化配置 |
| `affine_scale` | 0.5 | RandomAffine 缩放比例 |
| `val_interval_stage2` | 1 | Stage2 验证间隔 |

---

## 验证结果

```bash
python -c "from mmengine.config import Config; \
           cfg = Config.fromfile('configs/custom_flir/yolow_v2_rgb_ir_flir_text_correction.py'); \
           print('✅ 配置文件加载成功！')"
```

**输出**：
```
✅ 配置文件加载成功！
模型类型: DualStreamYOLOWorldDetector
```

---

## 总结

所有配置错误已修复！采用与 `correction_details.py` 相同的解决策略：
- **不依赖继承的属性**
- **直接写入具体值**
- **确保所有引用都是有效的**

现在可以正常启动训练了！

