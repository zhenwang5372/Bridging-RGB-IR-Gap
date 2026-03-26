# 超参数调优指南

## 🎯 当前问题诊断

### 训练状态（Epoch 29）
- **mAP_50**: 0.236 ❌ (目标: >0.5)
- **car**: 0.739 ✅
- **person**: 0.168 ❌
- **bicycle**: 0.035 ❌
- **dog**: 0.001 ❌

### 主要问题
1. ✅ 模型能学习（car效果还可以）
2. ❌ 学习率可能过高（新模块震荡）
3. ❌ 小类别学习不足（样本不平衡）
4. ❌ Warmup不够（新模块初始化不稳定）

---

## 🔧 优化方案对比

### 方案1: 降低学习率（推荐优先尝试）

**配置文件**: `yolow_v2_rgb_ir_flir_simplified_tuned.py`

**关键修改**:
```python
base_lr = 1e-3  # 从2e-3降到1e-3（降低50%）

paramwise_cfg=dict(
    custom_keys={
        'neck.rgb_enhance': dict(lr_mult=0.1),   # 新模块降低10倍
        'neck.text_update': dict(lr_mult=0.1),   # 新模块降低10倍
        'gamma': dict(lr_mult=0.01),             # gamma参数特殊处理
    }
)

# 增加warmup
warmup_epochs = 5  # 从默认的3增加到5
```

**预期效果**:
- mAP_50: 0.30-0.35 (提升30-50%)
- 训练更稳定，loss下降更平滑

---

### 方案2: 调整数据增强

**问题**: 当前增强可能过强，导致小类别难以学习

**修改**:
```python
# 降低Mosaic概率
dict(type='SyncMosaic', prob=0.8)  # 从1.0降到0.8

# 降低光度变换强度
dict(type='DualModalityPhotometricDistortion',
     brightness_delta=20,  # 从32降到20
     contrast_range=(0.7, 1.3))  # 从(0.5, 1.5)缩小

# 降低热成像增强
dict(type='ThermalSpecificAugmentation',
     prob=0.2)  # 从0.3降到0.2
```

---

### 方案3: 类别平衡（如果方案1效果不好）

**问题**: 类别样本不平衡

| 类别 | 样本占比（估计） |
|------|----------------|
| car | ~70% |
| person | ~20% |
| bicycle | ~8% |
| dog | ~2% |

**解决方案**:

#### 3.1 Focal Loss权重
```python
bbox_head=dict(
    loss_cls=dict(
        type='mmdet.FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    )
)
```

#### 3.2 类别采样权重
```python
train_dataloader = dict(
    sampler=dict(
        type='ClassBalancedSampler',  # 类别平衡采样
        oversample_thr=0.5
    )
)
```

---

## 📊 实验建议

### 阶段1: 学习率调优（最优先）

```bash
# 实验1: 降低学习率（推荐）
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_simplified_tuned.py \
    --work-dir work_dirs/simplified_lr1e-3

# 实验2: 更激进的降低（如果实验1还不够）
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_simplified_tuned.py \
    --work-dir work_dirs/simplified_lr5e-4 \
    --cfg-options base_lr=5e-4
```

**判断标准**:
- ✅ loss平滑下降，无震荡
- ✅ mAP持续增长
- ✅ grad_norm < 50

---

### 阶段2: 数据增强调优

如果阶段1效果好但仍不够：

```bash
# 使用tuned配置（已包含降低的增强强度）
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_simplified_tuned.py \
    --work-dir work_dirs/simplified_tuned_aug
```

---

### 阶段3: 类别平衡（最后手段）

如果小类别始终学不好：

```bash
# 需要修改配置添加Focal Loss或类别采样
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_simplified_focal.py \
    --work-dir work_dirs/simplified_focal_loss
```

---

## 🔍 训练监控指标

### 关键指标

| 指标 | 健康范围 | 当前值 | 状态 |
|------|---------|--------|------|
| **grad_norm** | 10-50 | 31.4 | ✅ 正常 |
| **loss** | 持续下降 | 74.5 | ⚠️ 需观察 |
| **lr** | 逐渐衰减 | 1.8e-3 | ❌ 偏高 |
| **mAP_50** | >0.3@30epoch | 0.236 | ❌ 偏低 |

### 异常信号

❌ **学习率过高**:
- grad_norm > 100
- loss震荡
- mAP不增长或下降

❌ **学习率过低**:
- loss下降极慢
- 训练100 epoch后mAP < 0.2

✅ **理想状态**:
- loss平滑下降
- grad_norm稳定在20-40
- mAP持续增长

---

## 💡 快速诊断

### 从训练日志判断

```python
# Epoch 30的日志
grad_norm: 31.3904  # ✅ 正常
loss: 74.4828       # ⚠️ 偏高（应该<70）
lr: 1.8152e-03      # ❌ 太高（建议<1e-3）
```

**结论**: 学习率是主要问题！

---

## 🚀 推荐行动

### 立即执行

```bash
# 停止当前训练（如果还在跑）
# Ctrl+C

# 使用优化配置重新训练
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_simplified_tuned.py \
    --work-dir work_dirs/simplified_tuned_lr1e-3
```

### 预期改进

| 指标 | 当前 | 预期（50 epoch） |
|------|------|-----------------|
| mAP_50 | 0.236 | **0.35-0.40** |
| car | 0.739 | **0.80-0.85** |
| person | 0.168 | **0.25-0.30** |
| bicycle | 0.035 | **0.08-0.12** |

如果50个epoch后还是<0.3，再考虑方案2和方案3。

