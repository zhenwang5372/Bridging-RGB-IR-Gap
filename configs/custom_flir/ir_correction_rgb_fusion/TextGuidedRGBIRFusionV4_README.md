# TextGuidedRGBIRFusionV4 接口配置说明

## 概述

V4 版本在 V3 基础上新增了两个重要接口，用于解决 V3 中发现的两个核心问题：
1. **β/γ 参数可能学习到负值**，导致 mask 语义反转
2. **mask 全部为正值**，缺乏抑制背景的能力

---

## V4 新增接口

### 接口1: `param_constraint` - β和γ参数约束方式

**问题背景**：V3 中 β 和 γ 是无约束的可学习参数。训练时可能学习到负值，当 β < 0 时：
- `mask_raw = σ(β·X_ir + γ·S_map)`
- 目标区域 X_ir 高 → β·X_ir 大负数 → sigmoid 输出小
- 背景区域 X_ir 低 → β·X_ir ≈ 0 → sigmoid 输出大

**结果**：mask 在目标区域反而变低，与预期相反。

| 选项 | 说明 | 公式 | 适用场景 |
|------|------|------|----------|
| `'softplus'` ⭐默认 | 使用 softplus 确保 > 0 | `β_pos = log(1 + exp(β))` | 大多数场景 |
| `'abs'` | 使用绝对值确保 > 0 | `β_pos = \|β\|` | 快速测试 |
| `'residual_alpha'` | 不约束，改用残差融合 | `x_fused = x_rgb + α·(x_rgb·mask)` | 让模型自学方向 |
| `'none'` | 不约束（V3行为） | `β_pos = β` | V3 兼容 |

**工作原理图解**：

```
softplus 函数:
                    ↑
                3   |        _____
                    |      /
                2   |    /
                    |  /
                1   | /
                    |/
              0 ────┼──────────────→
               -2  -1   0   1   2   x

特点: 平滑、可导、当 x→-∞ 时趋近于 0+
```

---

### 接口2: `mask_center` - Mask 零中心化方式

**问题背景**：V3 中由于使用 sigmoid，mask 值全部在 (0, 1) 范围内，导致：
- 所有区域都被增强，只是程度不同
- 缺乏"抑制背景"的能力

| 选项 | 说明 | 公式 | 效果 |
|------|------|------|------|
| `'spatial_mean'` ⭐默认 | 减去空间均值 | `mask = mask - mean(mask)` | 高于均值→正→增强，低于均值→负→抑制 |
| `'tanh'` | 用 tanh 代替 sigmoid | 最后激活改为 Tanh | 输出范围 (-1, 1) |
| `'smap_center'` | S_map 阶段零中心化 | `S_map = S_map - mean(S_map)` | 更早引入负值 |
| `'none'` | 不零中心化（V3行为） | mask 保持 (0, 1) | V3 兼容 |

**零中心化效果对比**：

```
V3 (无零中心化):
mask 值:  [0.3, 0.5, 0.7, 0.4]  ← 全正值
效果:     所有区域都增强，只是程度不同

V4 (spatial_mean):
原始 mask: [0.3, 0.5, 0.7, 0.4]
均值:      0.475
零中心化:  [-0.175, 0.025, 0.225, -0.075]  ← 有正有负
效果:      正值区域增强，负值区域抑制
```

---

## 继承自 V3 的接口

### 接口3: `gap_method` - GAP 计算方式

用于计算类别重要性分数。

| 选项 | 说明 | 公式 |
|------|------|------|
| `'logits'` ⭐默认 | 使用注意力 logits 均值 | `gap = mean(attn_logits)` |
| `'max'` | 使用注意力概率最大值 | `gap = max(softmax(attn_logits))` |
| `'entropy'` | 使用注意力熵（反向） | `gap = 1 - entropy/max_entropy` |

---

### 接口4: `smap_method` - S_map 归一化方式

控制 A_rgb 和 A_ir 的归一化方式。

| 选项 | 说明 | 公式 |
|------|------|------|
| `'sigmoid'` | 固定缩放的 sigmoid | `A = σ(logits / √d_k)` |
| `'sigmoid_temp'` | 可学习温度的 sigmoid | `A = σ(logits / τ)`, τ 可学习 |
| `'normalized'` ⭐默认 | 减均值后 sigmoid | `S_map = σ(S_map - mean(S_map))` |

---

### 接口5: `smap_order` - S_map 计算顺序

控制类别加权和 Hadamard 积的顺序。

| 选项 | 说明 | 公式 |
|------|------|------|
| `'sum_first'` ⭐默认 | 先加后乘 | `S_map = (Σw·A_rgb) ⊙ (Σw·A_ir)` |
| `'multiply_first'` | 先乘后加 | `S_map = Σ(w·A_rgb ⊙ w·A_ir)` |

**计算流程对比**：

```
sum_first (先加后乘):
  A_rgb^1, A_rgb^2, ... → 加权求和 → A_rgb_agg
  A_ir^1,  A_ir^2,  ... → 加权求和 → A_ir_agg
                                       ↓
                            S_map = A_rgb_agg ⊙ A_ir_agg

multiply_first (先乘后加):
  A_rgb^1 ⊙ A_ir^1 → hadamard_1
  A_rgb^2 ⊙ A_ir^2 → hadamard_2  → 求和 → S_map
  ...
```

---

### 接口6: `mask_method` - Mask 生成方式

| 选项 | 说明 | 流程 |
|------|------|------|
| `'conv_gen'` ⭐默认 | 卷积生成器 | `σ(β·X_ir + γ·S_map) → Conv → mask` |
| `'residual'` | 残差细化 | `mask_raw + 0.1·tanh(Conv(mask_raw))` |
| `'dual_branch'` | 双分支融合 | `spatial(S_map) × channel(X_ir)` |
| `'se_spatial'` | SE + 空间卷积 | `σ(SE(X_ir)·X_ir + spatial_conv(S_map))` |

---

## 推荐配置组合

### 配置1: V4 默认推荐
```python
param_constraint = 'softplus'
mask_center = 'spatial_mean'
```
适用：大多数场景，解决 V3 的两个核心问题

### 配置2: tanh 方案
```python
param_constraint = 'softplus'
mask_center = 'tanh'
```
适用：需要更大的负值范围 (-1, 1)

### 配置3: 残差补偿方案
```python
param_constraint = 'residual_alpha'
mask_center = 'spatial_mean'
```
适用：让模型自己学习增强/抑制方向

### 配置4: V3 兼容模式
```python
param_constraint = 'none'
mask_center = 'none'
```
适用：与 V3 完全一致的行为

---

## 完整配置参数表

| 参数名 | 类型 | 默认值 | 可选值 | 说明 |
|--------|------|--------|--------|------|
| `rgb_channels` | List[int] | [128,256,512] | - | RGB 特征通道数 |
| `ir_channels` | List[int] | [64,128,256] | - | IR 特征通道数 |
| `text_dim` | int | 512 | - | 文本特征维度 |
| `num_classes` | int | 4 | - | 类别数 |
| `beta` | float | 1.0 | - | X_ir 权重初始值 |
| `gamma` | float | 0.5 | - | S_map 权重初始值 |
| `alpha` | float | 0.1 | - | 残差融合系数 |
| `gap_method` | str | 'logits' | logits/max/entropy | GAP 计算方式 |
| `smap_method` | str | 'sigmoid' | sigmoid/sigmoid_temp/normalized | S_map 归一化 |
| `smap_order` | str | 'sum_first' | sum_first/multiply_first | S_map 计算顺序 |
| `mask_method` | str | 'conv_gen' | conv_gen/residual/dual_branch/se_spatial | Mask 生成方式 |
| `mask_reduction` | int | 8 | - | 通道缩减比例 |
| `temperature` | float | 1.0 | - | 温度参数 |
| `param_constraint` | str | 'softplus' | softplus/abs/residual_alpha/none | ⭐V4: β/γ约束 |
| `mask_center` | str | 'spatial_mean' | spatial_mean/tanh/smap_center/none | ⭐V4: mask零中心化 |

---

## 数据流程图

```
                    ┌──────────────┐
                    │   txt_feats  │ [B, N, 512]
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ text_query   │
                    │   _proj      │
                    └──────┬───────┘
                           │ Q [B, N, d_k]
          ┌────────────────┴────────────────┐
          │                                  │
   ┌──────▼───────┐                  ┌──────▼───────┐
   │    x_rgb     │                  │    x_ir      │
   └──────┬───────┘                  └──────┬───────┘
          │                                  │
          ▼                                  ▼
   ┌─────────────────────────────────────────────────┐
   │              Attention & Weights                 │
   │   attn_logits_rgb, attn_logits_ir               │
   │   weights = class_weight_mlp(gap_rgb, gap_ir)   │
   └──────────────────┬──────────────────────────────┘
                      │
                      ▼
   ┌─────────────────────────────────────────────────┐
   │              _compute_smap                       │
   │   (根据 smap_order 计算)                         │
   │                                                  │
   │   ⭐ V4: if mask_center='smap_center':          │
   │         S_map = S_map - mean(S_map)             │
   └──────────────────┬──────────────────────────────┘
                      │ S_map [B, 1, H, W]
                      ▼
   ┌─────────────────────────────────────────────────┐
   │              _generate_mask                      │
   │                                                  │
   │   ⭐ V4: 获取约束后的 β, γ (根据 param_constraint)|
   │         β_pos = softplus(β) / abs(β) / β        │
   │                                                  │
   │   mask = mask_method(x_ir, S_map, β_pos, γ_pos) │
   │                                                  │
   │   ⭐ V4: if mask_center='spatial_mean':         │
   │         mask = mask - mean(mask)                │
   └──────────────────┬──────────────────────────────┘
                      │ mask [B, C, H, W]
                      ▼
   ┌─────────────────────────────────────────────────┐
   │              Fusion                              │
   │                                                  │
   │   if param_constraint='residual_alpha':         │
   │     x_fused = x_rgb + α * (x_rgb * mask)        │
   │   else:                                          │
   │     x_fused = x_rgb + x_rgb * mask              │
   │                                                  │
   │   注: mask 有正有负时，正区域增强，负区域抑制    │
   └──────────────────┬──────────────────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │   x_fused    │
               └──────────────┘
```

---

## 使用示例

```python
# 在配置文件中
text_guided_fusion=dict(
    type='TextGuidedRGBIRFusionV4',
    rgb_channels=[128, 256, 512],
    ir_channels=[64, 128, 256],
    text_dim=512,
    num_classes=4,
    beta=1.0,
    gamma=0.5,
    alpha=0.1,
    # V2 参数
    gap_method='logits',
    smap_method='normalized',
    temperature=1.0,
    # V3 参数
    smap_order='sum_first',
    mask_method='conv_gen',
    mask_reduction=8,
    # ⭐ V4 新增参数
    param_constraint='softplus',  # β/γ 始终 > 0
    mask_center='spatial_mean',   # mask 有正有负
)
```

---

## 相关文件

- **模块实现**: `yolo_world/models/necks/ir_correction_rgb_fusion/text_guided_rgb_ir_fusion_v4.py`
- **配置文件**: `configs/custom_flir/ir_correction_rgb_fusion/yolow_v2_rgb_ir_flir_with_text_guided_fusion_v4.py`
- **V3 文档**: `text_guided_rgb_ir_fusion_v3.py` 头部注释
