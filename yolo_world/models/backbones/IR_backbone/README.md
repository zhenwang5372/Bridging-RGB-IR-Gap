# IR Backbone V2 说明文档

## 📋 版本对比

| 特性 | V1 (原版) | V2 (新版) |
|------|----------|----------|
| **文件位置** | `yolo_world/models/backbones/lite_fft_ir_backbone.py` | `yolo_world/models/backbones/IR_backbone/lite_fft_ir_backboneV2.py` |
| **base_channels 默认值** | 32 | **64** |
| **Spatial Branch** | Depthwise + Pointwise Conv | **Standard 3×3 Conv** |
| **类名后缀** | 无 | **V2** |
| **注册名称** | `LiteFFTIRBackbone` | `LiteFFTIRBackboneV2` |

---

## 🔧 主要改动

### 1️⃣ base_channels 默认值变化

```python
# V1 (原版)
def __init__(self, base_channels: int = 32, ...):
    # P3: 64, P4: 128, P5: 256

# V2 (新版)
def __init__(self, base_channels: int = 64, ...):
    # P3: 128, P4: 256, P5: 512
```

**影响**：
- V2 输出通道数翻倍，特征表达能力更强
- 参数量和计算量增加约 4 倍

---

### 2️⃣ Spatial Branch 架构变化

#### V1 (原版) - Depthwise Separable Convolution

```python
self.spatial_conv = nn.Sequential(
    # Depthwise: 每个通道独立卷积
    nn.Conv2d(in_channels, in_channels, kernel_size=3, 
             padding=1, groups=in_channels, bias=False),
    nn.BatchNorm2d(in_channels),
    nn.SiLU(inplace=True),
    # Pointwise: 跨通道融合
    nn.Conv2d(in_channels, self.spatial_channels, kernel_size=1, bias=False),
    nn.BatchNorm2d(self.spatial_channels),
)
```

**参数量**（假设 in_channels=256, spatial_channels=256）：
- Depthwise: 256 × 9 = 2,304
- Pointwise: 256 × 256 = 65,536
- **总计: 67,840**

#### V2 (新版) - Standard Convolution

```python
self.spatial_conv = nn.Sequential(
    # Standard 3×3 Conv: 直接卷积到目标通道数
    nn.Conv2d(in_channels, self.spatial_channels, kernel_size=3, 
             padding=1, bias=False),
    nn.BatchNorm2d(self.spatial_channels),
    nn.SiLU(inplace=True),
)
```

**参数量**（假设 in_channels=256, spatial_channels=256）：
- Standard Conv: 256 × 256 × 9 = **589,824**
- **总计: 589,824**

**对比**：
- V2 参数量是 V1 的 **8.7 倍**
- V2 特征提取能力更强（跨通道信息交互更充分）
- V1 更轻量（适合移动端/嵌入式设备）

---

### 3️⃣ 类名变化（避免注册冲突）

所有类都添加了 `V2` 后缀：

| V1 类名 | V2 类名 |
|---------|---------|
| `SELayer` | `SELayerV2` |
| `SpectralBlock` | `SpectralBlockV2` |
| `SpectralBlockPreSE` | `SpectralBlockPreSEV2` |
| `SpectralBlockPostSE` | `SpectralBlockPostSEV2` |
| `LiteFFTIRBackbone` | `LiteFFTIRBackboneV2` |
| `LiteFFTIRBackbonePreSE` | `LiteFFTIRBackbonePreSEV2` |
| `LiteFFTIRBackbonePostSE` | `LiteFFTIRBackbonePostSEV2` |

**重要**：类名必须不同，否则会导致 MMYOLO 注册表冲突！

---

## 📊 性能对比（理论分析）

### 参数量对比

假设 `base_channels=32` (V1) vs `base_channels=64` (V2)，`freq_ratio=0.5`：

| 组件 | V1 参数量 | V2 参数量 | 倍数 |
|------|----------|----------|------|
| **Stem** | ~1K | ~4K | 4× |
| **Stage1 (P3)** | ~20K | ~160K | 8× |
| **Stage2 (P4)** | ~80K | ~640K | 8× |
| **Stage3 (P5)** | ~320K | ~2.5M | 8× |
| **总计** | ~420K | ~3.3M | **~8×** |

### 计算量对比 (FLOPs)

输入 640×640 图像：
- **V1**: ~1.5 GFLOPs
- **V2**: ~12 GFLOPs
- **倍数**: ~8×

### 适用场景

| 场景 | 推荐版本 | 理由 |
|------|---------|------|
| **移动端部署** | V1 | 轻量级，推理速度快 |
| **嵌入式设备** | V1 | 内存占用小 |
| **服务器端/GPU** | V2 | 特征表达能力强，精度更高 |
| **高精度要求** | V2 | 更多参数，更强的拟合能力 |
| **实时检测** | V1 | 计算量小，FPS 更高 |

---

## 🚀 使用方法

### 在配置文件中使用

```python
# V1 (原版)
model = dict(
    backbone=dict(
        type='LiteFFTIRBackbone',
        base_channels=32,
        freq_ratio=0.5,
    )
)

# V2 (新版)
model = dict(
    backbone=dict(
        type='LiteFFTIRBackboneV2',  # ← 注意类名后缀
        base_channels=64,             # ← 默认值已改为 64
        freq_ratio=0.5,
    )
)
```

### 在代码中导入

```python
# V1 (原版)
from yolo_world.models.backbones.lite_fft_ir_backbone import (
    LiteFFTIRBackbone,
    SpectralBlock,
)

# V2 (新版)
from yolo_world.models.backbones.IR_backbone import (
    LiteFFTIRBackboneV2,
    SpectralBlockV2,
)
```

---

## ⚠️ 注意事项

1. **不要混用 V1 和 V2 的权重**
   - 两个版本的架构不同，权重不兼容
   - 通道数不同，无法直接迁移

2. **配置文件中的类名必须正确**
   - V1: `type='LiteFFTIRBackbone'`
   - V2: `type='LiteFFTIRBackboneV2'`

3. **V1 和 V2 可以同时存在**
   - 两个版本的类名不同，不会冲突
   - 可以在同一项目中同时使用

4. **V2 需要更多 GPU 显存**
   - 参数量增加 8 倍
   - 建议使用 batch_size 更小或更大的 GPU

---

## 🔬 消融实验建议

### 实验 1: 架构对比
- **V1 (Depthwise)** vs **V2 (Standard Conv)**
- 固定 `base_channels=32`，对比两种卷积方式

### 实验 2: 通道数对比
- **V2 (base=32)** vs **V2 (base=64)**
- 在 V2 架构下对比不同通道数

### 实验 3: SE 位置对比
- `LiteFFTIRBackboneV2` (无 SE)
- `LiteFFTIRBackbonePreSEV2` (Pre-SE)
- `LiteFFTIRBackbonePostSEV2` (Post-SE)

---

## 📝 修改记录

### V2 (2026-01-17)
- ✅ 将 Depthwise + Pointwise 改为 Standard 3×3 Conv
- ✅ base_channels 默认值从 32 改为 64
- ✅ 所有类名添加 V2 后缀
- ✅ 创建独立文件夹 `IR_backbone/`
- ✅ 避免与原版本注册冲突

### V1 (原版)
- 使用 Depthwise Separable Convolution
- base_channels=32
- 轻量级设计

---

## 📧 联系方式

如有问题，请查看：
- 原版文件: `yolo_world/models/backbones/lite_fft_ir_backbone.py`
- V2 文件: `yolo_world/models/backbones/IR_backbone/lite_fft_ir_backboneV2.py`
