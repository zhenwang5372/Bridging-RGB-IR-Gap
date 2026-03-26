# TextGuidedRGBEnhancementV2 文本引导的RGB特征增强模块

## 1. 模块概述

`TextGuidedRGBEnhancementV2` 是一个利用**文本语义信息**来生成**类别特定 (Class-Specific)** RGB 特征的模块。该模块使用标准的 **QKV Attention** 机制，让每个类别的文本描述（如"person"、"car"）从融合后的视觉特征中提取与该类别最相关的信息。

### 设计动机

在目标检测中，不同类别的目标具有不同的视觉特征。通过文本语义引导，可以：
1. 为每个类别生成专属的特征表示
2. 让检测头能够区分不同类别的目标
3. 提升多类别检测的精度

---

## 2. 核心算法

### 2.1 标准 QKV Attention 机制

```
Q (Query): 来自 Text（"我要找什么"）
K (Key):   来自 Fused Features（"视觉特征的索引"）
V (Value): 来自 Fused Features（"视觉特征的内容"）

Attention(Q, K, V) = Softmax(Q × K^T / √d_k) × V
```

### 2.2 类别特定特征生成

对于每个类别，使用其文本特征作为 Query，从视觉特征中提取该类别相关的信息：

```
Text["person"]  →  Query_person  →  Attention  →  RGB_person
Text["car"]     →  Query_car     →  Attention  →  RGB_car
Text["bicycle"] →  Query_bicycle →  Attention  →  RGB_bicycle
```

---

## 3. 详细处理流程

```
输入:
  - rgb_feats: 原始RGB特征 (P3, P4, P5) [未使用，保留接口兼容]
  - fused_feats: Fusion后的特征 (P3, P4, P5)
  - text_feats: Text embedding [num_cls, text_dim]

对每个尺度 (P3, P4, P5) 独立处理:

┌────────────────────────────────────────────────────────────────┐
│               SingleLevelRGBEnhancementV2                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Text Features                 Fused Features                  │
│  [num_cls, 512]                [B, C, H, W]                    │
│        │                            │                          │
│        │                            ├──────────────┐           │
│        ▼                            │              │           │
│  ┌──────────────┐                   ▼              ▼           │
│  │ Query Proj   │            ┌──────────┐   ┌──────────┐       │
│  │ Linear       │            │ Key Conv │   │Value Conv│       │
│  │ (512 → d_k)  │            │ 1x1 Conv │   │ 1x1 Conv │       │
│  │              │            │ (C → d_k)│   │ (C → C)  │       │
│  └──────┬───────┘            └────┬─────┘   └────┬─────┘       │
│         │                         │              │             │
│         ▼                         ▼              │             │
│    Q [B, num_cls, d_k]      K [B, d_k, H*W]     │             │
│         │                         │              │             │
│         │                         │              │             │
│         └────────────┬────────────┘              │             │
│                      │                           │             │
│                      ▼                           │             │
│         ┌────────────────────────┐               │             │
│         │   Attention Scores     │               │             │
│         │   Q @ K^T / √d_k       │               │             │
│         │   [B, num_cls, H*W]    │               │             │
│         └───────────┬────────────┘               │             │
│                     │                            │             │
│                     ▼                            │             │
│         ┌────────────────────────┐               │             │
│         │   Softmax (dim=-1)     │               │             │
│         │   A [B, num_cls, H*W]  │               │             │
│         │   每个类别对空间位置    │               │             │
│         │   的注意力权重         │               │             │
│         └───────────┬────────────┘               │             │
│                     │                            │             │
│                     │                            ▼             │
│                     │                   V [B, C, H*W]          │
│                     │                            │             │
│                     │    ┌───────────────────────┘             │
│                     │    │                                     │
│                     ▼    ▼                                     │
│         ┌────────────────────────────────┐                     │
│         │   Per-Class Feature Generation │                     │
│         │                                │                     │
│         │   For each class c:            │                     │
│         │     A_c = A[:, c, :]           │ [B, 1, H*W]         │
│         │     A_spatial = reshape(A_c)   │ [B, 1, H, W]        │
│         │     RGB_c = V × A_spatial      │ [B, C, H, W]        │
│         │                                │                     │
│         │   Stack all classes            │                     │
│         │   RGB_cs = stack(RGB_0...RGB_n)│ [B, num_cls, C,H,W] │
│         └─────────────┬──────────────────┘                     │
│                       │                                        │
│                       ▼                                        │
│         Output: [B, num_cls, C, H, W]                          │
│                 类别特定的RGB特征                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

输出:
  - rgb_class_specific: List of [B, num_cls, C, H, W] for each level
```

---

## 4. 关键设计细节

### 4.1 Query 投影

```python
# 将 text 特征投影到 attention 空间
self.query_proj = nn.Linear(text_dim, d_k)  # 512 → 128

# text_feat: [num_cls, 512]
# Q: [B, num_cls, 128]
Q = self.query_proj(text_expanded)
```

### 4.2 Key/Value 投影

```python
# Key: 用于计算 attention 权重
self.key_conv = nn.Conv2d(rgb_channels, d_k, kernel_size=1)

# Value: 用于生成输出特征
self.value_conv = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1)
```

**设计理念**：
- Key 和 Query 维度相同 (d_k)，便于计算 attention
- Value 保持原始通道数，保留完整的特征信息

### 4.3 类别特定特征的生成

```python
for cls_idx in range(num_cls):
    # 获取该类别的 attention map
    A_spatial = A[:, cls_idx, :].view(B, 1, H, W)  # [B, 1, H, W]
    
    # 用 attention 加权 Value
    rgb_cs_cls = V * A_spatial  # [B, C, H, W]
    
    rgb_class_specific_list.append(rgb_cs_cls)

# 堆叠所有类别
rgb_class_specific = torch.stack(rgb_class_specific_list, dim=1)
# [B, num_cls, C, H, W]
```

**理解**：
- 每个类别有自己的 attention map
- 告诉模型"这个类别的目标可能在哪里"
- 生成该类别专属的特征表示

---

## 5. 关键参数配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `rgb_channels` | [128, 256, 512] | RGB/Fused 特征通道数 |
| `text_dim` | 512 | 文本特征维度 |
| `num_classes` | 4 | 类别数 |
| `d_k` | 128 | Attention 的 key/query 维度 |

---

## 6. 数据流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TextGuidedRGBEnhancementV2 数据流                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          输入数据                                     │   │
│  │                                                                      │   │
│  │  Text Features            Fused Features (from RGB-IR Fusion)        │   │
│  │  [num_cls, 512]          [B, C, H, W] × 3 levels                     │   │
│  │                                                                      │   │
│  │  Example (LLVIP):         P3: [B, 128, H/8, W/8]                     │   │
│  │  - "person" [512]         P4: [B, 256, H/16, W/16]                   │   │
│  │                           P5: [B, 512, H/32, W/32]                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                         │                                   │
│                                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Per-Level Processing                               │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │   ╔═══════════════════════════════════════════════════════════════╗  │   │
│  │   ║                      Level P3                                  ║  │   │
│  │   ╠═══════════════════════════════════════════════════════════════╣  │   │
│  │   ║                                                               ║  │   │
│  │   ║   Text [num_cls, 512]          Fused_P3 [B, 128, H/8, W/8]   ║  │   │
│  │   ║         │                              │                      ║  │   │
│  │   ║         ▼ Linear(512→128)              ├───────┬──────────    ║  │   │
│  │   ║    Q [B, num_cls, 128]                 │       │              ║  │   │
│  │   ║         │                              ▼       ▼              ║  │   │
│  │   ║         │                         K [B,128,HW] V [B,128,HW]   ║  │   │
│  │   ║         │                              │       │              ║  │   │
│  │   ║         └──────────────┬───────────────┘       │              ║  │   │
│  │   ║                        │                       │              ║  │   │
│  │   ║                        ▼                       │              ║  │   │
│  │   ║              Attention = Q @ K^T               │              ║  │   │
│  │   ║              [B, num_cls, H*W]                 │              ║  │   │
│  │   ║                        │                       │              ║  │   │
│  │   ║                        ▼ Softmax               │              ║  │   │
│  │   ║              A [B, num_cls, H*W]               │              ║  │   │
│  │   ║                        │                       │              ║  │   │
│  │   ║                        │                       │              ║  │   │
│  │   ║   ┌────────────────────┴───────────────────────┴────────────┐║  │   │
│  │   ║   │              Per-Class Feature Generation               │║  │   │
│  │   ║   │                                                         │║  │   │
│  │   ║   │   Class 0 ("person"):                                   │║  │   │
│  │   ║   │     A_0 [B,1,H,W] × V → RGB_person [B,128,H/8,W/8]     │║  │   │
│  │   ║   │                                                         │║  │   │
│  │   ║   │   (LLVIP只有1类，FLIR有4类)                             │║  │   │
│  │   ║   │                                                         │║  │   │
│  │   ║   │   Stack: RGB_P3_class_specific [B, num_cls, 128, H/8, W/8] │║  │   │
│  │   ║   └─────────────────────────────────────────────────────────┘║  │   │
│  │   ║                                                               ║  │   │
│  │   ╚═══════════════════════════════════════════════════════════════╝  │   │
│  │                                                                       │   │
│  │   ╔═══════════════════════════════════════════════════════════════╗  │   │
│  │   ║   Level P4 (同样流程，256通道)                                 ║  │   │
│  │   ║   RGB_P4_class_specific [B, num_cls, 256, H/16, W/16]         ║  │   │
│  │   ╚═══════════════════════════════════════════════════════════════╝  │   │
│  │                                                                       │   │
│  │   ╔═══════════════════════════════════════════════════════════════╗  │   │
│  │   ║   Level P5 (同样流程，512通道)                                 ║  │   │
│  │   ║   RGB_P5_class_specific [B, num_cls, 512, H/32, W/32]         ║  │   │
│  │   ╚═══════════════════════════════════════════════════════════════╝  │   │
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                         │                                   │
│                                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           输出数据                                    │   │
│  │                                                                      │   │
│  │  rgb_class_specific: List of 3 tensors                               │   │
│  │    - P3: [B, num_cls, 128, H/8, W/8]                                 │   │
│  │    - P4: [B, num_cls, 256, H/16, W/16]                               │   │
│  │    - P5: [B, num_cls, 512, H/32, W/32]                               │   │
│  │                                                                      │   │
│  │  每个类别有独立的特征表示！                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Attention 可视化理解

```
                        Attention Map 示意图
    
    Text Query: "person"
    
    ┌──────────────────────────────────────┐
    │  Fused Feature Map [H, W]            │
    │                                      │
    │    Low      Medium     High          │
    │   Attention Attention Attention      │
    │                                      │
    │   ░░░░░░    ▒▒▒▒▒▒    ████████      │
    │   ░░░░░░    ▒▒▒▒▒▒    ████████      │  ← 人所在位置
    │   ░░░░░░    ▒▒▒▒▒▒    ████████      │     attention 高
    │                                      │
    │   ░░░░░░    ░░░░░░    ░░░░░░░░      │
    │   ░░░░░░    ░░░░░░    ░░░░░░░░      │  ← 背景位置
    │                                      │     attention 低
    └──────────────────────────────────────┘
    
    RGB_person = V × Attention_person
    → 生成只关注人的类别特定特征
```

---

## 8. 与 V1 版本的对比

| 特性 | V1 | V2 |
|------|----|----|
| Query 来源 | Text | Text |
| Key 来源 | Fused | Fused |
| Value 来源 | RGB (原始) | **Fused** (融合后) |
| 特征质量 | 仅 RGB 信息 | **包含 IR 信息** |

**V2 改进的意义**：
- V1 的 Value 是原始 RGB，不包含 IR 的补充信息
- V2 使用 Fused 特征作为 Value，类别特定特征包含了 RGB+IR 的信息

---

## 9. 使用示例

```python
# 配置文件中的使用方式
rgb_enhancement=dict(
    type='TextGuidedRGBEnhancementV2',
    rgb_channels=[128, 256, 512],
    text_dim=512,
    num_classes=1,  # LLVIP 只有 person 类
    d_k=128,
)
```

---

## 10. 设计优势

1. **语义引导**：文本描述指导生成类别相关特征
2. **类别特定**：每个类别有独立的特征表示
3. **标准 Attention**：使用成熟的 QKV 机制
4. **多尺度独立**：P3、P4、P5 独立处理
5. **融合特征作为 Value**：包含 RGB+IR 的信息

---

## 11. 后续流程

类别特定特征生成后，会进入：
1. **Aggregator**：聚合类别维度
2. **Detection Head**：进行目标检测

```
RGB_class_specific [B, num_cls, C, H, W]
         │
         ▼
    Aggregator (聚合)
         │
         ▼
    [B, C, H, W] (聚合后的特征)
         │
         ▼
    Detection Head
```

---

## 12. 相关文件

- 源代码：`yolo_world/models/necks/text_guided_rgb_enhancement_v2.py`
- 配置示例：`configs/custom_llvip/yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling.py`
