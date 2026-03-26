# MultiScaleTextUpdateV4 多尺度文本更新模块

## 1. 模块概述

`MultiScaleTextUpdateV4` 是一个让**文本特征从视觉信息中学习并更新**的模块，参考了 YOLO-World 的 `ImagePoolingAttentionModule` 设计。核心思想是：文本语义不应该是静态的，而应该能够根据当前图像的内容进行自适应调整。

### 设计动机

在传统方案中，文本特征（来自 CLIP 等预训练模型）是固定的。但在实际检测场景中：
- 同一类别在不同场景下的外观可能差异很大
- 图像中的视觉信息可以帮助文本特征更好地理解当前场景
- 更新后的文本特征可以提供更精准的语义指导

---

## 2. 核心设计

### 2.1 参考 YOLO-World 的 I-Pooling 机制

YOLO-World 使用 Image Pooling Attention 来让文本特征与视觉特征交互，本模块借鉴了这一设计：

1. **Adaptive Pooling**：将多尺度视觉特征池化到固定大小
2. **Multi-Head Cross-Attention**：Text 作为 Query，Image 作为 Key/Value
3. **残差更新**：使用可学习的 scale 参数控制更新强度

### 2.2 关键改进

| 特性 | 说明 |
|------|------|
| Adaptive Pooling | 使用 avg pooling（支持确定性训练） |
| Multi-Head Attention | 8 头并行计算 |
| LayerNorm | 稳定训练 |
| Learnable Scale | 初始为 0，逐渐学习更新强度 |
| 跨 Batch 聚合（可选） | 可以跨 batch 聚合文本更新 |

---

## 3. 详细处理流程

```
输入:
  - fused_feats: 融合后的视觉特征 (P3, P4, P5)
  - text_feats: 文本特征 [B, num_cls, text_dim] 或 [num_cls, text_dim]

处理流程:

Step 1: 多尺度视觉特征的池化与投影
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Fused_P3 [B, 128, H/8, W/8]                                   │
│      │                                                         │
│      ▼ Conv1x1(128 → 256) + BN                                 │
│      │                                                         │
│      ▼ AdaptiveAvgPool2d(3×3)                                  │
│      │                                                         │
│      ▼ Reshape                                                 │
│      [B, 256, 9]                                                │
│                                                                │
│  Fused_P4 [B, 256, H/16, W/16]                                 │
│      │                                                         │
│      ▼ Conv1x1(256 → 256) + BN                                 │
│      │                                                         │
│      ▼ AdaptiveAvgPool2d(3×3)                                  │
│      │                                                         │
│      ▼ Reshape                                                 │
│      [B, 256, 9]                                                │
│                                                                │
│  Fused_P5 [B, 512, H/32, W/32]                                 │
│      │                                                         │
│      ▼ Conv1x1(512 → 256) + BN                                 │
│      │                                                         │
│      ▼ AdaptiveAvgPool2d(3×3)                                  │
│      │                                                         │
│      ▼ Reshape                                                 │
│      [B, 256, 9]                                                │
│                                                                │
│  Concatenate along last dim:                                   │
│  image_features = [B, 256, 27] → transpose → [B, 27, 256]      │
│  (3 levels × 9 patches = 27 visual tokens)                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Step 2: Multi-Head Cross-Attention
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Text Features [B, num_cls, 512]                               │
│        │                                                       │
│        ▼ LayerNorm + Linear(512 → 256)                         │
│    Q [B, num_cls, 256]                                         │
│        │                                                       │
│        ▼ Reshape for multi-head                                │
│    Q [B, num_cls, 8, 32]  (8 heads, 32 dim per head)          │
│                                                                │
│  Image Features [B, 27, 256]                                   │
│        │                                                       │
│        ├──────────────────────────┐                            │
│        ▼                          ▼                            │
│    LayerNorm + Linear        LayerNorm + Linear                │
│    K [B, 27, 256]            V [B, 27, 256]                    │
│        │                          │                            │
│        ▼ Reshape                  ▼ Reshape                    │
│    K [B, 27, 8, 32]          V [B, 27, 8, 32]                  │
│                                                                │
│  Attention (using einsum):                                     │
│    attn = einsum('bnmc,bkmc->bmnk', Q, K)                      │
│         = [B, 8, num_cls, 27]                                  │
│                                                                │
│    attn = attn / √32  (scale by √head_dim)                     │
│    attn = softmax(attn, dim=-1)                                │
│                                                                │
│  Output:                                                       │
│    x = einsum('bmnk,bkmc->bnmc', attn, V)                      │
│      = [B, num_cls, 8, 32]                                     │
│                                                                │
│    x = reshape → [B, num_cls, 256]                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Step 3: 投影与残差更新
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Attention Output [B, num_cls, 256]                            │
│        │                                                       │
│        ▼ Linear(256 → 512)                                     │
│    Update [B, num_cls, 512]                                    │
│                                                                │
│  Residual Update:                                              │
│    text_updated = text_feats + scale × Update                  │
│                                                                │
│    其中 scale 初始为 0，可学习                                   │
│    训练初期: text_updated ≈ text_feats (保持稳定)              │
│    训练后期: text_updated = text_feats + learned_scale × Update│
│                                                                │
└────────────────────────────────────────────────────────────────┘

输出:
  - text_updated: [B, num_cls, text_dim] 更新后的文本特征
```

---

## 4. 关键设计细节

### 4.1 Adaptive Pooling 的选择

```python
# 默认使用 avg pooling（支持确定性训练）
if pool_type == 'max':
    self.image_pools = nn.ModuleList([
        nn.AdaptiveMaxPool2d((pool_size, pool_size))
        for _ in range(self.num_feats)
    ])
else:  # 'avg'
    self.image_pools = nn.ModuleList([
        nn.AdaptiveAvgPool2d((pool_size, pool_size))
        for _ in range(self.num_feats)
    ])
```

**为什么用 AvgPool**：
- MaxPool 的 CUDA 反向传播不支持确定性模式
- AvgPool 支持确定性训练，结果可复现

### 4.2 可学习的 Scale 参数

```python
# 初始为 0，让模型学习何时更新
self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

# 残差更新
text_updated = text_feats + self.scale * x
```

**设计理念**：
- `scale=0`：完全保持原始文本特征
- 训练过程中学习合适的更新强度
- 避免一开始就大幅改变预训练的文本特征

### 4.3 跨 Batch 聚合（可选）

```python
if self.cross_batch:
    # 跨 Batch 聚合：对所有图片的更新量取平均
    x_avg = x.mean(dim=0)  # [num_cls, text_dim]
    text_feats_2d = text_feats[0]  # [num_cls, text_dim]
    text_updated_2d = text_feats_2d + self.scale * x_avg
    text_updated = text_updated_2d.unsqueeze(0).expand(B, -1, -1)
else:
    # 每张图片独立更新
    text_updated = text_feats + self.scale * x
```

**应用场景**：
- `cross_batch=True`：适合单类别或类别较少的场景
- `cross_batch=False`：每张图片有独立的文本更新

---

## 5. 关键参数配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `in_channels` | [128, 256, 512] | 输入特征通道数 |
| `text_dim` | 512 | 文本特征维度 |
| `embed_channels` | 256 | Attention 嵌入维度 |
| `num_heads` | 8 | Multi-Head 数量 |
| `pool_size` | 3 | Pooling 输出大小 (3×3=9 tokens) |
| `with_scale` | True | 是否使用可学习 scale |
| `pool_type` | 'avg' | Pooling 类型 ('avg' 或 'max') |
| `cross_batch` | False | 是否跨 Batch 聚合 |

---

## 6. 调试输出

模块内置了训练过程的监控：

```python
# 每 200 个 iteration 打印一次
[TextUpdateV4] iter=200: mode=per_image, scale=0.0123, update_norm=1.2345, orig_norm=45.6789
```

这帮助我们监控：
- `scale`：更新强度学习进展
- `update_norm`：更新量的大小
- `orig_norm`：原始文本特征的大小

---

## 7. 数据流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MultiScaleTextUpdateV4 数据流                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          输入数据                                     │   │
│  │                                                                      │   │
│  │  Fused Features (P3, P4, P5)              Text Features              │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐        [B, num_cls, 512]          │   │
│  │  │P3: 128 │ │P4: 256 │ │P5: 512 │             或                     │   │
│  │  │H/8×W/8 │ │H/16×W/16│ │H/32×W/32│        [num_cls, 512]            │   │
│  │  └────────┘ └────────┘ └────────┘                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                    │          │          │              │                   │
│                    │          │          │              │                   │
│                    ▼          ▼          ▼              │                   │
│  ┌─────────────────────────────────────────────────────┐│                   │
│  │              Step 1: 视觉特征池化                    ││                   │
│  ├─────────────────────────────────────────────────────┤│                   │
│  │                                                     ││                   │
│  │  P3 ─► Conv1x1(128→256) ─► BN ─► AvgPool(3×3)      ││                   │
│  │        ─► [B, 256, 9]                               ││                   │
│  │                                                     ││                   │
│  │  P4 ─► Conv1x1(256→256) ─► BN ─► AvgPool(3×3)      ││                   │
│  │        ─► [B, 256, 9]                               ││                   │
│  │                                                     ││                   │
│  │  P5 ─► Conv1x1(512→256) ─► BN ─► AvgPool(3×3)      ││                   │
│  │        ─► [B, 256, 9]                               ││                   │
│  │                                                     ││                   │
│  │  Concat: [B, 256, 27] ─► Transpose ─► [B, 27, 256] ││                   │
│  │                                                     ││                   │
│  │  (27 visual tokens = 3 levels × 9 patches)         ││                   │
│  │                                                     ││                   │
│  └─────────────────────────────┬───────────────────────┘│                   │
│                                │                        │                   │
│                                ▼                        ▼                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                 Step 2: Multi-Head Cross-Attention                    │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │   Text Features [B, num_cls, 512]                                     │   │
│  │         │                                                             │   │
│  │         ▼ LayerNorm + Linear(512→256)                                 │   │
│  │    Q [B, num_cls, 256]                                                │   │
│  │         │                                                             │   │
│  │         ▼ Reshape: [B, num_cls, 8heads, 32dim]                        │   │
│  │                                                                       │   │
│  │   Image Features [B, 27, 256]                                         │   │
│  │         │                                                             │   │
│  │         ├─────────────────────────────────┐                           │   │
│  │         ▼                                 ▼                           │   │
│  │    LayerNorm + Linear              LayerNorm + Linear                 │   │
│  │    K [B, 27, 8, 32]                V [B, 27, 8, 32]                   │   │
│  │                                                                       │   │
│  │   ┌────────────────────────────────────────────────────────────┐     │   │
│  │   │         Multi-Head Attention Computation                    │     │   │
│  │   │                                                             │     │   │
│  │   │   attn_weight = einsum('bnmc,bkmc->bmnk', Q, K)             │     │   │
│  │   │              = [B, 8heads, num_cls, 27tokens]               │     │   │
│  │   │                                                             │     │   │
│  │   │   attn_weight = attn_weight / √32                           │     │   │
│  │   │   attn_weight = softmax(attn_weight, dim=-1)                │     │   │
│  │   │                                                             │     │   │
│  │   │   output = einsum('bmnk,bkmc->bnmc', attn_weight, V)        │     │   │
│  │   │          = [B, num_cls, 8heads, 32dim]                      │     │   │
│  │   │                                                             │     │   │
│  │   │   output = reshape → [B, num_cls, 256]                      │     │   │
│  │   │                                                             │     │   │
│  │   └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                       │   │
│  └───────────────────────────────────┬───────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Step 3: 投影与残差更新                             │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │   Attention Output [B, num_cls, 256]                                  │   │
│  │         │                                                             │   │
│  │         ▼ Linear(256 → 512)                                           │   │
│  │   Update [B, num_cls, 512]                                            │   │
│  │         │                                                             │   │
│  │         │           Original Text [B, num_cls, 512]                   │   │
│  │         │                    │                                        │   │
│  │         │                    │                                        │   │
│  │         ▼                    ▼                                        │   │
│  │   ┌─────────────────────────────────────────────────┐                 │   │
│  │   │           Residual Update                        │                 │   │
│  │   │                                                  │                 │   │
│  │   │   text_updated = text_feats + scale × Update    │                 │   │
│  │   │                                                  │                 │   │
│  │   │   scale 初始为 0.0，可学习                       │                 │   │
│  │   │                                                  │                 │   │
│  │   │   训练初期: text_updated ≈ text_feats           │                 │   │
│  │   │   训练后期: 学习到合适的 scale                   │                 │   │
│  │   │                                                  │                 │   │
│  │   └─────────────────────────────────────────────────┘                 │   │
│  │                                                                       │   │
│  └───────────────────────────────────┬───────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           输出数据                                    │   │
│  │                                                                      │   │
│  │   text_updated: [B, num_cls, 512]                                    │   │
│  │                                                                      │   │
│  │   文本特征已经根据当前图像的视觉信息进行了更新                         │   │
│  │   → 更适合当前检测场景的语义表示                                      │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 与 YOLO-World I-Pooling 的对比

| 特性 | YOLO-World I-Pooling | MultiScaleTextUpdateV4 |
|------|---------------------|------------------------|
| 输入视觉特征 | RGB 特征 | **Fused RGB-IR 特征** |
| Pooling 类型 | Max Pooling | **Avg Pooling** (确定性) |
| 跨 Batch 聚合 | 支持 | 支持 (可选) |
| Scale 初始值 | 1.0 | **0.0** (更保守) |

---

## 9. 使用示例

```python
# 配置文件中的使用方式
text_update=dict(
    type='MultiScaleTextUpdateV4',
    in_channels=[128, 256, 512],
    text_dim=512,
    embed_channels=256,
    num_heads=8,
    with_scale=True,
    # pool_type='avg',  # 默认使用 avg，支持确定性训练
    # cross_batch=False,  # 默认每张图片独立更新
)
```

---

## 10. 设计优势

1. **动态文本特征**：文本语义可以根据图像内容调整
2. **多尺度视觉信息**：融合 P3、P4、P5 三个尺度的信息
3. **Multi-Head Attention**：并行学习多种 attention 模式
4. **稳定训练**：LayerNorm + scale 初始为 0
5. **确定性支持**：使用 AvgPool 支持可复现的训练
6. **灵活配置**：支持跨 Batch 聚合或独立更新

---

## 11. 在整体架构中的位置

```
           ┌────────────────────────────────────────────────────────┐
           │                    完整数据流                          │
           ├────────────────────────────────────────────────────────┤
           │                                                        │
  RGB ──►  │  RGB Backbone                                          │
           │       │                                                │
           │       ▼                                                │
  IR  ──►  │  IR Backbone (LiteFFTIRBackbone)                       │
           │       │                                                │
           │       ├──────────────────────────────────────────┐     │
           │       ▼                                          │     │
  Text ──► │  IR Correction (TextGuidedIRCorrectionV4)       │     │
           │       │                                          │     │
           │       ▼                                          │     │
           │  RGB-IR Fusion (MultiLevelRGBIRFusion)          │     │
           │       │                                          │     │
           │       ▼                                          │     │
           │  RGB Enhancement (TextGuidedRGBEnhancementV2)   │     │
           │       │                                          │     │
           │       ├──────────────────────────────────────────┘     │
           │       ▼                                                │
           │  ⭐ Text Update (MultiScaleTextUpdateV4)               │
           │       │                                                │
           │       ▼ (返回更新后的 text_feats)                      │
           │                                                        │
           │  Aggregator ──► Detection Head                         │
           │                                                        │
           └────────────────────────────────────────────────────────┘
```

---

## 12. 相关文件

- 源代码：`yolo_world/models/necks/multiscale_text_update_v4.py`
- 配置示例：`configs/custom_llvip/yolow_v2_rgb_ir_llvip_stable_with_text_update_maxpooling.py`
- 参考：YOLO-World 的 `ImagePoolingAttentionModule`
