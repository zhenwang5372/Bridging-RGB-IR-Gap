# Text-Only Neck 数据流详解

## 📊 完整数据流图

```
输入: RGB图像 [B,3,H,W] + IR图像 [B,3,H,W] + Text ["person", "car", "bicycle", "dog"]
                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                          BACKBONE 阶段                                     │
└───────────────────────────────────────────────────────────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │                               │
    ↓                               ↓
RGB Backbone                    IR Backbone
(CSPDarknet)                    (CSPDarknet)
    │                               │
    ↓                               ↓
RGB原始特征:                     IR原始特征:
├─ P3_rgb: [B, 128, H/8, W/8]   ├─ P3_ir: [B, 64, H/8, W/8]
├─ P4_rgb: [B, 256, H/16,W/16]  ├─ P4_ir: [B, 128, H/16, W/16]
└─ P5_rgb: [B, 512, H/32,W/32]  └─ P5_ir: [B, 256, H/32, W/32]
    │                               │
    └───────────────┬───────────────┘
                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                    FUSION MODULE (MultiLevelRGBIRFusionV2)                │
│                                                                            │
│  对每个尺度 i ∈ {P3, P4, P5}:                                              │
│                                                                            │
│  1. IR通道对齐:                                                            │
│     IR_i_aligned = Conv1x1(IR_i)  # 对齐到与RGB_i相同通道数                │
│     - P3: [B,64,H/8,W/8] → [B,128,H/8,W/8]                                │
│     - P4: [B,128,H/16,W/16] → [B,256,H/16,W/16]                           │
│     - P5: [B,256,H/32,W/32] → [B,512,H/32,W/32]                           │
│                                                                            │
│  2. 注意力融合:                                                            │
│     attention = Sigmoid(MLP(IR_i_aligned))                                 │
│     RGB_attended = RGB_i * attention                                       │
│                                                                            │
│  3. 跨模态融合:                                                            │
│     fused = Conv3x3(Concat[RGB_attended, IR_i_aligned])                    │
│     RGB_i_fused = RGB_i + γ * fused                                        │
│                                                                            │
│  输出:                                                                     │
│     - RGB_fused: (P3, P4, P5) - 融合后的RGB特征 [用于检测头]               │
│     - IR_aligned: (P3, P4, P5) - 对齐后的IR特征 [用于Text更新]             │
└───────────────────────────────────────────────────────────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │                               │
    ↓                               ↓
RGB_fused (用于检测头)           IR_aligned (用于Text更新)
├─ P3: [B,128,H/8,W/8]          ├─ P3: [B,128,H/8,W/8]
├─ P4: [B,256,H/16,W/16]        ├─ P4: [B,256,H/16,W/16]
└─ P5: [B,512,H/32,W/32]        └─ P5: [B,512,H/32,W/32]
    │                               │
    │                               ├──────────────┐
    │                               │              │
    ↓                               ↓              ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                    NECK 阶段 (TextOnlyUpdateNeck)                          │
│                                                                            │
│  核心理念: RGB特征直接透传，只更新Text                                      │
│                                                                            │
│  输入:                                                                     │
│    - rgb_feats: RGB_fused (P3, P4, P5)  ← 来自Fusion模块输出1             │
│    - ir_feats: IR_aligned (P3, P4, P5)  ← 来自Fusion模块输出2             │
│    - text_feats: [num_cls, 512]         ← 来自Text Encoder               │
│                                                                            │
│  处理流程:                                                                 │
│                                                                            │
│  【方案A: 单尺度Text更新 (text_update_scale='P4')】                         │
│    1. 只使用P4尺度:                                                        │
│       X_rgb = RGB_fused[P4]  # [B, 256, H/16, W/16]                       │
│       X_ir = IR_aligned[P4]  # [B, 256, H/16, W/16]                       │
│                                                                            │
│    2. IR-Guided CBAM增强RGB:                                               │
│       channel_attn = ChannelAttention(X_rgb)                               │
│       spatial_mask = SpatialAttention(X_ir)                                │
│       X_rgb' = X_rgb + α * (channel_attn * spatial_mask)                   │
│                                                                            │
│    3. Text-as-Query交叉注意力:                                             │
│       Q = Text W_q         # [num_cls, hidden_dim]                         │
│       K = Flatten(X_rgb') W_k  # [B, N, hidden_dim], N=H*W                │
│       V = Flatten(X_rgb') W_v  # [B, N, hidden_dim]                       │
│       A = Softmax(Q K^T / √d)  # [B, num_cls, N]                          │
│       Y = A V              # [B, num_cls, hidden_dim]                      │
│                                                                            │
│    4. IR权重计算:                                                          │
│       u_ir = Norm(Pool(X_ir) W_ir)  # [B, text_dim]                       │
│       w = Softmax(u_ir Text^T / τ)  # [B, num_cls]                        │
│                                                                            │
│    5. Text更新:                                                            │
│       Y_aligned = MLP(Y)   # [B, num_cls, text_dim]                       │
│       Text_new = Norm(Text + γ * w * Y_aligned)                            │
│                                                                            │
│  【方案B: 多尺度Text更新 (use_multiscale_text_update=True)】                │
│    - multiscale_fusion_first=False: 每个尺度分别更新，再融合增量            │
│    - multiscale_fusion_first=True: 先融合多尺度特征，再统一更新 ⭐推荐      │
│                                                                            │
│    方案B.2 (融合优先，当前配置):                                            │
│      1. 多尺度特征融合 (fusion_method='fpn'):                               │
│         # FPN自顶向下融合                                                  │
│         P5_up = Upsample(P5) → 与P4对齐                                    │
│         P4_fused = Conv(P4 + P5_up)                                        │
│         P4_up = Upsample(P4_fused) → 与P3对齐                              │
│         P3_fused = Conv(P3 + P4_up)                                        │
│         # 统一到target_size=40                                             │
│         X_rgb_fused = Resize([P3_fused, P4_fused, P5]) → [B,C,40,40]      │
│         X_ir_fused = Resize([P3_ir, P4_ir, P5_ir]) → [B,C,40,40]          │
│                                                                            │
│      2. 使用融合特征进行单次Text更新 (同方案A的步骤2-5)                      │
│                                                                            │
│  输出:                                                                     │
│    - rgb_feats: RGB_fused (P3, P4, P5) ← **直接透传，未修改！**            │
│    - text_new: [B, num_cls, 512]       ← **唯一被更新的模态**             │
└───────────────────────────────────────────────────────────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │                               │
    ↓                               ↓
RGB特征 (未修改)                  Text特征 (已更新)
├─ P3: [B,128,H/8,W/8]          Text_new: [B, 4, 512]
├─ P4: [B,256,H/16,W/16]        ├─ "person" embedding (updated)
└─ P5: [B,512,H/32,W/32]        ├─ "car" embedding (updated)
    │                           ├─ "bicycle" embedding (updated)
    │                           └─ "dog" embedding (updated)
    │                               │
    └───────────────┬───────────────┘
                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                          DETECTION HEAD                                    │
│                                                                            │
│  输入:                                                                     │
│    - RGB特征 (P3, P4, P5): 来自Fusion模块，未被Neck修改                    │
│    - Text特征: 动态更新后的文本嵌入                                         │
│                                                                            │
│  处理:                                                                     │
│    - 使用RGB特征提取视觉信息                                                │
│    - 使用更新后的Text特征进行类别分类                                        │
│    - Text特征已经融合了当前图像的RGB+IR视觉信息                              │
└───────────────────────────────────────────────────────────────────────────┘
                    ↓
            检测结果输出
```

---

## 🔍 关键问题解答

### Q1: RGB和IR特征来自Backbone的哪一层？

**答**: 来自**Backbone的最后输出层**（多尺度特征金字塔）

```python
# RGB Backbone (CSPDarknet)
RGB原始输出:
├─ stage3 output → P3_rgb: [B, 128, H/8, W/8]
├─ stage4 output → P4_rgb: [B, 256, H/16, W/16]
└─ stage5 output → P5_rgb: [B, 512, H/32, W/32]

# IR Backbone (CSPDarknet)  
IR原始输出:
├─ stage3 output → P3_ir: [B, 64, H/8, W/8]
├─ stage4 output → P4_ir: [B, 128, H/16, W/16]
└─ stage5 output → P5_ir: [B, 256, H/32, W/32]
```

---

### Q2: 各尺度经过了怎样的处理？

#### **阶段1: Backbone → Fusion (在Backbone内部完成)**

```python
# 对每个尺度 i ∈ {P3, P4, P5}:

1. IR通道对齐:
   IR_i_aligned = Conv1x1(IR_i_raw)
   # 目的: 使IR通道数与RGB相同，便于后续融合
   
   P3: [B,64,H/8,W/8]     → [B,128,H/8,W/8]
   P4: [B,128,H/16,W/16]  → [B,256,H/16,W/16]
   P5: [B,256,H/32,W/32]  → [B,512,H/32,W/32]

2. 注意力融合:
   attention_map = Sigmoid(MLP(IR_i_aligned))
   RGB_attended = RGB_i * attention_map
   # 目的: IR引导RGB关注重要区域

3. 跨模态融合:
   fused = Conv3x3(Concat[RGB_attended, IR_i_aligned])
   RGB_i_fused = RGB_i + γ * fused
   # 目的: 融合RGB和IR信息，增强RGB特征

输出:
├─ RGB_fused: (P3, P4, P5) - 送入检测头
└─ IR_aligned: (P3, P4, P5) - 送入Neck用于Text更新
```

---

#### **阶段2: Fusion → Neck (Text更新)**

**当前配置**: `multiscale_fusion_first=True`, `fusion_method='fpn'`

```python
# 输入:
rgb_feats = (RGB_P3_fused, RGB_P4_fused, RGB_P5_fused)  # 来自Fusion
ir_feats = (IR_P3_aligned, IR_P4_aligned, IR_P5_aligned)  # 来自Fusion
text = [num_cls, 512]  # 来自Text Encoder

# 处理流程:

Step 1: 多尺度特征融合 (FPN方式)
  # RGB特征融合
  P5_up = Upsample(RGB_P5_fused, size=P4.shape)  # [B,512,H/32,W/32] → [B,512,H/16,W/16]
  P4_fused = Conv(RGB_P4_fused + Conv1x1(P5_up))  # [B,256,H/16,W/16]
  
  P4_up = Upsample(P4_fused, size=P3.shape)  # [B,256,H/16,W/16] → [B,256,H/8,W/8]
  P3_fused = Conv(RGB_P3_fused + Conv1x1(P4_up))  # [B,128,H/8,W/8]
  
  # 统一到target_size (40x40)
  P3_resized = Resize(P3_fused, 40x40)  # [B,128,40,40]
  P4_resized = Resize(P4_fused, 40x40)  # [B,256,40,40]
  P5_resized = Resize(RGB_P5_fused, 40x40)  # [B,512,40,40]
  
  # 通道对齐并融合
  P3_aligned = Conv1x1(P3_resized)  # [B,128,40,40] → [B,256,40,40]
  P5_aligned = Conv1x1(P5_resized)  # [B,512,40,40] → [B,256,40,40]
  
  X_rgb_fused = (P3_aligned + P4_resized + P5_aligned) / 3  # [B,256,40,40]
  
  # IR特征同样处理
  X_ir_fused = FPN_Fusion(IR_P3, IR_P4, IR_P5)  # [B,256,40,40]

Step 2: IR-Guided CBAM
  channel_attn = ChannelAttention(X_rgb_fused)  # [B,256,1,1]
  spatial_mask = SpatialAttention(X_ir_fused)   # [B,1,40,40]
  X_rgb' = X_rgb_fused + α * (channel_attn * spatial_mask)

Step 3: Text-as-Query交叉注意力
  X_rgb_flat = Flatten(X_rgb')  # [B, 1600, 256], N=40*40=1600
  Q = Text W_q  # [num_cls, hidden_dim]
  K = X_rgb_flat W_k  # [B, 1600, hidden_dim]
  V = X_rgb_flat W_v  # [B, 1600, hidden_dim]
  
  A = Softmax(Q K^T / √d)  # [B, num_cls, 1600]
  Y = A V  # [B, num_cls, hidden_dim]

Step 4: IR权重计算
  u_ir = Norm(GlobalPool(X_ir_fused) W_ir)  # [B, text_dim]
  w = Softmax(u_ir Text^T / τ)  # [B, num_cls]

Step 5: Text更新
  Y_aligned = MLP(Y)  # [B, num_cls, text_dim]
  delta = γ * w.unsqueeze(-1) * Y_aligned  # [B, num_cls, text_dim]
  Text_new = Norm(Text + delta)  # [B, num_cls, text_dim]

# 输出:
return rgb_feats, Text_new  # rgb_feats直接透传，未修改！
```

---

### Q3: 为什么RGB特征不修改？

**核心理念**: **RGB特征作为稳定的"锚点"**

1. **稳定性**: RGB特征已经在Fusion阶段融合了IR信息，质量较高
2. **避免累积误差**: 如果RGB也更新，Text更新会依赖不稳定的RGB
3. **简化训练**: 只有Text一个学习目标，梯度回传更清晰
4. **检测头适配**: 检测头已经在Fusion后的RGB特征上预训练

---

### Q4: Text更新使用的是哪些特征？

**答**: 使用**Fusion模块输出的IR_aligned特征**

```python
# 不是IR Backbone的原始输出！
# 而是经过通道对齐后的IR特征

输入Neck的IR特征:
├─ P3_ir: [B, 128, H/8, W/8]   ← 已对齐到RGB_P3通道数
├─ P4_ir: [B, 256, H/16, W/16] ← 已对齐到RGB_P4通道数
└─ P5_ir: [B, 512, H/32, W/32] ← 已对齐到RGB_P5通道数

# 这些IR特征:
# 1. 通道数已对齐
# 2. 空间分辨率已对齐
# 3. 可以直接与RGB特征配合使用
```

---

## 📋 总结

### 数据来源

| 模态 | 来源 | 尺度 | 通道数 | 用途 |
|------|------|------|--------|------|
| **RGB_fused** | Fusion模块输出1 | P3/P4/P5 | 128/256/512 | 送入检测头（未修改） |
| **IR_aligned** | Fusion模块输出2 | P3/P4/P5 | 128/256/512 | Text更新的辅助信息 |
| **Text** | Text Encoder | - | 512 | 被动态更新 |

### 处理流程

1. **Backbone**: RGB和IR分别提取特征
2. **Fusion**: RGB+IR融合，输出RGB_fused和IR_aligned
3. **Neck**: 使用RGB_fused和IR_aligned更新Text，RGB_fused透传
4. **Head**: 使用RGB_fused和Text_new进行检测

### 关键特点

- ✅ **RGB特征**: 来自Fusion，直接透传到Head
- ✅ **IR特征**: 来自Fusion（已对齐），仅用于Text更新
- ✅ **Text特征**: 唯一被更新的模态
- ✅ **最简单**: 只有一个学习目标
- ✅ **最稳定**: RGB作为稳定锚点

