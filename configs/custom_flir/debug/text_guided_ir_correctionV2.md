# Text-guided IR Correction V2 说明文档

## 日期
2026-01-15

---

## 1. 方案概述

### 图片形式的流程图名称
这种形式的图叫做 **"ASCII Art 架构图"** 或 **"字符流程图/树形图"**。

---

## 2. 架构流程图 (ASCII Art)

```
DualStreamMultiModalYOLOBackboneWithCorrectionV2
│
├─ RGB Backbone (YOLOv8CSPDarknet)
│  └─ 输出: P3[128, 80, 80], P4[256, 40, 40], P5[512, 20, 20]
│
├─ IR Backbone (LiteFFTIRBackbone)
│  └─ 输出: P3[64, 80, 80], P4[128, 40, 40], P5[256, 20, 20]
│
├─ Text Model (CLIP)
│  └─ 输出: txt_feats [B, N, 512] (N=4类别)
│
├─ IR Correction V2 (TextGuidedIRCorrectionV2)  ⭐ 核心改进
│  │
│  ├─ Step 1: 文本引导的语义激活 (Semantic Activation)
│  │  ├─ Q_text = Text @ W_Q         [B, N, d_k]
│  │  ├─ K_rgb = φ(X_rgb)            [B, d_k, H*W]
│  │  ├─ K_ir = φ(X_ir)              [B, d_k, H*W]
│  │  ├─ A_rgb = Softmax(Q @ K_rgb^T / √d_k)  [B, N, H*W]
│  │  └─ A_ir = Softmax(Q @ K_ir^T / √d_k)   [B, N, H*W]
│  │
│  ├─ Step 2: 语义一致性度量 (Semantic Consistency)
│  │  ├─ G_c = cosine(A_rgb[:, c, :], A_ir[:, c, :])
│  │  └─ G ∈ [0, 1]: G≈1一致, G≈0不一致
│  │
│  ├─ Step 3: 加权差异图生成 (Weighted Difference)  ⭐ 改进
│  │  ├─ D_spatial = |A_rgb - A_ir|           [B, N, H*W]
│  │  ├─ M_err = Σ_c (1-G_c) × D_spatial^c    [B, H*W]
│  │  └─ M_err = MinMax_Normalize(M_err)      [B, 1, H, W]  ⭐ 新增
│  │
│  ├─ Step 4: 空间门控 (Spatial Gating)  ⭐ 核心改进
│  │  ├─ F_extracted = X_ir ⊙ M_err    [B, C_ir, H, W] (乘法 mask)
│  │  └─ Error_map = f_conv(F_extracted)  (轻量级CNN)
│  │
│  └─ Step 5: 特征纠正 (Feature Rectification)
│     └─ X_ir^corrected = X_ir - α × Error_map
│
└─ RGB-IR Fusion (MultiLevelRGBIRFusion)
   ├─ P3: LightweightCrossFusion
   │  └─ 输出: [B, 128, 80, 80]
   ├─ P4: LightweightCrossFusion
   │  └─ 输出: [B, 256, 40, 40]
   └─ P5: LightweightCrossFusion
      └─ 输出: [B, 512, 20, 20]
```

---

## 3. V1 vs V2 核心差异

### 3.1 Step 3: 归一化方式

| 版本 | 方法 | 公式 |
|-----|------|------|
| **V1** | Softmax 加权 | `w = Softmax((1-G) / τ)` |
| **V2** | Min-Max 归一化 | `M_err = (M_err - min) / (max - min + ε)` |

**V2 优势**: Min-Max 归一化保持更好的数值分布，避免 Softmax 的权重过度集中问题。

### 3.2 Step 4: 错误特征提取方式

| 版本 | 方法 | 公式 |
|-----|------|------|
| **V1** | Concat 方式 | `concat(X_ir, Diff_spatial) → CNN` |
| **V2** | 空间门控 (Spatial Gating) | `X_ir ⊙ M_err → f_conv` |

**V2 优势**:
1. **物理可解释性更强**: `M_err` 直接指出"哪里错了"，`X_ir ⊙ M_err` 直接提取"那里的特征是什么"
2. **避免维度混淆**: CNN 不需要学习如何从 concat 中分离差异信息
3. **更精准的错误定位**: 乘法 mask 直接筛选错误区域

---

## 4. 数学公式详解

### Step 1: 文本引导的语义激活

$$
Q_{text} = Text_{emb} W_Q, \quad Q_{text} \in \mathbb{R}^{B \times N \times d_k}
$$

$$
K_{rgb} = \phi(X_{rgb}), \quad K_{ir} = \phi(X_{ir}), \quad K \in \mathbb{R}^{B \times d_k \times HW}
$$

$$
A_{rgb} = \text{Softmax}\left(\frac{Q_{text} K_{rgb}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times N \times HW}
$$

### Step 2: 语义一致性度量

$$
G_c = \frac{A_{rgb}[:, c, :] \cdot A_{ir}[:, c, :]}{\|A_{rgb}[:, c, :]\|_2 \|A_{ir}[:, c, :]\|_2}, \quad G \in \mathbb{R}^{B \times N}
$$

- $G_c \approx 1$: 该类别在两模态中表现一致（可信）
- $G_c \approx 0$: 该类别表现不一致（IR 可能存在语义错误）

### Step 3: 加权差异图生成 (V2 改进)

$$
D_{spatial}^c = | A_{rgb}[:, c, :] - A_{ir}[:, c, :] | \in \mathbb{R}^{B \times HW}
$$

$$
M_{err} = \sum_{c=1}^{N} (1 - G_c) \cdot D_{spatial}^c
$$

**⭐ V2 新增: Min-Max 归一化**
$$
M_{err} = \frac{M_{err} - \min(M_{err})}{\max(M_{err}) - \min(M_{err}) + \epsilon}
$$

### Step 4: 空间门控 (V2 核心改进)

**⭐ V2 使用乘法 mask（空间选择器）**
$$
F_{extracted} = X_{ir} \odot M_{err}
$$

$$
Error_{map} = f_{conv}(F_{extracted})
$$

### Step 5: 特征纠正

$$
X_{ir}^{corrected} = X_{ir} - \alpha \cdot Error_{map}
$$

---

## 5. 代码结构

### 文件位置

```
yolo_world/models/necks/
└── text_guided_ir_correction/                # IR 纠错模块子包
    ├── __init__.py                           # 导出 V1 和 V2
    ├── text_guided_ir_correction_v1.py       # V1 原始版本
    └── text_guided_ir_correction_v2.py       # V2 核心实现（空间门控）
```

**说明**: 参考 `simplified_trimodal/` 的组织方式，将 V1 和 V2 放在同一个子包中，便于管理和维护。

### 类注册

```python
@MODELS.register_module()
class TextGuidedIRCorrectionV2(BaseModule):
    """多尺度 IR 纠错模块 V2"""
    pass

@MODELS.register_module()
class DualStreamMultiModalYOLOBackboneWithCorrectionV2(BaseModule):
    """双流 Backbone + IR 纠错 V2"""
    pass
```

---

## 6. 配置文件

### 配置路径
`configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV2.py`

### 关键配置

```python
# IR Correction V2 参数
correction_alpha = 0.3  # 纠错强度初始值

# Backbone 配置
backbone=dict(
    type='DualStreamMultiModalYOLOBackboneWithCorrectionV2',
    ir_correction=dict(
        type='TextGuidedIRCorrectionV2',
        rgb_channels=[128, 256, 512],
        ir_channels=[64, 128, 256],
        text_dim=512,
        num_classes=4,
        correction_alpha=0.3,
    ),
    ...
)
```

---

## 7. 训练命令

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# 单卡训练
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV2.py

# 多卡训练
bash tools/dist_train.sh configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV2.py 2
```

---

## 8. 预期效果

### V2 相比 V1 的预期改进

| 指标 | 预期变化 | 原因 |
|-----|---------|------|
| **收敛速度** | ⬆️ 更快 | 空间门控更精准地定位错误 |
| **mAP** | ⬆️ 略有提升 | 更强的物理可解释性 |
| **训练稳定性** | ⬆️ 更稳定 | Min-Max 归一化避免极端值 |
| **显存占用** | ≈ 相当 | 计算复杂度相似 |

---

## 9. 可视化建议

### M_err 差异图可视化

```python
# 在 forward 中保存 M_err 用于可视化
M_err_spatial = M_err.view(B, 1, H, W)  # [B, 1, H, W]
# 保存或返回用于可视化
```

**可视化内容**:
1. `M_err`: 错误区域热力图（高亮"文本关注的、且RGB/IR产生巨大分歧"的区域）
2. `F_extracted`: 门控后的特征
3. `Error_map`: 估计的纠正量
4. `ir_corrected - x_ir`: 实际纠正差异

---

## 10. 参考资料

- 原始方案文档: `strategy/ir_correction.markdown`
- V1 实现: `yolo_world/models/necks/text_guided_ir_correction.py`
- RGB-IR 融合: `yolo_world/models/necks/rgb_ir_fusion.py`

