# Stable 配置文件夹

存放稳定版/简化版的 RGB-IR 双流检测配置，用于消融实验。

---

## 配置文件对比

| 配置文件 | Text Update | Aggregator Fusion | 说明 |
|----------|:-----------:|:-----------------:|------|
| `yolow_v2_rgb_ir_flir_stable_no_text_update.py` | ❌ | ❌ | 基础版：两者都禁用 |
| `yolow_v2_rgb_ir_flir_stable_with_text_update.py` | ✅ | ❌ | 仅启用 Text Update |
| `yolow_v2_rgb_ir_flir_stable_with_fusion.py` | ❌ | ✅ | 仅启用 Aggregator Fusion |

---

## 配置文件详细说明

### 1. `yolow_v2_rgb_ir_flir_stable_no_text_update.py`

**基础版** - 两者都禁用

```
数据流：
RGB → Image_Model ──┐
                    ├─→ IR_Correction → Fusion → RGB_Enhancement → Aggregator → Head
IR  → IR_Model   ───┘         ↑                        ↑              ↓
                              │                        │        aggregated_feats
Text → Text_Model ────────────┴────────────────────────┘              │
       │                                                              │
       └──────────────────────────────────────────────────────────────┘
                           text_feats (原始，不更新)
```

### 2. `yolow_v2_rgb_ir_flir_stable_with_text_update.py`

**启用 Text Update** - 文本特征被视觉信息更新

```
数据流：
RGB → Image_Model ──┐
                    ├─→ IR_Correction → Fusion → RGB_Enhancement → Aggregator → Head
IR  → IR_Model   ───┘         ↑                        ↑              ↓
                              │                        │        aggregated_feats
Text → Text_Model ────────────┴────────────────────────┼──→ Text_Update ───┘
                                                       │         ↓
                                                       └── text_updated (更新后)
```

### 3. `yolow_v2_rgb_ir_flir_stable_with_fusion.py`

**启用 Aggregator Fusion** - 聚合特征与 fused_feats 融合

```
数据流：
RGB → Image_Model ──┐
                    ├─→ IR_Correction → Fusion → RGB_Enhancement → Aggregator → Head
IR  → IR_Model   ───┘         ↑              │         ↑              ↓
                              │              │         │    ┌─→ aggregated
Text → Text_Model ────────────┴──────────────│─────────┘    │
       │                                     │              │
       │                                     └── fused_feats ─┘ (融合)
       └─────────────────────────────────────────────────→ final_feats
```

---

## Fusion Type 可选值

在 `aggregator` 配置中，`fusion_type` 参数支持以下值：

| 值 | 说明 | 公式 |
|----|------|------|
| `'none'` | 不融合，直接返回聚合特征 | `output = aggregated` |
| `'add'` | 简单相加 | `output = aggregated + fused` |
| `'concat'` | Concat后1x1卷积降维 (推荐) | `output = Conv1x1(Concat([aggregated, fused]))` |

---

## 训练命令

```bash
# 基础版 (无 Text Update, 无 Fusion)
python tools/train.py configs/custom_flir/stable/yolow_v2_rgb_ir_flir_stable_no_text_update.py

# 启用 Text Update
python tools/train.py configs/custom_flir/stable/yolow_v2_rgb_ir_flir_stable_with_text_update.py

# 启用 Aggregator Fusion
python tools/train.py configs/custom_flir/stable/yolow_v2_rgb_ir_flir_stable_with_fusion.py

# 多卡训练
bash tools/dist_train.sh configs/custom_flir/stable/yolow_v2_rgb_ir_flir_stable_with_fusion.py 2
```

---

## 消融实验设计

| 实验 | Text Update | Fusion | 预期验证 |
|------|:-----------:|:------:|----------|
| Baseline | ❌ | ❌ | 基准性能 |
| +TextUpdate | ✅ | ❌ | Text Update 的贡献 |
| +Fusion | ❌ | ✅ | Aggregator Fusion 的贡献 |
| +Both | ✅ | ✅ | 两者结合的效果 |

---

## 模块作用总结

| 模块 | 作用 |
|------|------|
| **Text Update** | 让文本特征从视觉信息中学习并更新 |
| **Aggregator Fusion** | 聚合后的类别特定特征与原始融合特征结合 |
