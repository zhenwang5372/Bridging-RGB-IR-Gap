# Class-Specific YOLO-World Implementation

## 概述

基于已验证有效的 `yolow_v2_rgb_ir_flir_text_correction.py`，新增类别特定特征生成和跨模态增强机制。

## 文件结构

### 新增模块

1. **TextGuidedRGBEnhancement** (`yolo_world/models/necks/text_guided_rgb_enhancement.py`)
   - 阶段2：为每个尺度独立生成类别特定的RGB特征
   - 输入：原始RGB特征、Fused特征、Text embedding
   - 输出：`[B, num_cls, C, H, W]` 类别特定特征

2. **MultiScaleTextUpdate** (`yolo_world/models/necks/multiscale_text_update.py`)
   - 阶段3：从多尺度特征更新Text embedding
   - 使用YOLO-World风格的残差更新（scale初始化为0）
   - 输出：`[num_cls, text_dim]` 视觉增强的Text

3. **DualStreamMultiModalYOLOBackboneWithClassSpecific** (`yolo_world/models/backbones/dual_stream_class_specific_backbone.py`)
   - 新Backbone：封装所有阶段
   - 保留原有的IR Correction和Fusion
   - 新增阶段2和阶段3的处理

4. **ClassSpecificYOLOHeadModule** (`yolo_world/models/dense_heads/class_specific_yolo_head.py`)
   - 新Head：处理类别特定特征
   - 分类：Region-Text相似度计算
   - 回归：每个类别独立预测bbox

### 配置文件

**yolow_v2_rgb_ir_flir_class_specific_v1.py**
- 基于 `text_correction.py` 的完整配置
- 只修改 backbone 和 head 部分
- 数据pipeline和训练设置完全保持不变

## 核心设计

### 阶段1：特征提取与IR纠错（保持不变）

```
RGB Backbone → RGB特征 [P3, P4, P5]
IR Backbone → IR特征 [P3, P4, P5]
Text Model → Text embedding [num_cls, 512]

TextGuidedIRCorrection → IR_corrected

MultiLevelRGBIRFusion → Fused特征
```

### 阶段2：Text-guided RGB Enhancement（新增）

**对每个尺度独立处理：**

```python
# Step 1: 使用Fused features计算attention
Q = Linear(Text)  # [B, num_cls, d_k]
K = Conv1x1(Fused_Pl)  # [B, d_k, H, W]
Attention = Softmax(Q @ K / sqrt(d_k))  # [B, num_cls, H*W]

# Step 2: 用原始RGB生成类别特定特征
RGB_class_specific = RGB_Pl * Attention  # [B, num_cls, C, H, W]
```

**关键设计：**
- ✓ Fused features用于计算attention（包含RGB和IR的互补信息）
- ✓ 原始RGB用于最终特征（保留完整的视觉细节）
- ✓ 每个类别独立的空间attention map

### 阶段3：Multi-scale Text Update（新增）

```python
# Step 1: 从每个尺度提取视觉证据
Y_text_Pl = GlobalAvgPool(RGB_class_specific_Pl)  # [B, num_cls, C]
Y_text_Pl = Linear(Y_text_Pl)  # [B, num_cls, text_dim]

# Step 2: 多尺度融合（可学习权重）
weights = Softmax([alpha_P3, alpha_P4, alpha_P5])
Y_text_fused = sum(w * Y for w, Y in zip(weights, Y_text_list))

# Step 3: 跨batch聚合
Y_text_avg = Y_text_fused.mean(dim=0)  # [num_cls, text_dim]

# Step 4: YOLO-World风格的残差更新
Text_updated = Text + scale * Y_text_avg
# scale初始化为0，逐渐学习
```

### Head：类别特定检测（修改）

```python
# 分类分支：Region-Text相似度
RGB_proj = Conv1x1(RGB_class_specific_Pl)  # [B, num_cls, 512, H, W]
RGB_norm = F.normalize(RGB_proj, dim=2)
Text_norm = F.normalize(Text_updated, dim=-1)
cls_score = (RGB_norm * Text_norm).sum(dim=2) / temperature

# 回归分支：共享卷积
bbox_pred = RegHead(RGB_class_specific_Pl)  # [B, num_cls, 4, H, W]
```

## 数据流对比

### 原架构（text_correction）
```
Input → RGB/IR/Text → IR Correction → Fusion → Fused特征
     → Neck → Head(Fused, Text) → cls_logit, bbox_pred
```

### 新架构（class_specific）
```
Input → RGB/IR/Text → IR Correction → Fusion → Fused特征
     → RGB Enhancement(RGB, Fused, Text) → RGB_class_specific
     → Text Update(RGB_class_specific, Text) → Text_updated
     → Head(RGB_class_specific, Text_updated) → cls_logit, bbox_pred
```

## 训练方法

### 1. 从头训练

```bash
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_class_specific_v1.py
```

### 2. 从text_correction checkpoint初始化（推荐）

修改配置文件：
```python
load_from = 'work_dirs/text_guided_ir_correction/best_coco_bbox_mAP_50_epoch_XXX.pth'
```

然后训练：
```bash
python tools/train.py configs/custom_flir/yolow_v2_rgb_ir_flir_class_specific_v1.py
```

### 3. 验证

```bash
python tools/test.py \
    configs/custom_flir/yolow_v2_rgb_ir_flir_class_specific_v1.py \
    work_dirs/class_specific_v1/best_coco_bbox_mAP_50_epoch_XXX.pth
```

## 关键参数

### RGB Enhancement
- `d_k = 128`: Attention的key/query维度
- 较小的值（64-128）计算高效
- 较大的值（256-512）表达能力更强

### Text Update
- `scale_init = 0.0`: 残差缩放初始值
- 设为0保护预训练的CLIP text embedding
- 训练过程中逐渐增大（0.1-0.5）

- `fusion_method = 'learned_weight'`: 多尺度融合方式
- 'learned_weight': 可学习的权重（推荐）
- 'equal': 等权重平均（简单baseline）

### Head
- `temperature = 0.07`: 分类温度参数
- 较小的值（0.05-0.1）使相似度更sharp
- 较大的值（0.1-0.2）使分布更平滑

## 消融实验建议

1. **只用Text更新，不用RGB Enhancement**
   - 注释掉 `rgb_enhancement` 配置
   - 测试text更新的单独效果

2. **只用RGB Enhancement，不用Text更新**
   - 注释掉 `text_update` 配置
   - 测试类别特定特征的单独效果

3. **多尺度融合方式**
   - `fusion_method = 'equal'` vs `'learned_weight'`
   - 观察哪个尺度权重最大

4. **Temperature参数扫描**
   - cls_temperature: [0.05, 0.07, 0.1, 0.15]
   - 找到最优的分类温度

## 预期改进

基于设计原理，相比text_correction版本，预期改进：

1. **分类准确度提升**
   - Text获得视觉信息，语义对齐更好
   - 类别特定特征减少类间混淆

2. **定位精度提升**
   - 类别特定的attention聚焦相关区域
   - RGB细节保留完整的边界信息

3. **小目标检测改进**
   - P3尺度的类别特定特征
   - 多尺度text更新平衡各尺度贡献

## 注意事项

1. **内存消耗**
   - RGB_class_specific是 `[B, num_cls, C, H, W]`
   - 相比原来的 `[B, C, H, W]` 增加了 `num_cls` 倍
   - 如果OOM，减小batch_size或使用梯度检查点

2. **训练初期**
   - scale参数从0开始，前几个epoch Text几乎不变
   - 这是正常的，保护CLIP的语义空间
   - 10-20 epoch后scale会逐渐增大

3. **收敛速度**
   - 新模块需要学习，可能比text_correction慢一些
   - 建议至少训练100 epoch观察效果
   - 可以适当降低新模块的学习率（已设置为1.0x）

## 调试建议

如果训练遇到问题：

1. **查看log中的scale参数**
   ```
   backbone.text_update.scale: 0.001 → 0.05 → 0.2 (逐渐增大)
   ```

2. **查看多尺度权重**
   ```
   backbone.text_update.scale_weights: [0.3, 0.5, 0.2]
   # P3, P4, P5的权重分布
   ```

3. **可视化attention map**
   - 在阶段2输出A_spatial，看是否聚焦到物体
   - 检查不同类别的attention是否有区分度

4. **检查Text更新幅度**
   ```python
   delta = torch.norm(text_updated - text_original, dim=-1)
   # 应该逐渐增大，但不要过大（<0.5倍norm）
   ```

## 文件清单

新创建的文件：
- ✓ yolo_world/models/necks/text_guided_rgb_enhancement.py
- ✓ yolo_world/models/necks/multiscale_text_update.py
- ✓ yolo_world/models/backbones/dual_stream_class_specific_backbone.py
- ✓ yolo_world/models/dense_heads/class_specific_yolo_head.py
- ✓ configs/custom_flir/yolow_v2_rgb_ir_flir_class_specific_v1.py

修改的文件：
- ✓ yolo_world/models/backbones/__init__.py
- ✓ yolo_world/models/necks/__init__.py
- ✓ yolo_world/models/dense_heads/__init__.py

所有文件已创建完成，无linter错误！

