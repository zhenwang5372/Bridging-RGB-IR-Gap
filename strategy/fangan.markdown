# 多模态（RGB-IR）文本引导目标检测架构设计方案

## 1. 问题分析：之前方案的致命缺陷

在之前的方案中，RGB 特征的处理方式导致了严重的定位问题：

*   **旧方案流程**：
    $$ \text{RGB\_embedding\_class}: [B, \text{num\_cls}, \text{embed\_dim}] \rightarrow \text{全局池化聚合} \rightarrow \text{丢失所有空间信息} $$
*   **后果**：无法区分同一类别的多个目标。
*   **示例**：
    *   场景中有 3 辆 `car`。
    *   特征 `RGB_embedding_class[:, car_idx, :]` 仅仅是一个向量。
    *   **结果**：无法表示 3 个不同的 `car`，导致分类和定位失败。

### 你的方案优势（改进后）

*   **新方案核心**：
    $$ \text{RGB\_class\_specific}: [B, \text{num\_cls}, C, H, W] $$
*   **特点**：
    1.  保留完整空间信息。
    2.  每个空间位置独立预测。
    3.  **可以区分同一类别的多个目标！**
*   **示例**：
    *   3 辆 `car` 在不同位置 $(x_1,y_1), (x_2,y_2), (x_3,y_3)$。
    *   `RGB_class_specific[:, car_idx, :, y1, x1]` $\rightarrow$ `car1` 的特征。
    *   每个位置独立预测 BBox 和分类。

---

## 2. 整体架构（简化后的完整方案）

**输入**：
*   RGB ($P3, P4, P5$)
*   IR ($P3, P4, P5$)
*   Text Prompt

**三阶段流程**：
1.  **Stage 1: New IR (IR 纠错)** $\rightarrow$ 输出 `IR_corrected`
2.  **Stage 2: New RGB (RGB 增强)** $\rightarrow$ 输出 `RGB_class_specific` (**唯一视觉输出**)
3.  **Stage 3: New Text (Text 更新)** $\rightarrow$ 输出 `Text_updated`

**输出到 Head**：
利用 `RGB_class_specific` 和 `Text_updated` 同时进行分类和回归。

---

## 3. 详细流程拆解

### 阶段 1：New IR - IR 纠错（保持不变）

对于每个尺度 $l \in \{3, 4, 5\}$：

1.  **Text-as-Query**：生成 $A_{rgb}, A_{ir}$。
2.  **计算一致性**：计算 $G$ 和差异图 $\text{Diff}$。
3.  **生成错误图**：$\text{Error\_map}$。
4.  **纠正公式**：
    $$ IR\_corrected\_P_l = IR\_P_l - \alpha \times Error\_map $$
    *   *注：已验证有效（16轮测试）。*

---

### 阶段 2：New RGB - RGB 增强（大幅简化）

对于每个尺度 $l \in \{3, 4, 5\}$，独立执行以下步骤：

#### Step 1: IR 过滤前景
利用 IR 的热显著性作为“软 Mask”，突出前景，抑制背景。
$$ RGB\_corrected\_P_l = RGB\_P_l \times IR\_corrected\_P_l $$
*   **形状**：$[B, C, H_l, W_l]$

#### Step 2: Text-guided Cross-Attention
让 Text 查询 `RGB_corrected`，生成每个类别的空间注意力图。

1.  **生成 Q, K**：
    $$ Q = \text{Linear}(\text{Text}) \rightarrow [B, \text{num\_cls}, d_k] $$
    $$ K = \text{Conv}(RGB\_corrected\_P_l) \rightarrow [B, d_k, H_l, W_l] $$
2.  **计算 Attention**：
    $$ A = \text{Softmax}(Q \cdot K_{\text{flatten}}) $$
    $$ A\_spatial = \text{Reshape}(A) \rightarrow [B, \text{num\_cls}, H_l, W_l] $$
    *   *含义*：$A\_spatial[b, c, h, w]$ 表示位置 $(h,w)$ 对类别 $c$ 的响应。

#### Step 3: 生成类别特定的空间特征（核心输出）

关键设计：
*   ✅ **用原始 `RGB_Pl`**（而不是 `RGB_corrected`）：保留完整的边界细节和纹理信息，用于精确边界。
*   ✅ **用 `A_spatial` 调制**：每个类别有独立的特征图，突出该类别相关区域。
*   ✅ **保留空间维度 $(H, W)$**：支持同一类别的多个目标。

**计算流程**：
1.  扩展维度：
    $$ RGB\_P_l^{expand} \rightarrow [B, 1, C, H_l, W_l] $$
    $$ A\_spatial^{expand} \rightarrow [B, \text{num\_cls}, 1, H_l, W_l] $$
2.  Element-wise 乘法：
    $$ RGB\_class\_specific\_P_l = RGB\_P_l^{expand} \times A\_spatial^{expand} $$

**输出形状**：
$$ [B, \text{num\_cls}, C, H_l, W_l] $$
*   相当于为每个类别生成了一个“类别感知的 RGB 特征”。

---

### 阶段 3：New Text - Text 更新（重新设计）

**目的**：从多尺度的 `RGB_class_specific` 中提取视觉证据，更新 Text embedding。

#### Step 1: 从每个尺度提取视觉证据
对于每个尺度 $P3, P4, P5$：
1.  **全局平均池化（聚合空间信息）**：
    $$ Y\_text\_P_l = \text{GlobalAvgPool}(RGB\_class\_specific\_P_l) \rightarrow [B, \text{num\_cls}, C] $$
2.  **投影到 text_dim**：
    $$ Y\_text\_P_l = \text{Linear}(Y\_text\_P_l) \rightarrow [B, \text{num\_cls}, \text{text\_dim}] $$

*说明*：
*   $P3$：捕捉小目标细节。
*   $P4$：捕捉中等目标。
*   $P5$：捕捉大目标和全局上下文。

#### Step 2: 多尺度融合
使用可学习权重的加权平均：
$$ w3, w4, w5 = \text{Softmax}([\alpha_3, \alpha_4, \alpha_5]) $$
$$ Y\_text\_fused = w3 \cdot Y\_text\_P3 + w4 \cdot Y\_text\_P4 + w5 \cdot Y\_text\_P5 $$

#### Step 3: 跨 Batch 聚合
$$ Y\_text\_avg = Y\_text\_fused.\text{mean}(\text{dim}=0) $$
*   **原因**：Text embedding 应该是 **batch-shared** 的，与 CLIP 设计一致，确保与 Detection Head 兼容。

#### Step 4: 可学习缩放的残差更新
$$ Text\_updated = Text\_original + scale \times Y\_text\_avg $$

*   **关键参数**：
    *   `scale`：初始化为 0。
    *   **作用**：训练初期保护 CLIP 语义，随着训练逐渐融入视觉信息。

---

## 4. Head 部分：统一的分类 + 回归

**输入**：
1.  `RGB_class_specific_Pl`: $[B, \text{num\_cls}, C, H, W]$
2.  `Text_updated`: $[\text{num\_cls}, \text{text\_dim}]$

### 分支 1：回归分支（共享头）
直接从 `RGB_class_specific` 预测边界框。

1.  **Reshape**：合并 Batch 和 Class 维度。
    $$ \text{feat} = RGB\_class\_specific.\text{view}(B \times \text{num\_cls}, C, H, W) $$
2.  **共享回归卷积**（2-3层）：
    $$ \text{feat} = \text{Conv3x3\_BN\_ReLU}(\text{feat}) $$
3.  **预测 BBox**：
    $$ \text{bbox\_pred} = \text{Conv3x3}(\text{feat}) \rightarrow [B \times \text{num\_cls}, 4, H, W] $$
4.  **Reshape 回原始维度**：
    $$ \rightarrow [B, \text{num\_cls}, 4, H, W] $$

### 分支 2：分类分支（Region-Text 对比 - YOLO-World 风格）
计算每个空间位置的特征与 Text 的相似度。

1.  **投影**：
    $$ RGB\_proj = \text{Conv1x1}(RGB\_class\_specific) \rightarrow [B, \text{num\_cls}, \text{text\_dim}, H, W] $$
2.  **归一化 (L2 Normalize)**：
    对 `RGB_proj` 和 `Text_updated` 分别进行归一化。
3.  **逐位置计算相似度（内积）**：
    $$ cls\_score = (RGB\_norm \times Text\_expanded).\text{sum}(\text{dim}=2) $$
4.  **温度系数调节**：
    $$ cls\_score = cls\_score / \tau $$
    *   $\tau = 0.07$ (可学习或固定)。

**输出形状**：
$$ [B, \text{num\_cls}, H, W] $$

---

## 5. 方案总结

### 完整数据流
1.  **输入**：RGB, IR_corrected, Text
2.  **Stage 2 (每个尺度独立)**：
    *   IR 过滤 $\rightarrow$ Text Attention $\rightarrow$ 生成 `RGB_class_specific`。
3.  **Stage 3 (跨尺度融合)**：
    *   提取视觉证据 $\rightarrow$ 多尺度加权 $\rightarrow$ Batch 聚合 $\rightarrow$ 残差更新 Text。
4.  **Head (每个尺度独立)**：
    *   **回归**：基于 `RGB_class_specific` 卷积预测。
    *   **分类**：`RGB_class_specific` 与 `Text_updated` 计算相似度。

### 关键优势
1.  **简单直接**：`RGB_class_specific` 直接输入 Head，无复杂的中间聚合。
2.  **多目标支持**：保留了 $[H, W]$ 空间维度，完美解决同类多目标区分问题。
3.  **解耦设计**：
    *   位置信息由 `RGB` 特征保留。
    *   类别语义由 `Text` 特征增强。
4.  **Text 渐进更新**：`scale=0` 初始化策略有效保护了 CLIP 的原始语义空间。
5.  **高效**：回归头参数共享，计算量低。