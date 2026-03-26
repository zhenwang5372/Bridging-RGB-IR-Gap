# Text-guided IR Correction V2 - Index Out of Bounds 修复

## 日期
2026-01-15

---

## 1. 问题描述

### 错误信息

训练启动后立即出现 CUDA Index Out of Bounds 错误：

```
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator(): block: [0,0,0], thread: [115,0,0]
```

**错误类型**: CUDA 索引越界（Index Out of Bounds）

---

## 2. 根本原因分析

### 2.1 文本特征维度动态变化

在 YOLO-World 训练中，`txt_feats` 的维度是 **动态的**：

```python
txt_feats: [B, N, text_dim]
```

其中：
- `B`: Batch size
- **`N`: 实际的类别数（动态）**
- `text_dim`: 文本特征维度（512）

**关键问题**: `N` 不是固定的 4 类，而是会根据 `RandomLoadText` 的采样包含负样本！

### 2.2 配置文件中的类别数

```python
# configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV2.py
num_classes = 4  # FLIR: car, person, bicycle, dog
num_training_classes = 4

# RandomLoadText 会采样负样本
dict(type='RandomLoadText',
     num_neg_samples=(num_classes, num_classes),  # ← 这里！
     max_num_samples=num_training_classes,
     padding_to_max=True,
     padding_value=''),
```

`num_neg_samples=(4, 4)` 意味着会随机采样 4 个负样本类别，所以：

**实际 txt_feats 的类别数 N = 4 (正样本) + 4 (负样本) = 8**

### 2.3 V2 代码的问题

**原始代码**（text_guided_ir_correction_v2.py）:

```python
def __init__(self, ..., num_classes: int = 4, ...):
    self.num_classes = num_classes  # 固定为 4

def forward(self, ..., txt_feats: torch.Tensor, ...):
    # txt_feats: [B, N, text_dim]
    # 假设 N = self.num_classes = 4
    Q = self.text_query_proj(txt_feats)  # [B, N, d_k]
    # 但实际 N = 8！导致索引越界
```

**为什么会导致索引越界？**

虽然 `text_query_proj` 是 `nn.Linear(text_dim, d_k)`，不会直接越界，但后续的操作（如 `einsum`, `view`, `reshape`）会假设 `N=4`，导致维度不匹配。

---

## 3. 解决方案

### 3.1 修改策略

**核心思想**: 不要假设 `txt_feats` 的类别数是固定的，而是**动态获取实际的类别数**。

### 3.2 具体修改

**修改位置**: `yolo_world/models/necks/text_guided_ir_correction/text_guided_ir_correction_v2.py`

#### 修改 1: 添加注释说明

```python
class SingleLevelTextGuidedCorrectionV2(nn.Module):
    def __init__(
        self,
        rgb_channels: int,
        ir_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,  # ← 仅作为配置参考
    ):
        super().__init__()
        self.num_classes = num_classes  # 仅作为参考，实际会动态适应
```

#### 修改 2: 动态获取类别数

```python
def forward(
    self,
    x_rgb: torch.Tensor,
    x_ir: torch.Tensor,
    txt_feats: torch.Tensor,
    alpha: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        txt_feats: [B, N, text_dim] (N是实际的类别数，可能包含负样本)
    """
    B, C_rgb, H, W = x_rgb.shape
    _, C_ir, _, _ = x_ir.shape
    N_actual = txt_feats.size(1)  # ← 动态获取实际的类别数
    
    # 后续所有操作都使用 N_actual 而不是 self.num_classes
    Q = self.text_query_proj(txt_feats)  # [B, N_actual, d_k] ✅
```

### 3.3 为什么V1没有这个问题？

查看 V1 代码（text_guided_ir_correction_v1.py），虽然也定义了 `num_classes`，但：

1. V1 的所有操作都是**动态的**，基于 `txt_feats` 的实际形状
2. V1 没有使用固定的 `num_classes` 进行任何 `view` 或 `reshape` 操作

---

## 4. 验证修复

### 4.1 测试命令

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
bash configs/custom_flir/run_train_text_correction.sh
```

### 4.2 预期结果

✅ **修复前**: 训练启动后立即出现 CUDA Index Out of Bounds 错误并 Abort

✅ **修复后**: 训练正常启动，模型正常前向传播，Loss 正常计算

---

## 5. 教训与最佳实践

### 5.1 动态维度处理

在处理文本特征时，**不要假设类别数是固定的**：

```python
# ❌ 错误做法
N = self.num_classes
Q = txt_feats.view(B, N, -1)  # 假设固定类别数

# ✅ 正确做法
N_actual = txt_feats.size(1)  # 动态获取
Q = txt_feats  # 保持原始维度
```

### 5.2 RandomLoadText 的影响

`RandomLoadText` 的 `num_neg_samples` 会影响 `txt_feats` 的维度：

```python
dict(type='RandomLoadText',
     num_neg_samples=(4, 4),  # 采样 4 个负样本
     max_num_samples=8,        # 总共 8 个类别
     padding_to_max=True,
     padding_value=''),
```

**实际类别数 = 正样本数 + 负样本数**

### 5.3 调试技巧

当遇到 CUDA Index Out of Bounds 错误时：

1. **检查所有涉及索引的操作**: `view()`, `reshape()`, `[]`, `einsum()`
2. **打印实际的 tensor shape**: `print(f"txt_feats.shape: {txt_feats.shape}")`
3. **对比配置参数和实际维度**: 配置中的 `num_classes` vs 实际的 `txt_feats.size(1)`

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `yolo_world/models/necks/text_guided_ir_correction/text_guided_ir_correction_v2.py` | V2 核心实现（已修复） |
| `configs/custom_flir/yolow_v2_rgb_ir_flir_text_correctionV2.py` | V2 配置文件 |
| `configs/custom_flir/run_train_text_correction.sh` | 训练脚本 |

---

## 7. 总结

**问题**: V2 代码假设 `txt_feats` 的类别数固定为配置中的 `num_classes`，但实际上由于 `RandomLoadText` 采样负样本，类别数会动态变化。

**解决**: 使用 `txt_feats.size(1)` 动态获取实际的类别数，而不是依赖固定的配置参数。

**经验**: 在处理动态维度的 tensor 时，始终从 tensor 本身获取维度信息，而不是依赖外部配置。

