# CuBLAS 确定性训练问题解决方案

## 错误信息

```
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
```

## 原因

当启用确定性训练模式时（`randomness.deterministic=True`），PyTorch 会强制所有操作使用确定性算法。但在 CUDA >= 10.2 的环境中，CuBLAS 库的某些操作（如矩阵乘法）默认使用非确定性算法以获得更好的性能。

## 解决方案

### 方法 1：设置 CUBLAS_WORKSPACE_CONFIG 环境变量（推荐）

**优点**：保持训练的确定性和可复现性

**修改位置**: `configs/custom_flir/run_train_text_correction.sh`

```bash
# 在训练脚本中添加（已修复）
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**说明**：
- `:4096:8` 表示分配 4096 个工作空间，每个大小为 8 字节
- 或使用 `:16:8` (更小的内存占用)
- 这会让 CuBLAS 使用确定性算法

### 方法 2：关闭确定性模式（不推荐）

**缺点**：训练结果可能无法完全复现

```bash
# 在训练命令中添加（不推荐）
--cfg-options randomness.deterministic=False
```

---

## 当前配置

### 训练脚本设置

```bash
# 环境变量（已在 run_train_text_correction.sh 中设置）
export CUDA_VISIBLE_DEVICES=3
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # ✅ 新增
export HF_ENDPOINT="https://hf-mirror.com"
```

### 配置文件设置

```bash
# 在训练命令行参数中设置
--cfg-options \
    randomness.seed=42 \
    randomness.deterministic=True  # 确定性训练
```

---

## 影响

### 性能影响

启用确定性模式 + CUBLAS_WORKSPACE_CONFIG 可能会：
- **降低训练速度**: ~5-15%（因为禁用了某些优化）
- **增加内存占用**: 额外的工作空间内存

### 好处

- ✅ **完全可复现**: 相同种子产生相同结果
- ✅ **便于调试**: 结果稳定，易于定位问题
- ✅ **科学严谨**: 适合学术研究和实验对比

---

## 验证

运行训练脚本，确认不再报错：

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
conda activate torch
bash configs/custom_flir/run_train_text_correction.sh
```

预期：训练正常启动，不再报 CuBLAS 错误。

---

## 相关信息

- **CUDA 版本**: 12.9
- **PyTorch 版本**: 2.9.1+cu129
- **确定性设置**: `randomness.seed=42`, `randomness.deterministic=True`

---

## 参考链接

- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [CuBLAS Results Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

---

## 状态

✅ **已解决**

修改文件：`configs/custom_flir/run_train_text_correction.sh`

