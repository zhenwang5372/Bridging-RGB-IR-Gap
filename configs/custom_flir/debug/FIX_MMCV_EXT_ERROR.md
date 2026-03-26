# 修复 mmcv._ext 错误

## 错误描述

```
ModuleNotFoundError: No module named 'mmcv._ext'
```

## 错误原因

**根本原因**：mmcv 的 CUDA 扩展未正确编译或安装。

**详细分析**：
1. 当前通过 `mim install mmcv==2.2.0` 安装的可能是 lite 版本或不完整的预编译版本
2. mmdet 需要使用 mmcv 的 CUDA 操作（如 NMS、RoI pooling 等）
3. 这些操作依赖于 `mmcv._ext` 模块，但该模块不存在

**触发路径**：
```
train.py 
  → mmyolo.utils 
    → mmyolo.models 
      → mmdet.models.backbones 
        → mmdet.models.layers.bbox_nms 
          → mmcv.ops.nms 
            → mmcv._ext (❌ 不存在)
```

## 解决方案

### 🚀 方案 1：使用预编译的 mmcv（推荐，快速）

**优点**：
- 安装速度快（2-5分钟）
- 不需要本地编译环境
- 稳定性好

**步骤**：

```bash
# 激活 torch 环境
conda activate torch

# 运行修复脚本
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
bash fix_mmcv_prebuilt.sh
```

**手动步骤**：

```bash
# 1. 卸载现有 mmcv
pip uninstall -y mmcv mmcv-lite mmcv-full

# 2. 查看 PyTorch 和 CUDA 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 3. 安装对应版本的预编译 mmcv
# 对于 CUDA 12.x (如 12.9)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 或使用 openmim (推荐)
pip install openmim
mim install "mmcv==2.2.0"

# 4. 验证安装
python -c "from mmcv.ops import nms; print('✓ mmcv CUDA 扩展可用')"
```

### 🔧 方案 2：从源码编译 mmcv（彻底解决）

**优点**：
- 完全匹配本地环境
- 包含所有 CUDA 操作
- 可以自定义编译选项

**缺点**：
- 编译时间长（10-30分钟）
- 需要完整的 CUDA 工具链

**步骤**：

```bash
# 激活 torch 环境
conda activate torch

# 运行编译脚本
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
bash fix_mmcv_cuda.sh
```

**手动步骤**：

```bash
# 1. 确保 CUDA 工具链可用
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version  # 验证 nvcc 可用

# 2. 卸载现有 mmcv
pip uninstall -y mmcv mmcv-lite mmcv-full

# 3. 克隆源码
cd /tmp
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0 --depth 1
cd mmcv

# 4. 设置编译环境变量
export FORCE_CUDA="1"
export MMCV_WITH_OPS=1

# 5. 编译安装（需要 10-30 分钟）
pip install -e . -v

# 6. 验证
python -c "from mmcv.ops import get_compiling_cuda_version; print(f'CUDA: {get_compiling_cuda_version()}')"
```

### ⚡ 方案 3：使用 mmcv-lite（不推荐，功能受限）

如果只是测试，不需要完整 CUDA 操作，可以临时使用：

```bash
pip uninstall -y mmcv
pip install mmcv-lite==2.2.0
```

**注意**：这会导致某些功能不可用（如自定义 CUDA ops）。

## 验证安装

运行以下命令验证 mmcv CUDA 扩展是否正确安装：

```bash
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
python verify_installation.py
```

或者手动验证：

```python
import mmcv
print(f"mmcv 版本: {mmcv.__version__}")

# 验证 CUDA 扩展
from mmcv.ops import nms, roi_align
print("✓ NMS 操作可用")
print("✓ RoI Align 操作可用")

# 验证编译版本
from mmcv.ops import get_compiling_cuda_version
print(f"✓ CUDA 编译版本: {get_compiling_cuda_version()}")
```

## 再次运行训练

修复后，重新运行训练脚本：

```bash
conda activate torch
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
bash configs/custom_flir/run_train_text_correction.sh
```

## 常见问题

### Q1: 预编译版本找不到匹配的 wheel？

**解决方案**：
1. 检查 PyTorch 和 CUDA 版本是否匹配
2. 尝试稍低版本的 CUDA wheel（如用 cu121 代替 cu129）
3. 如果都不行，使用方案 2 从源码编译

### Q2: 编译时出现 "nvcc not found"？

**解决方案**：
```bash
# 查找 CUDA 安装路径
find /usr/local -name "nvcc" 2>/dev/null

# 设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1  # 根据实际路径调整
export PATH=$CUDA_HOME/bin:$PATH
```

### Q3: 编译时内存不足？

**解决方案**：
```bash
# 限制并行编译数
export MAX_JOBS=4
pip install -e . -v
```

### Q4: 编译后仍然报错？

**解决方案**：
1. 完全卸载 mmcv：`pip uninstall mmcv mmcv-lite mmcv-full`
2. 清理缓存：`pip cache purge`
3. 重新安装
4. 检查是否有多个 Python 环境

## 相关文件

- `fix_mmcv_prebuilt.sh` - 预编译版本安装脚本
- `fix_mmcv_cuda.sh` - 源码编译脚本
- `verify_installation.py` - 验证脚本
- `INSTALLATION_SUMMARY.md` - 完整安装总结

## 推荐执行顺序

1. ✅ **首先尝试方案 1**（预编译版本，5分钟）
2. ❌ 如果方案 1 失败，尝试方案 2（源码编译，30分钟）
3. ✅ 验证安装
4. ✅ 运行训练脚本

---

**注意**：修复后需要重新验证 mmdet 和 mmyolo 的版本兼容性（已在之前的安装中修复）。

