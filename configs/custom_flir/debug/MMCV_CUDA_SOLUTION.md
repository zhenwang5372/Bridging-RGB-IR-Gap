# mmcv CUDA 扩展编译解决方案（最终成功版本）

## 问题回顾

**错误**：`ModuleNotFoundError: No module named 'mmcv._ext'`

**原因**：mmcv 的 CUDA 扩展未正确编译。

## ✅ 成功的解决方案

### 编译命令（已验证成功）

```bash
# 1. 克隆 mmcv 源码
cd /tmp
rm -rf mmcv_build
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0 --depth 1 mmcv_build

# 2. 激活环境并编译
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /tmp/mmcv_build

# 3. 卸载现有 mmcv 并设置环境变量
pip uninstall -y mmcv
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"  # 根据 GPU 架构调整

# 4. 使用 setup.py develop 编译
python setup.py develop 2>&1 | tee /tmp/mmcv_setup_develop.log
```

### 关键成功因素

1. **使用 `setup.py develop`**：而不是 `pip install`
2. **设置正确的 GPU 架构**：`TORCH_CUDA_ARCH_LIST="8.9"` (NVIDIA L20)
3. **设置必要的环境变量**：
   - `MMCV_WITH_OPS=1`：启用 CUDA 操作
   - `FORCE_CUDA=1`：强制编译 CUDA 扩展
   - `TORCH_CUDA_ARCH_LIST`：指定 GPU 架构

### GPU 架构查询

```python
import torch
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    print(f'GPU {i}: {name} - Compute Capability: {cap}')
```

**本服务器配置**：
- GPU: NVIDIA L20 x4
- Compute Capability: 8.9
- CUDA: 12.9
- PyTorch: 2.9.1+cu129

### 为什么预编译版本失败？

预编译的 mmcv wheel 是针对 PyTorch 2.1 编译的，与 PyTorch 2.9.1 的 ABI 不兼容，导致符号未定义错误：

```
ImportError: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEE...
```

## 验证安装

### 方法 1：使用验证脚本

```bash
conda activate torch
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
python verify_installation.py
```

### 方法 2：手动验证

```python
import mmcv
print(f'mmcv 版本: {mmcv.__version__}')

from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(f'CUDA 编译版本: {get_compiling_cuda_version()}')
print(f'编译器: {get_compiler_version()}')

from mmcv.ops import nms, roi_align
print('✓ NMS 和 RoI Align 操作可用')
```

**预期输出**：
```
✓ mmcv 版本: 2.2.0
✓ CUDA 扩展已编译
  CUDA 版本: 12.9
  编译器: GCC 11.4
✓ NMS 操作可用
✓ RoI Align 操作可用
```

## 最终安装状态

| 组件 | 版本 | 状态 | 说明 |
|------|------|------|------|
| PyTorch | 2.9.1+cu129 | ✅ | 4x NVIDIA L20 |
| mmcv | 2.2.0 | ✅ | CUDA 扩展已编译 |
| mmengine | 0.10.3 | ✅ | |
| mmdet | 3.0.0 | ✅ | 已修改版本检查 |
| mmyolo | - | ✅ | 开发模式，已修改版本检查 |
| yolo_world | 0.1.0 | ✅ | 开发模式 |

## 运行训练

现在可以正常运行训练脚本了：

```bash
conda activate torch
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2

# 测试训练脚本
python tools/train.py --help

# 运行实际训练
bash configs/custom_flir/run_train_text_correction.sh
```

## 故障排查

### 如果重启后 CUDA 扩展失效

mmcv 以开发模式（`setup.py develop`）安装在 `/tmp/mmcv_build`，如果服务器重启，`/tmp` 目录可能被清空。

**解决方案**：重新执行编译命令，或将源码移到永久目录：

```bash
# 移动到永久位置
mv /tmp/mmcv_build /home/ssd1/users/wangzhen01/mmcv_build

# 重新链接
cd /home/ssd1/users/wangzhen01/mmcv_build
python setup.py develop
```

### 如果需要在其他服务器安装

1. 检查 GPU 架构：`python -c "import torch; print(torch.cuda.get_device_capability(0))"`
2. 设置对应的 `TORCH_CUDA_ARCH_LIST`
3. 执行相同的编译命令

## 编译日志

完整编译日志保存在：`/tmp/mmcv_setup_develop.log`

## 参考命令历史

```bash
# 完整一键编译命令
source /home/disk1/users/linsong/miniconda3/etc/profile.d/conda.sh && \
conda activate torch && \
cd /tmp && \
rm -rf mmcv_build && \
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0 --depth 1 mmcv_build && \
cd /tmp/mmcv_build && \
pip uninstall -y mmcv && \
export MMCV_WITH_OPS=1 && \
export FORCE_CUDA=1 && \
export TORCH_CUDA_ARCH_LIST="8.9" && \
python setup.py develop 2>&1 | tee /tmp/mmcv_setup_develop.log | tail -100
```

## 日期

- 问题发现：2026-01-14
- 解决日期：2026-01-14
- 编译时间：约 5-10 分钟

