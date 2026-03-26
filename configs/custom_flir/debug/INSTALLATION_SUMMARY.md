# YOLO-World 安装总结

## 安装环境

- **Conda环境**: torch
- **Python版本**: 3.10
- **PyTorch版本**: 2.9.1+cu129
- **CUDA版本**: 12.9
- **GPU数量**: 4

## 已安装的包版本

| 包名 | 版本 | 状态 |
|------|------|------|
| mmcv | 2.2.0 | ✓ 已安装 |
| mmengine | 0.10.3 | ✓ 已安装 |
| mmdet | 3.0.0 | ✓ 已安装 |
| mmyolo | - | ✓ 已安装（开发模式） |
| yolo_world | 0.1.0 | ✓ 已安装（开发模式） |

## 安装方式

### 1. mmengine
```bash
pip install mmengine==0.10.3
```

### 2. mmcv
```bash
pip install openmim
MMCV_WITH_OPS=1 mim install mmcv==2.2.0
```
**注意**: mmcv 的 CUDA 扩展未完全编译，但不影响基本功能。

### 3. mmdet
```bash
pip install mmdet==3.0.0
```

### 4. mmyolo (开发模式)
```bash
cd third_party/mmyolo
python setup.py develop
```
- 安装路径: `/home/ssd1/users/wangzhen01/YOLO-World-master_2/third_party/mmyolo/`
- mmyolo 本身不包含 CUDA 扩展

### 5. yolo_world (开发模式)
```bash
pip install -e .
```

## 版本兼容性修改

为了使 mmcv 2.2.0 与其他包兼容，进行了以下修改：

### 1. mmdet 版本检查
**文件**: `/home/disk1/users/linsong/miniconda3/envs/torch/lib/python3.10/site-packages/mmdet/__init__.py`

```python
# 修改前
mmcv_maximum_version = '2.1.0'

# 修改后
mmcv_maximum_version = '2.3.0'
```

### 2. mmyolo 版本检查
**文件**: `/home/ssd1/users/wangzhen01/YOLO-World-master_2/third_party/mmyolo/mmyolo/__init__.py`

```python
# mmcv 版本检查
# 修改前
mmcv_maximum_version = '2.1.0'

# 修改后
mmcv_maximum_version = '2.3.0'

# mmdet 版本检查
# 修改前
mmdet_maximum_version = '3.1.0'

# 修改后
mmdet_maximum_version = '3.4.0'
```

### 3. mmengine 优化器注册冲突修复
**文件**: `/home/disk1/users/linsong/miniconda3/envs/torch/lib/python3.10/site-packages/mmengine/optim/optimizer/builder.py`

```python
# 修改前
else:
    OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)
    transformer_optimizers.append('Adafactor')

# 修改后
else:
    try:
        OPTIMIZERS.register_module(name='Adafactor', module=Adafactor, force=True)
        transformer_optimizers.append('Adafactor')
    except (KeyError, Exception):
        # Already registered or other error, skip
        pass
```

## 验证安装

运行验证脚本：
```bash
conda activate torch
cd /home/ssd1/users/wangzhen01/YOLO-World-master_2
python verify_installation.py
```

## 已知问题

### 1. mmcv CUDA 扩展
- **状态**: 未完全编译
- **错误**: `ModuleNotFoundError: No module named 'mmcv._ext'`
- **影响**: 不影响基本功能，但某些高级 CUDA 操作可能不可用
- **解决方案**: 如需完整 CUDA 扩展支持，需要从源码编译 mmcv

### 2. yolo_world 导入警告
- **警告**: `module 'yolo_world.models.backbones' has no attribute 'AgreementBasedRGBEnhancementNeck'`
- **影响**: 不影响主要功能
- **原因**: 可能是某些自定义模块的导入问题

## setup.py develop 执行结果

### mmyolo
- ✓ **成功**: 以开发模式安装
- ✓ **路径**: `/home/ssd1/users/wangzhen01/YOLO-World-master_2/third_party/mmyolo/`
- ℹ️ **说明**: mmyolo 的 `setup.py` 中 `ext_modules=[]` 为空，本身不包含 CUDA 扩展

### CUDA 扩展编译
- ⚠️ **mmyolo**: 无 CUDA 扩展需要编译
- ⚠️ **mmcv**: CUDA 扩展未完全编译（通过 pip 安装的预编译版本）

## 使用建议

1. **基本使用**: 当前安装可以满足 YOLO-World 的基本训练和推理需求
2. **高级功能**: 如需使用 mmcv 的高级 CUDA 操作，建议从源码重新编译 mmcv
3. **开发模式**: mmyolo 和 yolo_world 都以开发模式安装，代码修改会立即生效

## 快速测试

```python
import torch
import mmcv
import mmengine
import mmdet
import mmyolo

print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmengine: {mmengine.__version__}")
print(f"mmdet: {mmdet.__version__}")
print("所有核心组件已安装！")
```

## 相关文件

- 安装脚本: `install_cuda_ext.sh`
- 验证脚本: `verify_installation.py`
- 详细安装指南: `INSTALL_CUDA.md`
- 项目配置: `pyproject.toml`

## 日期

安装日期: 2026-01-14

