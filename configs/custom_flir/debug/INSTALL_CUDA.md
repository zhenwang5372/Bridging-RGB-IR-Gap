# CUDA扩展编译安装指南

本指南说明如何在torch环境中使用指定版本安装并编译CUDA扩展。

## 环境要求

- Python 3.7+
- PyTorch (已安装并配置CUDA)
- CUDA工具链

## 版本要求

- **mmcv**: v2.2.0
- **mmengine**: 0.10.3
- **mmdet**: 3.0.0

## 安装步骤

### 方法1: 使用自动化脚本（推荐）

```bash
# 在torch环境中执行
bash install_cuda_ext.sh
```

### 方法2: 手动安装

#### 步骤1: 设置环境变量

```bash
export FORCE_CUDA="1"
export MMCV_WITH_OPS=1
```

#### 步骤2: 安装基础依赖

```bash
pip install --upgrade pip setuptools wheel
```

#### 步骤3: 安装 mmengine

```bash
pip install mmengine==0.10.3
```

#### 步骤4: 安装 mmcv (带CUDA扩展)

**选项A: 使用 mim (推荐)**

```bash
pip install openmim
MMCV_WITH_OPS=1 mim install mmcv==2.2.0
```

**选项B: 从源码编译**

```bash
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0
cd mmcv
MMCV_WITH_OPS=1 pip install -e . -v
cd ..
```

#### 步骤5: 安装 mmdet

```bash
pip install mmdet==3.0.0
```

#### 步骤6: 编译安装 mmyolo (使用 setup.py develop)

```bash
cd third_party/mmyolo
python setup.py develop
cd ../..
```

#### 步骤7: 安装 YOLO-World 主项目

```bash
pip install -e .
```

## 验证安装

```python
# 验证版本
import mmcv
import mmengine
import mmdet
print(f"mmcv版本: {mmcv.__version__}")
print(f"mmengine版本: {mmengine.__version__}")
print(f"mmdet版本: {mmdet.__version__}")

# 验证CUDA扩展
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(f"CUDA扩展编译成功")
```

## 常见问题

### 1. CUDA扩展编译失败

- 确保已设置 `MMCV_WITH_OPS=1` 环境变量
- 检查CUDA和PyTorch版本兼容性
- 确保安装了CUDA开发工具包

### 2. 版本冲突

如果遇到版本冲突，建议使用虚拟环境：

```bash
conda create -n yolo_world python=3.8
conda activate yolo_world
# 然后按照上述步骤安装
```

### 3. 编译时间较长

CUDA扩展编译可能需要较长时间（10-30分钟），请耐心等待。

## 注意事项

- 确保在安装mmcv之前设置 `MMCV_WITH_OPS=1` 环境变量
- `setup.py develop` 会以开发模式安装，代码修改会立即生效
- 如果使用conda环境，确保CUDA路径正确配置

