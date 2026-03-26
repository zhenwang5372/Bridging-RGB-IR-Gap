#!/bin/bash
# 强制从源码编译 mmcv 的 CUDA 扩展（非 editable 模式）

set -e

echo "=========================================="
echo "强制编译 mmcv 的 CUDA 扩展"
echo "=========================================="

# 设置环境变量
export FORCE_CUDA="1"
export MMCV_WITH_OPS=1
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=8

echo "环境变量："
echo "  FORCE_CUDA=$FORCE_CUDA"
echo "  MMCV_WITH_OPS=$MMCV_WITH_OPS"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  MAX_JOBS=$MAX_JOBS"

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    echo ""
    echo "CUDA 信息："
    nvcc --version | head -4
else
    echo "警告: nvcc 不可用，但会尝试继续"
fi

echo ""
echo "步骤 1: 卸载现有的 mmcv"
pip uninstall -y mmcv mmcv-lite mmcv-full 2>/dev/null || true

echo ""
echo "步骤 2: 下载 mmcv 源码"
cd /tmp
rm -rf mmcv
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0 --depth 1
cd mmcv

echo ""
echo "步骤 3: 编译并安装（非 editable 模式）"
echo "编译时间约 10-30 分钟，请耐心等待..."
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v . 

echo ""
echo "步骤 4: 验证安装"
python -c "
import mmcv
print(f'✓ mmcv 版本: {mmcv.__version__}')

# 检查 _ext 模块
import os
mmcv_path = os.path.dirname(mmcv.__file__)
ext_file = os.path.join(mmcv_path, '_ext.cpython-310-x86_64-linux-gnu.so')
if os.path.exists(ext_file):
    print(f'✓ _ext.so 文件存在: {ext_file}')
else:
    print(f'✗ _ext.so 文件不存在')
    exit(1)

# 尝试导入 CUDA 操作
try:
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(f'✓ CUDA 扩展已编译')
    print(f'  CUDA 版本: {get_compiling_cuda_version()}')
    print(f'  编译器: {get_compiler_version()}')
except Exception as e:
    print(f'✗ CUDA 扩展验证失败: {e}')
    exit(1)

# 测试基本操作
from mmcv.ops import nms
print(f'✓ NMS 操作可用')
"

echo ""
echo "=========================================="
echo "mmcv CUDA 扩展编译完成！"
echo "=========================================="

