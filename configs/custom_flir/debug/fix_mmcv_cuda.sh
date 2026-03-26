#!/bin/bash
# 从源码编译 mmcv 的 CUDA 扩展

set -e

echo "=========================================="
echo "重新编译 mmcv 的 CUDA 扩展"
echo "=========================================="

# 设置环境变量
export FORCE_CUDA="1"
export MMCV_WITH_OPS=1
export CUDA_HOME=/usr/local/cuda

echo "当前 CUDA 环境："
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version || echo "警告: nvcc 不可用"

echo ""
echo "步骤 1: 卸载现有的 mmcv"
pip uninstall -y mmcv mmcv-lite mmcv-full 2>/dev/null || true

echo ""
echo "步骤 2: 克隆 mmcv 源码（v2.2.0）"
cd /tmp
rm -rf mmcv
git clone https://github.com/open-mmlab/mmcv.git -b v2.2.0 --depth 1
cd mmcv

echo ""
echo "步骤 3: 编译并安装 mmcv（带 CUDA 扩展）"
echo "这可能需要 10-30 分钟，请耐心等待..."
MMCV_WITH_OPS=1 pip install -e . -v

echo ""
echo "步骤 4: 验证安装"
python -c "
import mmcv
print(f'mmcv 版本: {mmcv.__version__}')
try:
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(f'✓ CUDA 扩展已编译')
    print(f'  CUDA 版本: {get_compiling_cuda_version()}')
    print(f'  编译器版本: {get_compiler_version()}')
except Exception as e:
    print(f'✗ CUDA 扩展验证失败: {e}')
    exit(1)
"

echo ""
echo "=========================================="
echo "mmcv CUDA 扩展编译完成！"
echo "=========================================="

