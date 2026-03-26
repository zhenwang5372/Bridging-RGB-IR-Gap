#!/bin/bash
# 使用预编译的 mmcv wheel 安装（包含 CUDA 扩展）

set -e

echo "=========================================="
echo "安装预编译的 mmcv (带 CUDA 扩展)"
echo "=========================================="

# 检查 PyTorch 和 CUDA 版本
echo "检测环境信息..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "unknown")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")

echo "PyTorch 版本: $TORCH_VERSION"
echo "CUDA 版本: $CUDA_VERSION"

echo ""
echo "步骤 1: 卸载现有的 mmcv"
pip uninstall -y mmcv mmcv-lite mmcv-full 2>/dev/null || true

echo ""
echo "步骤 2: 安装预编译的 mmcv"

# 根据 CUDA 版本选择合适的 index URL
if [[ "$CUDA_VERSION" == "12.9" ]] || [[ "$CUDA_VERSION" == 12.* ]]; then
    # CUDA 12.x 使用 cu121 的预编译包（通常向后兼容）
    echo "使用 CUDA 12.1 的预编译包..."
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "使用 CUDA 11.x 的预编译包..."
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
else
    echo "警告：无法确定 CUDA 版本，尝试使用默认源..."
    pip install mmcv==2.2.0
fi

echo ""
echo "步骤 3: 验证安装"
python -c "
import mmcv
print(f'✓ mmcv 版本: {mmcv.__version__}')
try:
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(f'✓ CUDA 扩展已编译')
    print(f'  CUDA 版本: {get_compiling_cuda_version()}')
    print(f'  编译器: {get_compiler_version()}')
except Exception as e:
    print(f'⚠ CUDA 扩展验证: {e}')
    print('尝试导入基本操作...')
    from mmcv.ops import nms
    print('✓ 基本操作可用')
"

echo ""
echo "=========================================="
echo "mmcv 安装完成！"
echo "=========================================="

