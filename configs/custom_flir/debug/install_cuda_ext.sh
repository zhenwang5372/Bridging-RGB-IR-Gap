#!/bin/bash
# 安装脚本：使用指定版本并编译CUDA扩展
# mmcv v2.2.0, mmengine==0.10.3, mmdet==3.0

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始安装 YOLO-World 及 CUDA 扩展"
echo "=========================================="

# 设置环境变量以启用CUDA扩展编译
export FORCE_CUDA="1"
export MMCV_WITH_OPS=1

# 检查是否在torch环境中
if ! python -c "import torch" 2>/dev/null; then
    echo "错误: 未检测到torch环境，请先激活torch环境"
    exit 1
fi

echo "检测到torch环境:"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"
fi

echo ""
echo "=========================================="
echo "步骤 1: 安装基础依赖"
echo "=========================================="
pip install --upgrade pip setuptools wheel

echo ""
echo "=========================================="
echo "步骤 2: 安装 mmengine==0.10.3"
echo "=========================================="
pip install mmengine==0.10.3

echo ""
echo "=========================================="
echo "步骤 3: 安装 mmcv v2.2.0 (带CUDA扩展)"
echo "=========================================="
# 方法1: 使用mim安装（推荐，会自动编译CUDA扩展）
if command -v mim &> /dev/null; then
    echo "使用 mim 安装 mmcv v2.2.0..."
    MMCV_WITH_OPS=1 mim install mmcv==2.2.0
else
    echo "mim 未安装，先安装 openmim..."
    pip install openmim
    MMCV_WITH_OPS=1 mim install mmcv==2.2.0
fi

echo ""
echo "=========================================="
echo "步骤 4: 安装 mmdet==3.0.0"
echo "=========================================="
pip install mmdet==3.0.0

echo ""
echo "=========================================="
echo "步骤 5: 编译安装 mmyolo (CUDA扩展)"
echo "=========================================="
cd third_party/mmyolo

# 确保在正确的目录
if [ ! -f "setup.py" ]; then
    echo "错误: 未找到 setup.py 文件"
    exit 1
fi

# 使用 setup.py develop 安装并编译CUDA扩展
echo "使用 setup.py develop 编译安装..."
python setup.py develop

cd ../..

echo ""
echo "=========================================="
echo "步骤 6: 安装 YOLO-World 主项目"
echo "=========================================="
pip install -e .

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "验证安装:"
python -c "import mmcv; print(f'✓ mmcv版本: {mmcv.__version__}')" 2>/dev/null || echo "✗ mmcv导入失败"
python -c "import mmengine; print(f'✓ mmengine版本: {mmengine.__version__}')" 2>/dev/null || echo "✗ mmengine导入失败"
python -c "import mmdet; print(f'✓ mmdet版本: {mmdet.__version__}')" 2>/dev/null || echo "✗ mmdet导入失败"
python -c "import mmyolo; print(f'✓ mmyolo安装成功')" 2>/dev/null || echo "✗ mmyolo导入失败"
python -c "import yolo_world; print(f'✓ yolo_world安装成功')" 2>/dev/null || echo "✗ yolo_world导入失败"

echo ""
echo "检查CUDA扩展:"
python -c "from mmcv.ops import get_compiling_cuda_version, get_compiler_version; print(f'✓ CUDA扩展已编译')" 2>/dev/null || echo "⚠ 无法验证CUDA扩展，但可能已安装"

echo ""
echo "=========================================="
echo "安装脚本执行完成！"
echo "=========================================="

