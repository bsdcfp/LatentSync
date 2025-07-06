#!/bin/bash

# torch.compile 缓存配置脚本
# 使用方法: ./scripts/run_with_cache.sh [cache_dir]

# 设置缓存目录
CACHE_DIR=${1:-"/data/torch_compile_cache"}

echo "🚀 Setting up torch.compile cache..."
echo "Cache directory: $CACHE_DIR"

# 创建缓存目录
mkdir -p "$CACHE_DIR"
mkdir -p "$CACHE_DIR/triton"
mkdir -p "$CACHE_DIR/torch_extensions"

# 设置环境变量
export TORCH_COMPILE_CACHE_DIR="$CACHE_DIR"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"

# 可选：设置其他torch相关环境变量
export TOKENIZERS_PARALLELISM="false"
export TORCH_LOGS=""  # 可设置为 "recompiles" 查看重编译信息

echo "✅ Environment variables set:"
echo "  TORCH_COMPILE_CACHE_DIR=$TORCH_COMPILE_CACHE_DIR"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

# 检查缓存状态
echo ""
echo "📊 Checking cache status..."
python torch_compile_utils.py --cache-dir "$CACHE_DIR" info

echo ""
echo "🎯 Ready to run your application!"
echo "   Example: python test/local_test.py --test-num 1"

# 如果传入了额外参数，执行命令
if [ $# -gt 1 ]; then
    echo ""
    echo "🏃 Running command: ${@:2}"
    "${@:2}"
fi 