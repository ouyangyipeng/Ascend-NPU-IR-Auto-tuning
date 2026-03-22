#!/bin/bash
# 应用优化补丁到AscendNPU_IR
# 使用方法: ./apply_patches.sh <ascendnpu-ir路径>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"

# 默认路径
TARGET_DIR="${1:-$SCRIPT_DIR/ascendnpu-ir}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 $TARGET_DIR 不存在"
    echo "使用方法: $0 <ascendnpu-ir路径>"
    exit 1
fi

echo "=== 应用优化补丁到 $TARGET_DIR ==="

# 1. 复制新增的OptimizedAutoSchedule.cpp
echo "[1/5] 复制 OptimizedAutoSchedule.cpp..."
cp "$PATCHES_DIR/OptimizedAutoSchedule.cpp" \
   "$TARGET_DIR/bishengir/lib/Dialect/HFusion/Transforms/"

# 2. 复制修改后的AnyPBRSchedule.cpp
echo "[2/5] 复制 AnyPBRSchedule.cpp..."
cp "$PATCHES_DIR/AnyPBRSchedule.cpp" \
   "$TARGET_DIR/bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/"

# 3. 复制修改后的Passes.td
echo "[3/5] 复制 Passes.td..."
cp "$PATCHES_DIR/Passes.td" \
   "$TARGET_DIR/bishengir/include/bishengir/Dialect/HFusion/Transforms/"

# 4. 复制修改后的Passes.h
echo "[4/5] 复制 Passes.h..."
cp "$PATCHES_DIR/Passes.h" \
   "$TARGET_DIR/bishengir/include/bishengir/Dialect/HFusion/Transforms/"

# 5. 复制修改后的CMakeLists.txt
echo "[5/5] 复制 CMakeLists.txt..."
cp "$PATCHES_DIR/CMakeLists.txt" \
   "$TARGET_DIR/bishengir/lib/Dialect/HFusion/Transforms/"

echo ""
echo "=== 补丁应用完成 ==="
echo ""
echo "下一步操作:"
echo "1. cd $TARGET_DIR"
echo "2. ./build-tools/build.sh -o ./build --build-type Release --apply-patches"
echo "3. 等待编译完成"
echo "4. 运行测试: cd build && ctest --output-on-failure"
echo ""
echo "验证Pass注册:"
echo "./build/bin/bishengir-opt --help | grep optimized-auto-schedule"