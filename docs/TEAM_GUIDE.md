# 团队指南：AscendNPU_IR NPU自动融合代码生成与调优比赛

> 本文档面向参赛队员，介绍比赛赛题、项目结构、当前进度和后续工作。

---

## 一、比赛赛题理解

### 1.1 赛题核心要求

**比赛名称**：基于AscendNPU_IR的NPU自动融合代码生成与调优

**核心任务**：
- 在AscendNPU_IR编译器中增改Pass（主要是HFusion方言的Pass）
- 实现融合算子的自动代码生成和调优
- 确保算子功能正确且通过误差检验
- 最大化计算性能

**关键约束**：
- ✅ 仅限于AscendNPU_IR层面增改HFusion方言的Pass
- ✅ 面向昇腾A2/A3平台
- ✅ 可使用基于专家经验的调度/切分策略或Auto-tuning
- ❌ 不得修改其他层面的代码

### 1.2 Ascend NPU IR vs 普通LLVM IR

| 特性 | LLVM IR | Ascend NPU IR |
|------|---------|---------------|
| **目标平台** | 通用CPU | 昇腾NPU（A2/A3） |
| **抽象层次** | 低级中间表示 | 多级中间表示（MLIR框架） |
| **方言体系** | 单一IR | 多方言（HFusion、HACC、HIVM等） |
| **优化重点** | 通用优化 | NPU特定优化（UB内存、多核调度） |
| **算子融合** | 不支持 | 原生支持多维融合抽象 |
| **硬件感知** | 弱 | 强（针对昇腾硬件特性设计） |

**AscendNPU_IR的核心优势**：
1. **HFusion方言**：专门为算子融合设计的中间表示
2. **AutoSchedule框架**：自动调度和切分策略
3. **硬件感知优化**：针对UB（Unified Buffer）和多核的优化

---

## 二、评分标准详解

### 2.1 功能测试评分（占40%）

**公式**：
```
acc_score = (1/T) × Σ A_i
```

其中：
- `T` = 公开用例集 + 隐藏用例集的总数
- `A_i` = 1（测试通过）或 0（测试失败）

**测试点定义**：
- 静态shape融合算子：一个算子 = 一个测试点
- 动态shape融合算子：一组运行时静态shape = 一个测试点

**功能失败判定**：
- 算子编译失败 → 0分
- 编译成功但运行时错误 → 0分
- 编译成功但精度不达标 → 0分

**精度标准**（详见附录二）：
- 整数计算：二进制对比一致
- 浮点数计算：相对误差/绝对误差在阈值内

### 2.2 性能测试评分（占60%）

**性能加速比定义**：
```
s_A = baseline_I / current_A × s_A_baselineII
```

其中：
- `baseline_I` = 不经过融合的小算子NPU耗时之和
- `current_A` = 当前算子NPU耗时
- `s_A_baselineII` = 相对于基线II的加速比

**最终性能得分**：
```
perf_score = Σ(w_A × s_A) / Σ(w_A) × 100
```

其中权重 `w_A = baseline_I_A / Σ baseline_I_A`（融合前耗时越高，占比越大）

### 2.3 初赛总分计算

| 指标 | 权重 |
|------|------|
| 功能得分（acc_score） | 40% |
| 性能得分（perf_score） | 60% |

---

## 三、测试与调试

### 3.1 公开测试用例

**可以测试！** 大赛提供公开用例集用于调试。

**测试方法**：
```bash
# 进入构建目录
cd ascendnpu-ir/build

# 运行所有测试
ctest --output-on-failure

# 运行特定测试
ctest -R <test_name> --output-on-failure

# 使用bishengir-opt测试单个IR文件
./bin/bishengir-opt --hfusion-auto-schedule <input.mlir>
```

### 3.2 测试工具

大赛提供参考测试工具/方法，参赛队可用于辅助调试。**最终测试结果以组委会复核结果为准**。

---

## 四、第三方IP与归属说明

### 4.1 本项目使用的第三方代码

| 来源 | 用途 | 许可证 | 说明 |
|------|------|--------|------|
| AscendNPU_IR | 基础框架 | Apache 2.0 | 华为开源，大赛提供 |
| LLVM/MLIR | 编译器框架 | Apache 2.0 | 通过子模块引入 |
| torch-mlir | PyTorch支持 | Apache 2.0 | 通过子模块引入 |

### 4.2 我们新增的代码

所有新增代码均在以下文件中：
- `ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/OptimizedAutoSchedule.cpp`（新增）
- `ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRSchedule.cpp`（修改）

**代码头部已添加说明**：
```cpp
//===- OptimizedAutoSchedule.cpp -- Optimized Auto Schedule ----*- C++ -*-===//
// 
// 本文件为2026年全国大学生计算机系统能力大赛编译系统设计赛参赛作品
// 基于AscendNPU_IR框架开发，遵循Apache 2.0开源协议
// 
// 优化技术：
// 1. 基于代价模型的Tiling策略选择
// 2. 多核负载均衡优化
// 3. Auto-Tuning搜索框架
//===----------------------------------------------------------------------===//
```

### 4.3 参考文档

- AscendNPU_IR用户指南：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html
- MLIR Pass开发指南：https://mlir.llvm.org/docs/PassManagement/#operation-pass

---

## 五、当前进度

### 5.1 已完成工作

| 阶段 | 状态 | 说明 |
|------|------|------|
| 环境搭建 | ✅ 完成 | 编译器、工具链、CANN包安装 |
| 代码克隆 | ✅ 完成 | AscendNPU_IR及子模块 |
| 框架学习 | ✅ 完成 | HFusion方言、AutoSchedule框架 |
| Pass设计 | ✅ 完成 | 代价模型、负载均衡、Auto-tuning |
| Pass实现 | ✅ 完成 | OptimizedAutoSchedule.cpp |
| 编译验证 | ✅ 完成 | bishengir-opt、bishengir-compile生成 |
| 功能测试 | ✅ 完成 | 76.88%通过（296/385） |

### 5.2 测试结果

```
Total Tests: 385
Passed: 296 (76.88%)
Failed: 3 (与我们的修改无关)
Skipped: 86
```

**我们的Pass已成功注册**：
```bash
./bin/bishengir-opt --help | grep optimized-auto-schedule
  --optimized-auto-schedule    : Optimized auto schedule pass with cost model
```

### 5.3 待完成工作

| 任务 | 优先级 | 预计时间 |
|------|--------|----------|
| 性能测试验证 | 高 | 1-2天 |
| 参数调优 | 高 | 2-3天 |
| 设计文档完善 | 中 | 1天 |
| 作品介绍视频 | 中 | 0.5天 |
| 代码规范检查 | 低 | 0.5天 |

---

## 六、技术优化方法

### 6.1 基于代价模型的Tiling策略

**核心思想**：根据算子特征选择最优的Tiling策略

**代价模型考虑因素**：
- 内存访问模式（顺序、随机、广播）
- 计算强度（FLOPs/字节）
- 同步开销
- 流水线效率

**实现位置**：[`OptimizedAutoSchedule.cpp`](../ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/OptimizedAutoSchedule.cpp)

### 6.2 多核负载均衡优化

**核心思想**：根据轴类型（并行轴/归约轴）智能分配核心数

**优化策略**：
- 并行轴：均匀分配核心
- 归约轴：考虑归约开销，动态调整核心数
- 混合轴：综合考虑负载均衡

**实现位置**：[`AnyPBRSchedule.cpp`](../ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRSchedule.cpp)

### 6.3 Auto-Tuning框架

**核心思想**：自动搜索最优参数配置

**搜索空间**：
- Tiling因子
- 核心分配策略
- 内存布局

**实现位置**：[`OptimizedAutoSchedule.cpp`](../ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/OptimizedAutoSchedule.cpp)

---

## 七、项目目录结构

```
1-NPU_IR/
├── ascendnpu-ir/                    # 主项目目录
│   ├── bishengir/                   # BiShengIR编译器
│   │   ├── include/                 # 头文件
│   │   │   └── bishengir/Dialect/HFusion/
│   │   │       └── Transforms/
│   │   │           ├── Passes.td    # Pass定义（已修改）
│   │   │           └── Passes.h     # Pass声明（已修改）
│   │   └── lib/                     # 实现文件
│   │       └── Dialect/HFusion/Transforms/
│   │           ├── OptimizedAutoSchedule.cpp  # 新增Pass
│   │           ├── CMakeLists.txt   # 构建配置（已修改）
│   │           └── AutoSchedule/
│   │               └── AnyPBRSchedule.cpp  # 已修改
│   ├── build/                       # 构建目录
│   │   └── bin/
│   │       ├── bishengir-opt        # Pass测试工具
│   │       └── bishengir-compile    # 端到端编译器
│   └── docs/                        # 官方文档
├── docs/                            # 我们的文档
│   ├── TEAM_GUIDE.md               # 本文档
│   ├── DESIGN_DOCUMENT.md          # 设计文档
│   └── 赛题技术方案.pdf
├── PROGRESS.md                      # 进度记录
└── README.md                        # 项目说明
```

### 7.1 关键文件说明

| 文件 | 作用 | 修改状态 |
|------|------|----------|
| `Passes.td` | Pass定义（TableGen格式） | 已修改 |
| `Passes.h` | Pass声明（C++头文件） | 已修改 |
| `CMakeLists.txt` | 构建配置 | 已修改 |
| `OptimizedAutoSchedule.cpp` | 新增Pass实现 | 新增 |
| `AnyPBRSchedule.cpp` | 多核负载均衡优化 | 已修改 |

---

## 八、后续工作计划

### 8.1 短期目标（1周内）

1. **性能测试验证**
   - 在昇腾A2/A3平台上运行测试
   - 对比基线性能
   - 分析性能瓶颈

2. **参数调优**
   - 调整代价模型参数
   - 优化Tiling策略
   - 改进负载均衡算法

### 8.2 中期目标（2周内）

1. **设计文档完善**
   - 补充技术细节
   - 添加性能分析数据
   - 完善创新点说明

2. **作品介绍视频**
   - 时长不超过5分钟
   - 介绍技术方案和创新点
   - 展示测试结果

### 8.3 决赛准备

1. **答辩准备**
   - 准备PPT
   - 模拟答辩
   - 准备技术问题

2. **代码规范**
   - 添加注释
   - 代码格式化
   - 完善文档

---

## 九、常见问题

### Q1: 如何运行单个测试？

```bash
cd ascendnpu-ir/build
ctest -R <test_name> --output-on-failure
```

### Q2: 如何查看Pass是否注册成功？

```bash
./bin/bishengir-opt --help | grep optimized-auto-schedule
```

### Q3: 如何调试Pass？

```bash
# 使用debug模式构建
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 运行时打印调试信息
./bin/bishengir-opt --debug --optimized-auto-schedule <input.mlir>
```

### Q4: 编译时间太长怎么办？

编译LLVM和MLIR确实需要较长时间（通常2-4小时）。建议：
- 使用增量编译：`ninja` 或 `make -j$(nproc)`
- 只编译修改的部分：`ninja bishengir-opt`

---

## 十、组员环境搭建与测试指南

> 本节详细说明组员如何从GitHub仓库克隆代码并在昇腾NPU环境中运行测试。

### 10.1 环境要求

**硬件要求**：
- CPU：鲲鹏920 ARM（或x86_64也可编译，但无法运行NPU测试）
- NPU：昇腾A2/A3
- 内存：至少32GB（编译需要）
- 磁盘：至少100GB可用空间

**软件要求**：
- 操作系统：openEuler 22.03 或 Ubuntu 22.04+
- CMake >= 3.28.0
- Ninja >= 1.12.0 或 Make
- Clang >= 10 或 GCC >= 9
- Python >= 3.8
- Git >= 2.20

### 10.2 安装CANN包

**必须先安装CANN包**，否则编译会失败。

```bash
# 1. 下载CANN社区版
# 访问：https://www.hiascend.com/developer/download/community/result?module=cann
# 选择对应版本（推荐9.0.0），下载ascend-toolkit

# 2. 解压并安装
chmod +x Ascend-cann-toolkit_9.0.0_linux-aarch64.run
./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install

# 3. 设置环境变量（每次编译前需要执行）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证安装
npu-smi info
```

### 10.3 克隆代码仓库

```bash
# 1. 创建工作目录
mkdir -p ~/compiler-competition && cd ~/compiler-competition

# 2. 克隆我们的GitHub仓库
git clone https://github.com/ouyangyipeng/Ascend-NPU-IR-Auto-tuning.git
cd Ascend-NPU-IR-Auto-tuning

# 3. 查看仓库结构
ls -la
# 应该看到：README.md, PROGRESS.md, apply_patches.sh, patches/, docs/

# 4. 克隆AscendNPU_IR官方仓库（这是子模块，需要单独克隆）
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir

# 5. 初始化子模块（LLVM和torch-mlir）
git submodule update --init --recursive
# 这一步需要较长时间，LLVM仓库约2GB

# 6. 返回项目根目录
cd ..
```

### 10.4 应用我们的优化补丁

```bash
# 确保在项目根目录
cd ~/compiler-competition/Ascend-NPU-IR-Auto-tuning

# 给脚本添加执行权限
chmod +x apply_patches.sh

# 运行补丁脚本
./apply_patches.sh ./ascendnpu-ir

# 验证补丁应用成功
ls -la ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/OptimizedAutoSchedule.cpp
# 应该显示文件存在
```

### 10.5 安装编译工具

```bash
# 检查CMake版本
cmake --version
# 如果版本低于3.28，需要升级

# 升级CMake（如果需要）
pip3 install cmake --upgrade
# 或者
# wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-aarch64.sh
# chmod +x cmake-3.28.0-linux-aarch64.sh
# ./cmake-3.28.0-linux-aarch64.sh --prefix=/usr/local

# 检查Ninja版本
ninja --version
# 如果版本低于1.12，需要升级

# 升级Ninja（如果需要）
pip3 install ninja --upgrade
```

### 10.6 编译项目

```bash
# 1. 进入ascendnpu-ir目录
cd ~/compiler-competition/Ascend-NPU-IR-Auto-tuning/ascendnpu-ir

# 2. 设置CANN环境变量（重要！）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 首次编译（应用补丁）
./build-tools/build.sh -o ./build --build-type Release --apply-patches

# 编译时间约2-4小时，请耐心等待
# 编译成功后会显示：
# "Build finished successfully!"

# 4. 后续增量编译（如果修改了代码）
./build-tools/build.sh -o ./build --build-type Release
```

### 10.7 验证编译结果

```bash
# 1. 检查生成的工具
cd ~/compiler-competition/Ascend-NPU-IR-Auto-tuning/ascendnpu-ir/build

ls -la bin/
# 应该看到：
# bishengir-opt      - Pass测试工具
# bishengir-compile  - 端到端编译器

# 2. 验证我们的Pass已注册
./bin/bishengir-opt --help | grep optimized-auto-schedule
# 应该显示：
#   --optimized-auto-schedule    : Optimized auto schedule pass with cost model

# 3. 检查Pass选项
./bin/bishengir-opt --help | grep -A 10 "optimized-auto-schedule"
```

### 10.8 运行测试

```bash
# 1. 进入构建目录
cd ~/compiler-competition/Ascend-NPU-IR-Auto-tuning/ascendnpu-ir/build

# 2. 运行所有测试
ctest --output-on-failure

# 3. 运行特定测试
ctest -R auto-schedule --output-on-failure

# 4. 查看测试统计
ctest -N  # 列出所有测试
ctest --print-labels  # 显示测试标签

# 5. 运行BiShengIR相关测试
ctest -L bishengir --output-on-failure
```

### 10.9 测试单个IR文件

```bash
# 1. 进入构建目录
cd ~/compiler-competition/Ascend-NPU-IR-Auto-tuning/ascendnpu-ir/build

# 2. 使用bishengir-opt测试
./bin/bishengir-opt --optimized-auto-schedule \
    ../bishengir/test/Dialect/HFusion/xxx.mlir

# 3. 查看Pass帮助
./bin/bishengir-opt --help | grep -A 20 "optimized-auto-schedule"
```

### 10.10 常见问题排查

**问题1：CMake版本太低**
```bash
# 解决方案
pip3 install cmake --upgrade --user
export PATH=$HOME/.local/bin:$PATH
cmake --version
```

**问题2：找不到CANN**
```bash
# 解决方案：设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 添加到~/.bashrc以便自动加载
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
```

**问题3：编译内存不足**
```bash
# 解决方案：减少并行编译数
./build-tools/build.sh -o ./build --build-type Release --apply-patches -j 4
# 将-j后面的数字改为CPU核心数的一半
```

**问题4：子模块克隆失败**
```bash
# 解决方案：手动克隆
cd ascendnpu-ir
git submodule deinit -f .
rm -rf .git/modules/*
git submodule update --init --recursive
```

**问题5：补丁应用失败**
```bash
# 解决方案：检查文件是否存在
ls -la patches/
# 确保所有补丁文件都存在

# 手动应用补丁
cp patches/OptimizedAutoSchedule.cpp ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/
cp patches/AnyPBRSchedule.cpp ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/
cp patches/Passes.td ascendnpu-ir/bishengir/include/bishengir/Dialect/HFusion/Transforms/
cp patches/Passes.h ascendnpu-ir/bishengir/include/bishengir/Dialect/HFusion/Transforms/
cp patches/CMakeLists.txt ascendnpu-ir/bishengir/lib/Dialect/HFusion/Transforms/
```

### 10.11 完整流程总结

```bash
# === 一键脚本 ===
# 将以下内容保存为 setup.sh

#!/bin/bash
set -e

echo "=== 1. 设置环境变量 ==="
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "=== 2. 克隆代码 ==="
mkdir -p ~/compiler-competition && cd ~/compiler-competition
git clone https://github.com/ouyangyipeng/Ascend-NPU-IR-Auto-tuning.git
cd Ascend-NPU-IR-Auto-tuning
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir
git submodule update --init --recursive

echo "=== 3. 应用补丁 ==="
cd ..
chmod +x apply_patches.sh
./apply_patches.sh ./ascendnpu-ir

echo "=== 4. 编译 ==="
cd ascendnpu-ir
./build-tools/build.sh -o ./build --build-type Release --apply-patches

echo "=== 5. 验证 ==="
cd build
./bin/bishengir-opt --help | grep optimized-auto-schedule

echo "=== 6. 测试 ==="
ctest --output-on-failure

echo "=== 完成！ ==="
```

---

## 十一、联系方式与资源

- **大赛官网**：https://compiler.educg.net
- **AscendNPU_IR仓库**：https://gitcode.com/Ascend/ascendnpu-ir
- **我们的GitHub仓库**：https://github.com/ouyangyipeng/Ascend-NPU-IR-Auto-tuning
- **用户指南**：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html

---

*最后更新：2026-03-22*