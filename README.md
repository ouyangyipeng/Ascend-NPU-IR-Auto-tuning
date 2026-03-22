# 基于AscendNPU_IR的NPU自动融合代码生成与调优

## 项目简介

本项目为2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）参赛作品，实现了基于AscendNPU_IR的NPU自动融合代码生成与调优优化Pass。

## 核心优化

### 1. 基于代价模型的Tiling策略
- 综合考虑内存访问、计算、同步等多维因素
- 硬件感知的性能评估
- 自动选择最优Tiling配置

### 2. 多核负载均衡优化
- 基于工作负载分析的核心分配
- 动态负载均衡因子计算
- 通信开销感知

### 3. Auto-Tuning框架
- 遗传算法参数搜索
- 可配置的搜索空间
- 支持多种评估指标

## 项目结构

```
.
├── ascendnpu-ir/                    # AscendNPU_IR源码
│   └── bishengir/
│       ├── include/bishengir/Dialect/HFusion/Transforms/
│       │   ├── Passes.h             # Pass接口声明（已修改）
│       │   └── Passes.td            # Pass定义（已修改）
│       └── lib/Dialect/HFusion/Transforms/
│           ├── CMakeLists.txt       # 构建配置（已修改）
│           ├── OptimizedAutoSchedule.cpp      # 新增优化调度Pass
│           └── AutoSchedule/
│               └── AnyPBRSchedule.cpp # 多核负载均衡优化（已修改）
├── docs/
│   ├── 2026年...技术方案.pdf        # 赛题文档
│   ├── TEAM_GUIDE.md               # 团队指南（新）
│   └── DESIGN_DOCUMENT.md           # 设计说明文档
├── PROGRESS.md                      # 进度记录
└── README.md                        # 本文档
```

## 构建说明

### 环境要求
- CMake >= 3.28
- Ninja >= 1.12.0
- Clang >= 10
- openEuler / Ubuntu 22.04+

### 构建步骤

```bash
# 1. 克隆项目（包含子模块）
git clone --recursive https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir

# 2. 应用优化代码
# 将本项目中的优化文件复制到对应位置

# 3. 首次构建
./build-tools/build.sh -o ./build --build-type Release --apply-patches

# 4. 后续构建
./build-tools/build.sh -o ./build --build-type Release
```

### 运行测试

```bash
# 在build目录下
cmake --build . --target "check-bishengir"
```

## 使用方法

### Pass选项

```
-optimized-auto-schedule:
  -block-dim=<N>                    : 使用的核心数量
  -enable-auto-tuning=<true|false> : 启用自动调优
  -enable-cost-model=<true|false>  : 启用代价模型引导
  -enable-load-balance=<true|false>: 启用负载均衡优化
  -max-buffer-count=<N>            : 最大Buffer数量
  -auto-tuning-iterations=<N>      : 自动调优迭代次数
```

### 示例用法

```bash
# 使用优化调度Pass
bishengir-opt --optimized-auto-schedule input.mlir -o output.mlir

# 启用自动调优
bishengir-opt --optimized-auto-schedule --enable-auto-tuning input.mlir -o output.mlir
```

## 性能评估

### 评分标准
| 指标 | 权重 |
|------|------|
| 功能得分 | 40% |
| 性能得分 | 60% |

### 基线对比
- 基线I：不经过融合的小算子NPU耗时之和
- 基线II：经过AscendNPU_IR原生自动融合模块编译的融合算子NPU耗时

## 技术文档

详细设计说明请参见 [docs/DESIGN_DOCUMENT.md](docs/DESIGN_DOCUMENT.md)

## 参考资料

1. [AscendNPU IR用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html)
2. [MLIR Pass管理](https://mlir.llvm.org/docs/PassManagement/)
3. [比赛官网](https://compiler.educg.net)

## 许可证

Apache License v2.0

---

*2026年全国大学生计算机系统能力大赛编译系统设计赛参赛作品*