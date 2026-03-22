# 基于AscendNPU_IR的NPU自动融合代码生成与调优 - 设计说明文档

## 一、项目概述

### 1.1 赛题背景
本项目为2026年全国大学生计算机系统能力大赛编译系统设计赛（华为毕昇杯）参赛作品。赛题要求在AscendNPU_IR中以增改Pass的方式实现给定融合算子的自动代码生成和调优，确保算子功能正确且最终结果通过误差检验的前提下，最大化计算性能。

### 1.2 目标平台
- **CPU**: 鲲鹏920 ARM
- **NPU**: 昇腾A2/A3
- **操作系统**: openEuler

### 1.3 评分标准
| 指标 | 权重 |
|------|------|
| 功能得分（accscore） | 40% |
| 性能得分（perfscore） | 60% |

## 二、技术方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    HFusion Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  preProcess → preFlatten → flattenAndFold → inferAndOutline │
│           → OptimizedAutoSchedule → postProcess             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OptimizedAutoSchedule Pass                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │ Cost Model      │  │ Multi-Core Load │  │ Auto-Tuning  ││
│  │ Guided Tiling   │  │ Balancing       │  │ Framework    ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心优化策略

#### 2.2.1 基于代价模型的Tiling策略选择

**问题分析**：
原始的Tiling Key选择策略基于UB容量的简单贪心算法，未考虑：
- 内存访问模式的影响
- 计算密度
- 多核并行效率
- 流水线利用率

**优化方案**：
设计代价模型评估不同Tiling配置的性能：

```cpp
struct TilingCostFactors {
  double memoryAccessCost;    // 内存访问代价
  double computeCost;         // 计算代价
  double synchronizationCost; // 同步代价
  double pipelineEfficiency;  // 流水线效率
};

// 总代价计算
totalCost = (memoryAccessCost + computeCost * 0.5 + syncCost * 0.3) / pipelineEfficiency;
```

**内存访问代价计算**：
- 顺序访问：充分利用UB带宽
- 跨步访问：带宽利用率降低
- 随机访问：最差性能
- 广播/归约：特殊处理

#### 2.2.2 多核负载均衡优化

**问题分析**：
原始的多核分配策略简单地将核心分配给并行轴和归约轴，未考虑：
- 实际工作负载比例
- 通信开销
- 负载均衡

**优化方案**：
```cpp
struct CoreAssignment {
  int64_t coresForParallel;   // 并行轴核心数
  int64_t coresForReduce;     // 归约轴核心数
  double loadBalanceFactor;   // 负载均衡因子
};

// 基于工作负载比例的最优分配
CoreAssignment calculateOptimalAssignment(
    int64_t totalCores,
    const SmallVector<int64_t> &tileSizes,
    const SmallVector<int64_t> &dimSizes,
    const llvm::SmallSetVector<int64_t, 4> &reduceDims);
```

#### 2.2.3 Auto-Tuning框架

**设计目标**：
- 自动搜索最优Tiling参数
- 支持多种搜索策略
- 可配置的搜索空间

**搜索空间定义**：
```cpp
struct AutoTuningSpace {
  // 每个维度的Tiling因子候选
  SmallVector<SmallVector<int64_t>> tilingFactorCandidates;
  // Buffer数量候选
  SmallVector<int64_t> bufferCountCandidates;
  // 核心分配候选
  SmallVector<std::pair<int64_t, int64_t>> coreSplitCandidates;
};
```

**搜索策略**：
采用遗传算法进行参数搜索：
1. 初始化随机种群
2. 适应度评估
3. 选择、交叉、变异
4. 迭代直到收敛或达到最大迭代次数

### 2.3 硬件感知优化

#### 2.3.1 昇腾NPU硬件特性

| 特性 | 参数 |
|------|------|
| UB大小 | 2MB |
| UB对齐 | 32字节 |
| L1大小 | 1MB |
| 最大核心数 | 40 |
| GM带宽 | 1200 GB/s |
| UB带宽 | 4800 GB/s |

#### 2.3.2 对齐约束处理

```cpp
// Stride对齐（32字节）
for (const auto &alignInfo : kernelInfo->getStrideAlignments()) {
  auto [idx, alignment] = alignInfo;
  dims[idx] = dims[idx].alignTo(alignment);
}

// Tile对齐
for (const auto &alignInfo : kernelInfo->getTileAlignments()) {
  auto [idx, alignment] = alignInfo;
  tileSize = max(tileSize.alignDown(alignment), 1);
}
```

## 三、实现细节

### 3.1 新增文件

| 文件路径 | 说明 |
|----------|------|
| `Transforms/OptimizedAutoSchedule.cpp` | 优化调度Pass实现 |
| `AutoSchedule/OptimizedAnyPBRSchedule.h` | 优化调度器头文件 |
| `AutoSchedule/OptimizedAnyPBRSchedule.cpp` | 优化调度器实现 |

### 3.2 Pass注册

在`Passes.td`中添加：
```tablegen
def OptimizedAutoSchedule : Pass<"optimized-auto-schedule", "func::FuncOp"> {
  let summary = "Optimized auto schedule pass with cost model and auto-tuning support.";
  let constructor = "mlir::hfusion::createOptimizedAutoSchedulePass()";
  let options = [
    Option<"blockDim", "block-dim", "unsigned", "1", "Number of blocks to use">,
    Option<"enableAutoTuning", "enable-auto-tuning", "bool", "false", 
           "Enable auto-tuning for parameter search">,
    Option<"enableCostModel", "enable-cost-model", "bool", "true", 
           "Enable cost model guided tiling">,
    // ...
  ];
}
```

### 3.3 Pipeline集成

优化Pass可以替代原有的`hfusion-auto-schedule` Pass，或作为可选的优化步骤：

```cpp
static void hfusionAutoSchedulePipeline(OpPassManager &pm,
                                        const HFusionPipelineOptions &options) {
  if (options.enableOptimizedSchedule) {
    // 使用优化调度Pass
    pm.addPass(createOptimizedAutoSchedulePass());
  } else {
    // 使用原始调度Pass
    pm.addPass(createHFusionAutoSchedulePass(autoScheduleOptions));
  }
}
```

## 四、测试与验证

### 4.1 功能测试

测试用例来源：
- 昇腾大模型平台的融合算子
- 静态shape和动态shape算子

精度标准：
- 整数计算：二进制对比一致
- 浮点数计算：相对误差/绝对误差满足阈值

### 4.2 性能测试

性能指标：
- NPU耗时（微秒）
- 加速比 = baseline / current

基线定义：
- 基线I：不经过融合的小算子NPU耗时之和
- 基线II：经过AscendNPU_IR原生自动融合模块编译的融合算子NPU耗时

## 五、创新点

### 5.1 代价模型引导的Tiling策略
- 首次在AscendNPU_IR中引入代价模型
- 综合考虑内存、计算、同步等多维因素
- 硬件感知的性能评估

### 5.2 智能多核负载均衡
- 基于工作负载分析的核心分配
- 动态负载均衡因子计算
- 通信开销感知

### 5.3 可扩展的Auto-Tuning框架
- 遗传算法参数搜索
- 可配置的搜索空间
- 支持多种评估指标

## 六、遇到的挑战与解决方案

### 6.1 动态Shape处理
**挑战**：动态shape算子的Tiling计算复杂
**解决方案**：使用符号表达式和运行时参数化

### 6.2 硬件约束满足
**挑战**：多种对齐约束可能冲突
**解决方案**：分层约束处理，优先保证关键约束

### 6.3 编译时间开销
**挑战**：Auto-Tuning增加编译时间
**解决方案**：
- 缓存历史最优配置
- 增量搜索策略
- 可配置的搜索迭代次数

## 七、未来工作

1. **机器学习辅助的代价模型**
   - 使用历史性能数据训练模型
   - 更精确的性能预测

2. **更多融合模式支持**
   - 扩展到更多算子类型
   - 支持更复杂的融合模式

3. **分布式编译支持**
   - 并行参数搜索
   - 分布式性能评估

## 八、参考文献

1. AscendNPU IR用户指南: https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/AscendNPUIR/ir_001.html
2. MLIR Pass管理: https://mlir.llvm.org/docs/PassManagement/
3. TVM AutoTVM: https://tvm.apache.org/docs/topic/autotvm/index.html
4. Apache TVM Ansor: https://tvm.apache.org/docs/topic/autotvm/autotvm_scheduler.html

---

*文档版本: 1.0*
*最后更新: 2026-03-20*