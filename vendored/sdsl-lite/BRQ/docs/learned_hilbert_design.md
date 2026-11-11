# 学习型Hilbert索引设计文档

> Status: Historical/reference note. The current maintained implementation is AC-BRQ v2 under `../code/`.


## 1. 概述

本文档详细描述了学习型Hilbert索引(Learned Hilbert Index)的设计与实现。学习型Hilbert索引是传统Hilbert曲线空间填充技术与现代机器学习方法的结合，旨在解决高维空间中窗口查询的效率和空间复杂度问题。

## 2. 背景与动机

### 2.1 传统方法的局限

传统基于Hilbert曲线的窗口查询方法在二维空间表现优异，但在高维空间面临两个主要挑战：

1. **空间复杂度爆炸**：维度每增加1，存储空间需求呈指数增长(O(2^(r·N)))
2. **计算复杂度增加**：高维空间中计算Hilbert编码的复杂度显著增加

例如，对于6阶Hilbert曲线：
- 2D: 2^12 个点，约4MB
- 3D: 2^18 个点，约1GB
- 4D: 2^24 个点，约256GB

### 2.2 Learned Index范式

Learned Index是由Kraska等人于2018年提出的概念，核心思想是：
- 数据结构的行为可以通过机器学习模型学习和近似
- 通过神经网络预测复杂函数映射，减少存储需求
- 对于少量预测错误的情况，维护小型修正表

## 3. 系统架构

学习型Hilbert索引由三个核心组件构成：

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  神经网络预测模型  │ --> │    误差修正表     │ --> │    查询处理器     │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

### 3.1 神经网络预测模型

**功能**：学习坐标点到Hilbert编码的映射关系  
**输入**：N维坐标点 (x₁, x₂, ..., xₙ)  
**输出**：对应的Hilbert编码  
**结构**：多层感知机(MLP)，具有以下特点：
- 输入层：N个神经元（坐标维度）
- 隐藏层：3层，神经元数量分别为32, 64, 32
- 输出层：1个神经元（Hilbert编码的归一化值）
- 激活函数：Leaky ReLU
- 训练方法：随机梯度下降，MSE损失函数

### 3.2 误差修正表

**功能**：存储神经网络预测错误的点的修正值  
**数据结构**：哈希表，键为坐标点，值为误差修正值  
**策略**：
- 仅存储误差超过阈值的点
- 阈值可动态调整，平衡准确性与空间效率
- 典型存储量：全空间的1%-5%

### 3.3 查询处理器

**功能**：处理窗口查询请求  
**算法**：
1. 提取查询窗口的边界面
2. 枚举每个边界面上的点
3. 使用神经网络+修正表预测每个点的Hilbert编码
4. 对结果进行排序和去重

## 4. 算法设计

### 4.1 训练流程

```
1. 数据生成
   a. 均匀采样(60%)：在整个空间均匀分布的随机点
   b. 边界样本(30%)：强制一个维度在边界的随机点
   c. 角点样本(10%)：坐标极值组合的点

2. 数据归一化
   a. 将坐标归一化到[0,1]范围
   b. 将Hilbert编码归一化到[0,1]范围

3. 模型训练
   a. 随机梯度下降优化
   b. 批量处理，学习率逐渐衰减
   c. 早停策略避免过拟合

4. 误差修正表构建
   a. 使用训练好的模型预测验证集
   b. 计算每个点的预测误差
   c. 将误差超过阈值的点添加到修正表
```

### 4.2 查询算法

```
函数 windowQuery(box):
    results = []
    
    # 对每个维度的边界面
    for dim = 0 to N-1:
        # 对每个边界类型(min/max)
        for boundType = 0 to 1:
            # 递归枚举边界面上的点
            enumerateBoundary(dim, boundType, box, point, 0)
    
    # 去重和排序
    排序(results)
    去重(results)
    return results

函数 enumerateBoundary(dim, boundType, box, point, depth):
    if depth == N:
        # 使用学习型索引预测编码
        code = predict(point)
        results.push(code)
        return
    
    if depth == dim:
        # 固定在边界
        point[depth] = boundType ? box.max_corner[depth] : box.min_corner[depth]
        enumerateBoundary(dim, boundType, box, point, depth+1)
    else:
        # 其他维度遍历范围
        for i = box.min_corner[depth] to box.max_corner[depth]:
            point[depth] = i
            enumerateBoundary(dim, boundType, box, point, depth+1)

函数 predict(point):
    # 使用神经网络预测
    normalized = model.forward(point)
    
    # 转换回实际编码
    predicted = normalized * max_hilbert_value
    
    # 应用误差修正
    if correctionTable.hasCorrection(point):
        predicted += correctionTable.getCorrection(point)
    
    return predicted
```

## 5. 性能分析

### 5.1 空间复杂度

| 组件          | 传统方法                | 学习型索引             |
|--------------|------------------------|----------------------|
| 主数据结构     | O(2^(r·N))            | O(模型参数数量)         |
| 索引结构       | O(N·2^(r·N))          | O(修正表大小)          |
| 典型空间(3D,r=6)| ~1GB                  | ~5MB                 |
| 扩展性         | 指数级增长              | 近似线性增长           |

### 5.2 时间复杂度

| 操作          | 传统方法                | 学习型索引             |
|--------------|------------------------|----------------------|
| 初始化        | O(2^(r·N))            | O(训练样本数·迭代次数)   |
| 单点查询       | O(1)                  | O(1)                 |
| 窗口查询       | O(N·L^(N-1))          | O(N·L^(N-1))         |
| 编码计算       | O(r·N)                | O(1) (推理)           |

### 5.3 准确性与空间权衡

| 误差阈值 | 修正表大小(占总空间%) | 准确率 | 典型内存占用 |
|---------|-------------------|-------|-----------|
| 5       | ~5%               | >99%  | 大        |
| 10      | ~3%               | >98%  | 中        |
| 50      | ~1%               | >95%  | 小        |
| 100     | ~0.5%             | >90%  | 极小      |

## 6. 实现详情

### 6.1 关键类设计

1. **NeuralNetwork<N>**
   - 实现前馈神经网络
   - 支持前向传播和训练
   - 提供模型保存/加载功能

2. **ErrorCorrectionTable<N>**
   - 使用哈希表存储修正值
   - 提供高效查询和更新操作
   - 支持序列化和反序列化

3. **LearnedHilbertIndex<N>**
   - 整合神经网络和修正表
   - 提供训练、评估和查询接口
   - 管理模型生命周期

### 6.2 优化技术

1. **分层采样**
   - 确保训练数据覆盖边界和角点
   - 提高对关键区域的预测准确性

2. **自适应修正**
   - 根据查询模式动态调整修正表
   - 热点区域使用更低的误差阈值

3. **查询优化**
   - 预计算窗口边界
   - 利用空间局部性优化访问模式

## 7. 扩展与未来工作

### 7.1 支持更高维度

当前实现已经支持N维空间，但可以通过以下方式进一步优化高维性能：
- 特化高维的网络结构
- 维度约减技术
- 层次化修正表

### 7.2 动态更新

实现增量学习机制，允许在不完全重训练的情况下更新模型：
- 在线学习算法
- 热点区域的优先更新
- 修正表的动态扩展

### 7.3 硬件加速

利用现代硬件加速神经网络推理：
- GPU加速
- 量化技术减少模型大小
- SIMD指令优化

## 8. 结论

学习型Hilbert索引代表了传统空间数据结构与现代机器学习技术的融合，解决了高维空间中传统方法面临的空间复杂度挑战。通过"神经网络预测+误差修正表"的设计模式，该方法在保持查询性能的同时显著降低了存储需求，特别适用于3D及更高维度的空间索引应用。

## 9. 参考文献

1. Kraska, T., et al. (2018). The Case for Learned Index Structures. SIGMOD.
2. Hamilton, C. H., & Rau-Chaplin, A. (2008). Compact Hilbert indices: Space-filling curves for domains with unequal side lengths.
3. Faloutsos, C. (1988). Gray codes for partial match and range queries. IEEE TOSE.
4. Nathan, V., et al. (2020). Learning Multi-dimensional Indexes. SIGMOD.
