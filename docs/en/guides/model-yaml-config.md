---
comments: true
description: 学习如何使用 Ultralytics YAML 配置文件构建和自定义模型架构。掌握模块定义、连接和缩放参数。
keywords: Ultralytics, YOLO, 模型架构, YAML 配置, 神经网络, 深度学习, 骨干网络, 头部, 模块, 自定义模型
---

# 模型 YAML 配置指南

模型 YAML 配置文件作为 Ultralytics 神经网络的架构蓝图。它定义了层如何连接、每个模块使用什么参数，以及整个网络如何在不同模型大小之间缩放。

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/yaml-configuration-guide.avif" alt="模型 YAML 配置工作流程。">

## 配置结构

模型 YAML 文件分为三个主要部分，它们共同定义架构。

### 参数部分

**参数**部分指定模型的全局特性和缩放行为：

```yaml
# 参数
nc: 80 # 类别数量
scales: # 复合缩放常数 [深度, 宽度, 最大通道数]
    n: [0.50, 0.25, 1024] # nano：浅层，窄通道
    s: [0.50, 0.50, 1024] # small：浅深度，标准宽度
    m: [0.50, 1.00, 512] # medium：中等深度，全宽度
    l: [1.00, 1.00, 512] # large：全深度和宽度
    x: [1.00, 1.50, 512] # extra-large：最大性能
kpt_shape: [17, 3] # 仅用于姿态模型
```

- `nc` 设置模型预测的类别数量。
- `scales` 定义复合缩放因子，调整模型深度、宽度和最大通道数，以生成不同大小的变体（从 nano 到 extra-large）。
- `kpt_shape` 适用于姿态模型。可以是 `[N, 2]` 用于 `(x, y)` 关键点，或 `[N, 3]` 用于 `(x, y, visibility)`。

!!! tip "使用 `scales` 减少冗余"

    `scales` 参数允许您从单个基础 YAML 生成多个模型大小。例如，当您加载 `yolo11n.yaml` 时，Ultralytics 读取基础 `yolo11.yaml` 并应用 `n` 缩放因子（`depth=0.50`，`width=0.25`）来构建 nano 变体。

!!! note "`nc` 和 `kpt_shape` 依赖于数据集"

    如果您的数据集指定了不同的 `nc` 或 `kpt_shape`，Ultralytics 将在运行时自动覆盖模型配置以匹配数据集 YAML。

### 骨干网络和头部架构

模型架构由骨干网络（特征提取）和头部（任务特定）部分组成：

```yaml
backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0: 初始卷积
    - [-1, 1, Conv, [128, 3, 2]] # 1: 下采样
    - [-1, 3, C2f, [128, True]] # 2: 特征处理

head:
    - [-1, 1, nn.Upsample, [None, 2, nearest]] # 6: 上采样
    - [[-1, 2], 1, Concat, [1]] # 7: 跳跃连接
    - [-1, 3, C2f, [256]] # 8: 处理特征
    - [[8], 1, Detect, [nc]] # 9: 检测层
```

## 层规范格式

每一层都遵循一致的模式：**`[from, repeats, module, args]`**

| 组件        | 用途       | 示例                                              |
| ----------- | ---------- | ------------------------------------------------- |
| **from**    | 输入连接   | `-1`（前一层），`6`（第 6 层），`[4, 6, 8]`（多输入） |
| **repeats** | 重复次数   | `1`（单次），`3`（重复 3 次）                     |
| **module**  | 模块类型   | `Conv`，`C2f`，`TorchVision`，`Detect`            |
| **args**    | 模块参数   | `[64, 3, 2]`（通道数，卷积核，步幅）              |

### 连接模式

`from` 字段在整个网络中创建灵活的数据流模式：

=== "顺序流"

    ```yaml
    - [-1, 1, Conv, [64, 3, 2]]    # 从前一层获取输入
    ```

=== "跳跃连接"

    ```yaml
    - [[-1, 6], 1, Concat, [1]]    # 将当前层与第 6 层组合
    ```

=== "多输入融合"

    ```yaml
    - [[4, 6, 8], 1, Detect, [nc]] # 使用 3 个特征尺度的检测头
    ```

!!! note "层索引"

    层从 0 开始索引。负索引引用前面的层（`-1` = 前一层），而正索引通过位置引用特定层。

### 模块重复

`repeats` 参数创建更深的网络部分：

```yaml
- [-1, 3, C2f, [128, True]] # 创建 3 个连续的 C2f 块
- [-1, 1, Conv, [64, 3, 2]] # 单个卷积层
```

实际重复次数会乘以模型大小配置中的深度缩放因子。

## 可用模块

模块按功能组织，定义在 [Ultralytics 模块目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules)中。以下表格按类别显示常用模块，源代码中还有更多可用模块：

### 基本操作

| 模块          | 用途                         | 源码                                                                                           | 参数                                    |
| ------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------- |
| `Conv`        | 卷积 + 批归一化 + 激活       | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py) | `[out_ch, kernel, stride, pad, groups]` |
| `nn.Upsample` | 空间上采样                   | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html)               | `[size, scale_factor, mode]`            |
| `nn.Identity` | 直通操作                     | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html)               | `[]`                                    |

### 复合块

| 模块     | 用途                       | 源码                                                                                             | 参数                            |
| -------- | -------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------- |
| `C2f`    | 带 2 个卷积的 CSP 瓶颈     | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, shortcut, expansion]` |
| `SPPF`   | 空间金字塔池化（快速）     | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, kernel_size]`         |
| `Concat` | 通道级联                   | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)   | `[dimension]`                   |

### 专用模块

| 模块          | 用途                       | 源码                                                                                             | 参数                                                     |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `TorchVision` | 加载任何 torchvision 模型  | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, model_name, weights, unwrap, truncate, split]` |
| `Index`       | 从列表中提取特定张量       | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, index]`                                        |
| `Detect`      | YOLO 检测头                | [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)   | `[nc, anchors, ch]`                                      |

!!! info "完整模块列表"

    这只是可用模块的子集。有关模块及其参数的完整列表，请探索[模块目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules)。

## 高级功能

### TorchVision 集成

TorchVision 模块可以无缝集成任何 [TorchVision 模型](https://docs.pytorch.org/vision/stable/models.html)作为骨干网络：

=== "Python"

    ```python
    from ultralytics import YOLO

    # 使用 ConvNeXt 骨干网络的模型
    model = YOLO("convnext_backbone.yaml")
    results = model.train(data="coco8.yaml", epochs=100)
    ```

=== "YAML 配置"

    ```yaml
    backbone:
      - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, False]]
    head:
      - [-1, 1, Classify, [nc]]
    ```

    **参数说明：**

    - `768`：预期输出通道数
    - `convnext_tiny`：模型架构（[可用模型](https://docs.pytorch.org/vision/stable/models.html)）
    - `DEFAULT`：使用预训练权重
    - `True`：移除分类头
    - `2`：截断最后 2 层
    - `False`：返回单个张量（非列表）

!!! tip "多尺度特征"

    将最后一个参数设置为 `True` 以获取用于多尺度检测的中间特征图。

### 用于特征选择的 Index 模块

当使用输出多个特征图的模型时，Index 模块选择特定输出：

```yaml
backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]] # 多输出
head:
    - [0, 1, Index, [192, 4]] # 选择第 4 个特征图（192 通道）
    - [0, 1, Index, [384, 6]] # 选择第 6 个特征图（384 通道）
    - [0, 1, Index, [768, 8]] # 选择第 8 个特征图（768 通道）
    - [[1, 2, 3], 1, Detect, [nc]] # 多尺度检测
```

## 模块解析系统

了解 Ultralytics 如何定位和导入模块对于自定义至关重要：

### 模块查找过程

Ultralytics 在 [`parse_model`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中使用三层系统：

```python
# 核心解析逻辑
m = getattr(torch.nn, m[3:]) if "nn." in m else getattr(torchvision.ops, m[4:]) if "ops." in m else globals()[m]
```

1. **PyTorch 模块**：以 `'nn.'` 开头的名称 → `torch.nn` 命名空间
2. **TorchVision 操作**：以 `'ops.'` 开头的名称 → `torchvision.ops` 命名空间
3. **Ultralytics 模块**：所有其他名称 → 通过导入的全局命名空间

### 模块导入链

标准模块通过 [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中的导入变得可用：

```python
from ultralytics.nn.modules import (  # noqa: F401
    SPPF,
    C2f,
    Conv,
    Detect,
    # ... 更多模块
    Index,
    TorchVision,
)
```

## 自定义模块集成

### 源代码修改

修改源代码是集成自定义模块最通用的方法，但可能比较棘手。要定义和使用自定义模块，请按照以下步骤操作：

1. **以开发模式安装 Ultralytics**，使用[快速入门指南](https://docs.ultralytics.com/quickstart/#git-clone)中的 Git 克隆方法。

2. **在 [`ultralytics/nn/modules/block.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) 中定义您的模块**：

    ```python
    class CustomBlock(nn.Module):
        """带有 Conv-BatchNorm-ReLU 序列的自定义块。"""

        def __init__(self, c1, c2):
            """使用输入和输出通道初始化 CustomBlock。"""
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU())

        def forward(self, x):
            """通过块的前向传递。"""
            return self.layers(x)
    ```

3. **在 [`ultralytics/nn/modules/__init__.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/__init__.py) 中在包级别公开您的模块**：

    ```python
    from .block import CustomBlock  # noqa 使 CustomBlock 可作为 ultralytics.nn.modules.CustomBlock 使用
    ```

4. **在 [`ultralytics/nn/tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中添加导入**：

    ```python
    from ultralytics.nn.modules import CustomBlock  # noqa
    ```

5. **在 `ultralytics/nn/tasks.py` 的 [`parse_model()`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中处理特殊参数**（如果需要）：

    ```python
    # 在 parse_model() 函数中添加此条件
    if m is CustomBlock:
        c1, c2 = ch[f], args[0]  # 输入通道，输出通道
        args = [c1, c2, *args[1:]]
    ```

6. **在模型 YAML 中使用该模块**：

    ```yaml
    # custom_model.yaml
    nc: 1
    backbone:
        - [-1, 1, CustomBlock, [64]]
    head:
        - [-1, 1, Classify, [nc]]
    ```

7. **检查 FLOPs** 以确保前向传递正常工作：

    ```python
    from ultralytics import YOLO

    model = YOLO("custom_model.yaml", task="classify")
    model.info()  # 如果正常工作应该打印非零 FLOPs
    ```

## 示例配置

### 基本检测模型

```yaml
# 简单的 YOLO 检测模型
nc: 80
scales:
    n: [0.33, 0.25, 1024]

backbone:
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]] # 2
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]] # 4
    - [-1, 1, SPPF, [256, 5]] # 5

head:
    - [-1, 1, Conv, [256, 3, 1]] # 6
    - [[6], 1, Detect, [nc]] # 7
```

### TorchVision 骨干网络模型

```yaml
# 带有 YOLO 头的 ConvNeXt 骨干网络
nc: 80

backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]]

head:
    - [0, 1, Index, [192, 4]] # P3 特征
    - [0, 1, Index, [384, 6]] # P4 特征
    - [0, 1, Index, [768, 8]] # P5 特征
    - [[1, 2, 3], 1, Detect, [nc]] # 多尺度检测
```

### 分类模型

```yaml
# 简单的分类模型
nc: 1000

backbone:
    - [-1, 1, Conv, [64, 7, 2, 3]]
    - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
    - [-1, 4, C2f, [64, True]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 8, C2f, [128, True]]
    - [-1, 1, nn.AdaptiveAvgPool2d, [1]]

head:
    - [-1, 1, Classify, [nc]]
```

## 最佳实践

### 架构设计技巧

**从简单开始**：在自定义之前从经过验证的架构开始。使用现有的 YOLO 配置作为模板，逐步修改而不是从头构建。

**增量测试**：逐步验证每个修改。一次添加一个自定义模块，并在继续下一个更改之前验证它是否有效。

**监控通道**：确保连接层之间的通道维度匹配。一层的输出通道（`c2`）必须与序列中下一层的输入通道（`c1`）匹配。

**使用跳跃连接**：利用 `[[-1, N], 1, Concat, [1]]` 模式进行特征重用。这些连接有助于梯度流动，并允许模型组合来自不同尺度的特征。

**适当缩放**：根据计算约束选择模型缩放。边缘设备使用 nano（`n`），平衡性能使用 small（`s`），最大精度使用更大的缩放（`m`、`l`、`x`）。

### 性能考虑

**深度与宽度**：深层网络通过多个变换层捕获复杂的层次特征，而宽层网络在每层并行处理更多信息。根据任务复杂性平衡这些。

**跳跃连接**：改善训练期间的梯度流动，并在整个网络中实现特征重用。它们在更深的架构中特别重要，以防止梯度消失。

**瓶颈块**：在保持模型表达能力的同时降低计算成本。像 `C2f` 这样的模块使用比标准卷积更少的参数，同时保持特征学习能力。

**多尺度特征**：对于检测同一图像中不同大小的对象至关重要。使用具有不同尺度多个检测头的特征金字塔网络（FPN）模式。

## 故障排除

### 常见问题

| 问题                                            | 原因                   | 解决方案                                                                                              |
| ----------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `KeyError: 'ModuleName'`                        | 模块未导入             | 添加到 [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 导入 |
| 通道维度不匹配                                  | `args` 规范不正确      | 验证输入/输出通道兼容性                                                                               |
| `AttributeError: 'int' object has no attribute` | 参数类型错误           | 检查模块文档以获取正确的参数类型                                                                      |
| 模型构建失败                                    | 无效的 `from` 引用     | 确保引用的层存在                                                                                      |

### 调试技巧

在开发自定义架构时，系统性调试有助于及早发现问题：

**使用 Identity 头进行测试**

用 `nn.Identity` 替换复杂的头以隔离骨干网络问题：

```yaml
nc: 1
backbone:
    - [-1, 1, CustomBlock, [64]]
head:
    - [-1, 1, nn.Identity, []] # 用于调试的直通
```

这允许直接检查骨干网络输出：

```python
import torch

from ultralytics import YOLO

model = YOLO("debug_model.yaml")
output = model.model(torch.randn(1, 3, 640, 640))
print(f"输出形状: {output.shape}")  # 应该匹配预期维度
```

**模型架构检查**

检查 FLOPs 计数并打印每一层也可以帮助调试自定义模型配置的问题。有效模型的 FLOPs 计数应该是非零的。如果是零，则前向传递可能存在问题。运行简单的前向传递应该显示遇到的确切错误。

```python
from ultralytics import YOLO

# 使用详细输出构建模型以查看层详情
model = YOLO("debug_model.yaml", verbose=True)

# 检查模型 FLOPs。失败的前向传递导致 0 FLOPs。
model.info()

# 检查各个层
for i, layer in enumerate(model.model.model):
    print(f"层 {i}: {layer}")
```

**逐步验证**

1. **从最小开始**：首先使用最简单的架构进行测试
2. **增量添加**：逐层构建复杂性
3. **检查维度**：验证通道和空间大小兼容性
4. **验证缩放**：使用不同的模型缩放进行测试（`n`、`s`、`m`）

## 常见问题

### 如何更改模型中的类别数量？

在 YAML 文件顶部设置 `nc` 参数以匹配数据集的类别数量。

```yaml
nc: 5 # 5 个类别
```

### 我可以在模型 YAML 中使用自定义骨干网络吗？

可以。您可以使用任何支持的模块，包括 TorchVision 骨干网络，或定义自己的自定义模块并按照[自定义模块集成](#自定义模块集成)中的描述导入它。

### 如何为不同大小（nano、small、medium 等）缩放模型？

使用 YAML 中的 [`scales` 部分](#参数部分)定义深度、宽度和最大通道的缩放因子。当您加载带有缩放后缀的基础 YAML 文件（例如 `yolo11n.yaml`）时，模型将自动应用这些。

### `[from, repeats, module, args]` 格式是什么意思？

此格式指定每层的构建方式：

- `from`：输入源
- `repeats`：重复模块的次数
- `module`：层类型
- `args`：模块的参数

### 如何排除通道不匹配错误？

检查一层的输出通道是否与下一层的预期输入通道匹配。使用 `print(model.model.model)` 检查模型的架构。

### 在哪里可以找到可用模块及其参数的列表？

查看 [`ultralytics/nn/modules` 目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules)中的源代码以获取所有可用模块及其参数。

### 如何将自定义模块添加到我的 YAML 配置中？

在源代码中定义您的模块，按照[源代码修改](#源代码修改)中所示导入它，并在 YAML 文件中按名称引用它。

### 我可以使用自定义 YAML 的预训练权重吗？

可以，您可以使用 `model.load("path/to/weights")` 从预训练检查点加载权重。但是，只有匹配层的权重才能成功加载。

### 如何验证我的模型配置？

使用 `model.info()` 检查 FLOPs 计数是否为非零。有效模型应显示非零 FLOPs 计数。如果是零，请按照[调试技巧](#调试技巧)中的建议查找问题。
