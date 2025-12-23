---
comments: true
description: 使用 DVCLive 解锁无缝的 YOLO11 跟踪。了解如何记录、可视化和分析实验以优化 ML 模型性能。
keywords: YOLO11, DVCLive, 实验跟踪, 机器学习, 模型训练, 数据可视化, Git 集成
---

# 使用 DVCLive 进行高级 YOLO11 实验跟踪

[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)中的实验跟踪对于模型开发和评估至关重要。它涉及记录和分析来自众多训练运行的各种参数、指标和结果。这个过程对于理解模型性能和做出数据驱动的决策来优化和改进模型至关重要。

将 DVCLive 与 [Ultralytics YOLO11](https://www.ultralytics.com/) 集成改变了实验跟踪和管理的方式。这种集成提供了一个无缝的解决方案，用于自动记录关键实验细节、比较不同运行的结果，以及可视化数据进行深入分析。在本指南中，我们将了解如何使用 DVCLive 来简化这个过程。

## DVCLive

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/dvclive-overview.avif" alt="DVCLive 概述">
</p>

[DVCLive](https://doc.dvc.org/dvclive) 由 DVC 开发，是一个创新的开源工具，用于机器学习中的实验跟踪。它与 Git 和 DVC 无缝集成，自动记录模型参数和训练指标等关键实验数据。DVCLive 设计简单，可轻松比较和分析多次运行，通过直观的[数据可视化](https://www.ultralytics.com/glossary/data-visualization)和分析工具提高机器学习项目的效率。

## 使用 DVCLive 进行 YOLO11 训练

可以使用 DVCLive 有效监控 YOLO11 训练会话。此外，DVC 提供了用于可视化这些实验的集成功能，包括生成报告，可以比较所有跟踪实验的指标图，提供训练过程的全面视图。

## 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 和 DVCLive 所需的包
        pip install ultralytics dvclive
        ```

有关安装过程的详细说明和最佳实践，请务必查看我们的 [YOLO11 安装指南](../quickstart.md)。在安装 YOLO11 所需的包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)以获取解决方案和提示。

## 配置 DVCLive

安装必要的包后，下一步是设置和配置您的环境以及所需的凭据。此设置确保 DVCLive 顺利集成到您现有的工作流程中。

首先初始化一个 Git 仓库，因为 Git 在代码和 DVCLive 配置的版本控制中起着至关重要的作用。

!!! tip "初始环境设置"

    === "CLI"

        ```bash
        # 初始化 Git 仓库
        git init -q

        # 使用您的详细信息配置 Git
        git config --local user.email "you@example.com"
        git config --local user.name "Your Name"

        # 在您的项目中初始化 DVCLive
        dvc init -q

        # 将 DVCLive 设置提交到您的 Git 仓库
        git commit -m "DVC init"
        ```

在这些命令中，确保将 "you@example.com" 替换为与您的 Git 账户关联的电子邮件地址，将 "Your Name" 替换为您的 Git 账户用户名。

## 使用

在深入了解使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型系列](../models/index.md)。这将帮助您选择最适合项目需求的模型。

### 使用 DVCLive 训练 YOLO11 模型

开始运行您的 YOLO11 训练会话。您可以使用不同的模型配置和训练参数来满足您的项目需求。例如：

```bash
# 使用不同配置的 YOLO11 训练命令示例
yolo train model=yolo11n.pt data=coco8.yaml epochs=5 imgsz=512
yolo train model=yolo11n.pt data=coco8.yaml epochs=5 imgsz=640
```

根据您的具体需求调整 model、data、[epochs](https://www.ultralytics.com/glossary/epoch) 和 imgsz 参数。有关模型训练过程和最佳实践的详细了解，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

### 使用 DVCLive 监控实验

DVCLive 通过启用关键指标的跟踪和可视化来增强训练过程。安装后，Ultralytics YOLO11 会自动与 DVCLive 集成进行实验跟踪，您可以稍后分析以获得性能洞察。有关训练期间使用的特定性能指标的全面了解，请务必探索[我们关于性能指标的详细指南](../guides/yolo-performance-metrics.md)。


### 分析结果

YOLO11 训练会话完成后，您可以利用 DVCLive 强大的可视化工具对结果进行深入分析。DVCLive 的集成确保所有训练指标都被系统地记录，便于对模型性能进行全面评估。

要开始分析，您可以使用 DVC 的 API 提取实验数据，并使用 Pandas 进行处理以便于处理和可视化：

```python
import dvc.api
import pandas as pd

# 定义感兴趣的列
columns = ["Experiment", "epochs", "imgsz", "model", "metrics.mAP50-95(B)"]

# 检索实验数据
df = pd.DataFrame(dvc.api.exp_show(), columns=columns)

# 清理数据
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 显示 DataFrame
print(df)
```

上述代码片段的输出提供了使用 YOLO11 模型进行的不同实验的清晰表格视图。每行代表一次不同的训练运行，详细说明实验名称、训练周期数、图像尺寸（imgsz）、使用的特定模型和 mAP50-95(B) 指标。此指标对于评估模型的[准确率](https://www.ultralytics.com/glossary/accuracy)至关重要，值越高表示性能越好。

#### 使用 Plotly 可视化结果

为了对实验结果进行更具交互性和视觉效果的分析，您可以使用 Plotly 的平行坐标图。这种类型的图对于理解不同参数和指标之间的关系和权衡特别有用。

```python
from plotly.express import parallel_coordinates

# 创建平行坐标图
fig = parallel_coordinates(df, columns, color="metrics.mAP50-95(B)")

# 显示图表
fig.show()
```

上述代码片段的输出生成一个图表，将直观地表示 epochs、图像尺寸、模型类型及其对应的 mAP50-95(B) 分数之间的关系，使您能够发现实验数据中的趋势和模式。

#### 使用 DVC 生成比较可视化

DVC 提供了一个有用的命令来为您的实验生成比较图。这对于比较不同模型在各种训练运行中的性能特别有帮助。

```bash
# 生成 DVC 比较图
dvc plots diff $(dvc exp list --names-only)
```

执行此命令后，DVC 会生成比较不同实验指标的图表，这些图表保存为 HTML 文件。下面是一个示例图像，说明了此过程生成的典型图表。该图像展示了各种图表，包括表示 mAP、[召回率](https://www.ultralytics.com/glossary/recall)、[精确度](https://www.ultralytics.com/glossary/precision)、损失值等的图表，提供了关键性能指标的视觉概览：

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/dvclive-comparative-plots.avif" alt="DVCLive 图表">
</p>

### 显示 DVC 图表

如果您使用的是 Jupyter Notebook 并且想要显示生成的 DVC 图表，您可以使用 IPython 显示功能。

```python
from IPython.display import HTML

# 将 DVC 图表显示为 HTML
HTML(filename="./dvc_plots/index.html")
```

此代码将直接在您的 Jupyter Notebook 中渲染包含 DVC 图表的 HTML 文件，提供一种简单方便的方式来分析可视化的实验数据。

### 做出数据驱动的决策

使用从这些可视化中获得的洞察来做出关于模型优化、[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)和其他修改的明智决策，以增强模型的性能。

### 迭代实验

根据您的分析，迭代您的实验。调整模型配置、训练参数甚至数据输入，并重复训练和分析过程。这种迭代方法是优化模型以获得最佳性能的关键。

## 总结

本指南引导您完成了将 DVCLive 与 Ultralytics 的 YOLO11 集成的过程。您已经学会了如何利用 DVCLive 的强大功能进行详细的实验监控、有效的可视化和在机器学习工作中进行有洞察力的分析。

有关使用的更多详细信息，请访问 [DVCLive 的官方文档](https://doc.dvc.org/dvclive/ml-frameworks/yolo)。

此外，通过访问 [Ultralytics 集成指南页面](../integrations/index.md)探索更多 Ultralytics 的集成和功能，这是一个包含大量资源和见解的集合。

## 常见问题

### 如何将 DVCLive 与 Ultralytics YOLO11 集成进行实验跟踪？

将 DVCLive 与 Ultralytics YOLO11 集成非常简单。首先安装必要的包：

!!! example "安装"

    === "CLI"

        ```bash
        pip install ultralytics dvclive
        ```

接下来，初始化一个 Git 仓库并在您的项目中配置 DVCLive：

!!! example "初始环境设置"

    === "CLI"

        ```bash
        git init -q
        git config --local user.email "you@example.com"
        git config --local user.name "Your Name"
        dvc init -q
        git commit -m "DVC init"
        ```

有关详细的设置说明，请遵循我们的 [YOLO11 安装指南](../quickstart.md)。

### 为什么应该使用 DVCLive 跟踪 YOLO11 实验？

将 DVCLive 与 YOLO11 一起使用提供了几个优势，例如：

- **自动记录**：DVCLive 自动记录模型参数和指标等关键实验细节。
- **轻松比较**：便于比较不同运行的结果。
- **可视化工具**：利用 DVCLive 强大的数据可视化功能进行深入分析。

有关更多详细信息，请参阅我们关于 [YOLO11 模型训练](../modes/train.md)和 [YOLO 性能指标](../guides/yolo-performance-metrics.md)的指南，以最大化您的实验跟踪效率。

### DVCLive 如何改善我对 YOLO11 训练会话的结果分析？

完成 YOLO11 训练会话后，DVCLive 帮助有效地可视化和分析结果。加载和显示实验数据的示例代码：

```python
import dvc.api
import pandas as pd

# 定义感兴趣的列
columns = ["Experiment", "epochs", "imgsz", "model", "metrics.mAP50-95(B)"]

# 检索实验数据
df = pd.DataFrame(dvc.api.exp_show(), columns=columns)

# 清理数据
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 显示 DataFrame
print(df)
```

要交互式地可视化结果，请使用 Plotly 的平行坐标图：

```python
from plotly.express import parallel_coordinates

fig = parallel_coordinates(df, columns, color="metrics.mAP50-95(B)")
fig.show()
```

有关更多示例和最佳实践，请参阅我们关于[使用 DVCLive 进行 YOLO11 训练](#使用-dvclive-进行-yolo11-训练)的指南。

### 为 DVCLive 和 YOLO11 集成配置环境的步骤是什么？

要为 DVCLive 和 YOLO11 的顺利集成配置您的环境，请按照以下步骤操作：

1. **安装所需的包**：使用 `pip install ultralytics dvclive`。
2. **初始化 Git 仓库**：运行 `git init -q`。
3. **设置 DVCLive**：执行 `dvc init -q`。
4. **提交到 Git**：使用 `git commit -m "DVC init"`。

这些步骤确保实验跟踪的正确版本控制和设置。有关深入的配置详细信息，请访问我们的[配置指南](../quickstart.md)。

### 如何使用 DVCLive 可视化 YOLO11 实验结果？

DVCLive 提供了强大的工具来可视化 YOLO11 实验的结果。以下是如何生成比较图：

!!! example "生成比较图"

    === "CLI"

        ```bash
        dvc plots diff $(dvc exp list --names-only)
        ```

要在 Jupyter Notebook 中显示这些图表，请使用：

```python
from IPython.display import HTML

# 将图表显示为 HTML
HTML(filename="./dvc_plots/index.html")
```

这些可视化有助于识别趋势和优化模型性能。查看我们关于 [YOLO11 实验分析](#分析结果)的详细指南，了解全面的步骤和示例。
