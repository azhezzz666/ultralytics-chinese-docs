---
comments: true
description: 通过我们的综合指南掌握 Ultralytics YOLO 的超参数调优，优化模型性能。立即提升您的机器学习模型！
keywords: Ultralytics YOLO, 超参数调优, 机器学习, 模型优化, 遗传算法, 学习率, 批量大小, 轮次
---

# Ultralytics YOLO [超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)指南

## 简介

超参数调优不仅仅是一次性设置，而是一个旨在优化[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型性能指标（如准确率、精确度和召回率）的迭代过程。在 Ultralytics YOLO 的上下文中，这些超参数可以从学习率到架构细节（如使用的层数或激活函数类型）不等。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j0MOGKBqx7E"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何调优超参数以获得更好的模型性能 🚀
</p>

### 什么是超参数？

超参数是算法的高级结构设置。它们在训练阶段之前设置，并在训练期间保持不变。以下是 Ultralytics YOLO 中一些常见的调优超参数：

- **学习率** `lr0`：确定在向[损失函数](https://www.ultralytics.com/glossary/loss-function)最小值移动时每次迭代的步长。
- **[批量大小](https://www.ultralytics.com/glossary/batch-size)** `batch`：在一次前向传递中同时处理的图像数量。
- **[轮次](https://www.ultralytics.com/glossary/epoch)数** `epochs`：一个轮次是所有训练样本的一次完整前向和反向传递。
- **架构细节**：如通道数、层数、激活函数类型等。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/hyperparameter-tuning-visual.avif" alt="超参数调优可视化">
</p>

有关 YOLO11 中使用的增强超参数的完整列表，请参阅[配置页面](../usage/cfg.md#augmentation-settings)。

### 遗传进化和变异

Ultralytics YOLO 使用[遗传算法](https://en.wikipedia.org/wiki/Genetic_algorithm)来优化超参数。遗传算法受自然选择和遗传机制的启发。

- **变异**：在 Ultralytics YOLO 的上下文中，变异通过对现有超参数应用小的随机变化来帮助局部搜索超参数空间，产生新的候选者进行评估。
- **交叉**：虽然交叉是一种流行的遗传算法技术，但目前在 Ultralytics YOLO 的超参数调优中未使用。重点主要是通过变异来生成新的超参数集。

## 准备超参数调优

在开始调优过程之前，重要的是：

1. **确定指标**：确定您将用于评估模型性能的指标。这可以是 AP50、F1 分数或其他指标。
2. **设置调优预算**：定义您愿意分配多少计算资源。超参数调优可能是计算密集型的。

## 涉及的步骤

### 初始化超参数

从一组合理的初始超参数开始。这可以是 Ultralytics YOLO 设置的默认超参数，也可以是基于您的领域知识或先前实验的内容。

### 变异超参数

使用 `_mutate` 方法根据现有集合生成一组新的超参数。[Tuner 类](https://docs.ultralytics.com/reference/engine/tuner/)会自动处理此过程。

### 训练模型

使用变异后的超参数集进行训练。然后使用您选择的指标评估训练性能。

### 评估模型

使用 AP50、F1 分数或自定义指标等指标来评估模型的性能。[评估过程](https://docs.ultralytics.com/modes/val/)有助于确定当前超参数是否优于之前的超参数。

### 记录结果

记录性能指标和相应的超参数以供将来参考至关重要。Ultralytics YOLO 会自动将这些结果保存为 CSV 格式。

### 重复

该过程重复进行，直到达到设定的迭代次数或性能指标令人满意。每次迭代都建立在从先前运行中获得的知识之上。

## 默认搜索空间描述

下表列出了 YOLO11 中超参数调优的默认搜索空间参数。每个参数都有一个由元组 `(min, max)` 定义的特定值范围。

| 参数         | 类型    | 值范围    | 描述                                                                                                      |
| ----------------- | ------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `lr0`             | `float` | `(1e-5, 1e-1)` | 训练开始时的初始学习率。较低的值提供更稳定的训练但收敛较慢 |
| `lrf`             | `float` | `(0.01, 1.0)`  | 最终学习率因子，作为 lr0 的分数。控制训练期间学习率下降的程度   |
| `momentum`        | `float` | `(0.6, 0.98)`  | SGD 动量因子。较高的值有助于保持一致的梯度方向，可以加速收敛      |
| `weight_decay`    | `float` | `(0.0, 0.001)` | L2 正则化因子，防止过拟合。较大的值强制执行更强的正则化                   |
| `warmup_epochs`   | `float` | `(0.0, 5.0)`   | 线性学习率预热的轮次数。有助于防止早期训练不稳定                       |
| `warmup_momentum` | `float` | `(0.0, 0.95)`  | 预热阶段的初始动量。逐渐增加到最终动量值                            |
| `box`             | `float` | `(0.02, 0.2)`  | 总损失函数中的边界框损失权重。平衡边界框回归与分类                   |
| `cls`             | `float` | `(0.2, 4.0)`   | 总损失函数中的分类损失权重。较高的值强调正确的类别预测          |
| `hsv_h`           | `float` | `(0.0, 0.1)`   | HSV 颜色空间中的随机色相增强范围。帮助模型泛化到颜色变化                 |
| `hsv_s`           | `float` | `(0.0, 0.9)`   | HSV 空间中的随机饱和度增强范围。模拟不同的光照条件                       |
| `hsv_v`           | `float` | `(0.0, 0.9)`   | 随机明度（亮度）增强范围。帮助模型处理不同的曝光水平                       |
| `degrees`         | `float` | `(0.0, 45.0)`  | 最大旋转增强角度。帮助模型对物体方向不变                     |
| `translate`       | `float` | `(0.0, 0.9)`   | 最大平移增强，作为图像大小的分数。提高对物体位置的鲁棒性               |
| `scale`           | `float` | `(0.0, 0.9)`   | 随机缩放增强范围。帮助模型检测不同大小的物体                                 |
| `shear`           | `float` | `(0.0, 10.0)`  | 最大剪切增强角度。为训练图像添加类似透视的失真                      |
| `perspective`     | `float` | `(0.0, 0.001)` | 随机透视增强范围。模拟不同的视角                                        |
| `flipud`          | `float` | `(0.0, 1.0)`   | 训练期间垂直翻转图像的概率。对俯视/航拍图像有用                           |
| `fliplr`          | `float` | `(0.0, 1.0)`   | 水平翻转图像的概率。帮助模型对物体方向不变                           |
| `mosaic`          | `float` | `(0.0, 1.0)`   | 使用马赛克增强的概率，将 4 张图像组合在一起。对小目标检测特别有用  |
| `mixup`           | `float` | `(0.0, 1.0)`   | 使用 mixup 增强的概率，混合两张图像。可以提高模型鲁棒性                   |
| `copy_paste`      | `float` | `(0.0, 1.0)`   | 使用复制粘贴增强的概率。有助于提高实例分割性能                    |

## 自定义搜索空间示例

以下是如何定义搜索空间并使用 `model.tune()` 方法利用 `Tuner` 类对 COCO8 上的 YOLO11n 进行 30 个轮次的超参数调优，使用 AdamW 优化器，并跳过绘图、检查点和验证（除了最后一个轮次），以加快调优速度。

!!! warning

    此示例仅用于**演示**。从短期或小规模调优运行中得出的超参数很少对实际训练是最优的。在实践中，调优应在与完整训练类似的设置下进行——包括可比较的数据集、轮次和增强——以确保可靠和可迁移的结果。快速调优可能会使参数偏向于更快的收敛或短期验证收益，而这些不会泛化。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 定义搜索空间
        search_space = {
            "lr0": (1e-5, 1e-1),
            "degrees": (0.0, 45.0),
        }

        # 在 COCO8 上调优超参数 30 个轮次
        model.tune(
            data="coco8.yaml",
            epochs=30,
            iterations=300,
            optimizer="AdamW",
            space=search_space,
            plots=False,
            save=False,
            val=False,
        )
        ```

## 恢复中断的超参数调优会话

您可以通过传递 `resume=True` 来恢复中断的超参数调优会话。您可以选择传递在 `runs/{task}` 下使用的目录 `name` 来恢复。否则，它将恢复最后一个中断的会话。您还需要提供所有先前的训练参数，包括 `data`、`epochs`、`iterations` 和 `space`。

!!! example "在 `model.tune()` 中使用 `resume=True`"

    ```python
    from ultralytics import YOLO

    # 定义 YOLO 模型
    model = YOLO("yolo11n.pt")

    # 定义搜索空间
    search_space = {
        "lr0": (1e-5, 1e-1),
        "degrees": (0.0, 45.0),
    }

    # 恢复先前的运行
    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, resume=True)

    # 恢复名为 'tune_exp' 的调优运行
    results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, name="tune_exp", resume=True)
    ```

## 结果

成功完成超参数调优过程后，您将获得几个封装调优结果的文件和目录。以下描述了每个文件：

### 文件结构

结果的目录结构如下所示。像 `train1/` 这样的训练目录包含单独的调优迭代，即使用一组超参数训练的一个模型。`tune/` 目录包含所有单独模型训练的调优结果：

```plaintext
runs/
└── detect/
    ├── train1/
    ├── train2/
    ├── ...
    └── tune/
        ├── best_hyperparameters.yaml
        ├── best_fitness.png
        ├── tune_results.csv
        ├── tune_scatter_plots.png
        └── weights/
            ├── last.pt
            └── best.pt
```

### 文件描述

#### best_hyperparameters.yaml

此 YAML 文件包含调优过程中找到的最佳性能超参数。您可以使用此文件以这些优化设置初始化未来的训练。

- **格式**：YAML
- **用途**：超参数结果
- **示例**：

    ```yaml
    # 558/900 迭代完成 ✅ (45536.81s)
    # 结果保存到 /usr/src/ultralytics/runs/detect/tune
    # 在迭代 498 观察到最佳适应度=0.64297
    # 最佳适应度指标为 {'metrics/precision(B)': 0.87247, 'metrics/recall(B)': 0.71387, 'metrics/mAP50(B)': 0.79106, 'metrics/mAP50-95(B)': 0.62651, 'val/box_loss': 2.79884, 'val/cls_loss': 2.72386, 'val/dfl_loss': 0.68503, 'fitness': 0.64297}
    # 最佳适应度模型是 /usr/src/ultralytics/runs/detect/train498
    # 最佳适应度超参数如下所示。

    lr0: 0.00269
    lrf: 0.00288
    momentum: 0.73375
    weight_decay: 0.00015
    warmup_epochs: 1.22935
    warmup_momentum: 0.1525
    box: 18.27875
    cls: 1.32899
    dfl: 0.56016
    hsv_h: 0.01148
    hsv_s: 0.53554
    hsv_v: 0.13636
    degrees: 0.0
    translate: 0.12431
    scale: 0.07643
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.08631
    mosaic: 0.42551
    mixup: 0.0
    copy_paste: 0.0
    ```

#### best_fitness.png

这是一个显示适应度（通常是 AP50 等性能指标）与迭代次数关系的图表。它帮助您可视化遗传算法随时间的表现。

- **格式**：PNG
- **用途**：性能可视化

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/best-fitness.avif" alt="超参数调优适应度与迭代">
</p>

#### tune_results.csv

一个 CSV 文件，包含调优期间每次迭代的详细结果。文件中的每一行代表一次迭代，包括适应度分数、[精确度](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)等指标，以及使用的超参数。

- **格式**：CSV
- **用途**：每次迭代结果跟踪。
- **示例**：
    ```csv
      fitness,lr0,lrf,momentum,weight_decay,warmup_epochs,warmup_momentum,box,cls,dfl,hsv_h,hsv_s,hsv_v,degrees,translate,scale,shear,perspective,flipud,fliplr,mosaic,mixup,copy_paste
      0.05021,0.01,0.01,0.937,0.0005,3.0,0.8,7.5,0.5,1.5,0.015,0.7,0.4,0.0,0.1,0.5,0.0,0.0,0.0,0.5,1.0,0.0,0.0
      0.07217,0.01003,0.00967,0.93897,0.00049,2.79757,0.81075,7.5,0.50746,1.44826,0.01503,0.72948,0.40658,0.0,0.0987,0.4922,0.0,0.0,0.0,0.49729,1.0,0.0,0.0
      0.06584,0.01003,0.00855,0.91009,0.00073,3.42176,0.95,8.64301,0.54594,1.72261,0.01503,0.59179,0.40658,0.0,0.0987,0.46955,0.0,0.0,0.0,0.49729,0.80187,0.0,0.0
    ```

#### tune_scatter_plots.png

此文件包含从 `tune_results.csv` 生成的散点图，帮助您可视化不同超参数与性能指标之间的关系。请注意，初始化为 0 的超参数将不会被调优，如下面的 `degrees` 和 `shear`。

- **格式**：PNG
- **用途**：探索性数据分析

<p align="center">
  <img width="1000" src="https://github.com/ultralytics/docs/releases/download/0/tune-scatter-plots.avif" alt="超参数调优散点图">
</p>

#### weights/

此目录包含超参数调优过程中最后一次和最佳迭代的已保存 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型。

- **`last.pt`**：last.pt 是训练最后一个轮次的权重。
- **`best.pt`**：达到最佳适应度分数的迭代的 best.pt 权重。

使用这些结果，您可以为未来的模型训练和分析做出更明智的决策。随时查阅这些工件，了解您的模型表现如何以及如何进一步改进它。

## 总结

由于其基于遗传算法的变异方法，Ultralytics YOLO 中的超参数调优过程既简化又强大。遵循本指南中概述的步骤将帮助您系统地调优模型以实现更好的性能。

### 延伸阅读

1. [维基百科上的超参数优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
2. [YOLOv5 超参数进化指南](../yolov5/tutorials/hyperparameter_evolution.md)
3. [使用 Ray Tune 和 YOLO11 进行高效超参数调优](../integrations/ray-tune.md)

如需更深入的了解，您可以探索 [`Tuner` 类](https://docs.ultralytics.com/reference/engine/tuner/)源代码和相关文档。如果您有任何问题、功能请求或需要进一步帮助，请随时通过 [GitHub](https://github.com/ultralytics/ultralytics/issues/new/choose) 或 [Discord](https://discord.com/invite/ultralytics) 与我们联系。

## 常见问题

### 如何在超参数调优期间优化 Ultralytics YOLO 的[学习率](https://www.ultralytics.com/glossary/learning-rate)？

要优化 Ultralytics YOLO 的学习率，首先使用 `lr0` 参数设置初始学习率。常见值范围从 `0.001` 到 `0.01`。在超参数调优过程中，此值将被变异以找到最佳设置。您可以使用 `model.tune()` 方法来自动化此过程。例如：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8 上调优超参数 30 个轮次
        model.tune(data="coco8.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

有关更多详细信息，请查看 [Ultralytics YOLO 配置页面](../usage/cfg.md#augmentation-settings)。

### 在 YOLO11 中使用遗传算法进行超参数调优有什么好处？

Ultralytics YOLO11 中的遗传算法提供了一种强大的方法来探索超参数空间，从而实现高度优化的模型性能。主要好处包括：

- **高效搜索**：像变异这样的遗传算法可以快速探索大量超参数集。
- **避免局部最小值**：通过引入随机性，它们有助于避免局部最小值，确保更好的全局优化。
- **性能指标**：它们根据 AP50 和 F1 分数等性能指标进行调整。

要了解遗传算法如何优化超参数，请查看[超参数进化指南](../yolov5/tutorials/hyperparameter_evolution.md)。

### Ultralytics YOLO 的超参数调优过程需要多长时间？

Ultralytics YOLO 超参数调优所需的时间在很大程度上取决于几个因素，如数据集的大小、模型架构的复杂性、迭代次数和可用的计算资源。例如，在 COCO8 等数据集上对 YOLO11n 进行 30 个轮次的调优可能需要几个小时到几天，具体取决于硬件。

为了有效管理调优时间，请事先定义明确的调优预算（[内部部分链接](#准备超参数调优)）。这有助于平衡资源分配和优化目标。

### 在 YOLO 超参数调优期间，我应该使用哪些指标来评估模型性能？

在 YOLO 超参数调优期间评估模型性能时，您可以使用几个关键指标：

- **AP50**：IoU 阈值为 0.50 时的平均精度。
- **F1 分数**：精确度和召回率的调和平均值。
- **精确度和召回率**：单独的指标，表示模型在识别真阳性与假阳性和假阴性方面的[准确率](https://www.ultralytics.com/glossary/accuracy)。

这些指标帮助您了解模型性能的不同方面。有关全面概述，请参阅 [Ultralytics YOLO 性能指标](../guides/yolo-performance-metrics.md)指南。

### 我可以使用 Ray Tune 对 YOLO11 进行高级超参数优化吗？

是的，Ultralytics YOLO11 与 [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 集成，用于高级超参数优化。Ray Tune 提供复杂的搜索算法，如贝叶斯优化和 Hyperband，以及并行执行功能来加速调优过程。

要将 Ray Tune 与 YOLO11 一起使用，只需在 `model.tune()` 方法调用中设置 `use_ray=True` 参数。有关更多详细信息和示例，请查看 [Ray Tune 集成指南](../integrations/ray-tune.md)。
