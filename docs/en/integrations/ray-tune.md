---
comments: true
description: 使用 Ray Tune 和 YOLO11 优化模型性能。学习使用高级搜索策略、并行处理和早停进行高效超参数调优。
keywords: YOLO11, Ray Tune, 超参数调优, 模型优化, 机器学习, 深度学习, AI, Ultralytics, Weights & Biases
---

# 使用 Ray Tune 和 YOLO11 进行高效超参数调优

超参数调优对于通过发现最佳超参数集来实现峰值模型性能至关重要。这涉及使用不同的超参数运行试验并评估每个试验的性能。

## 使用 Ultralytics YOLO11 和 Ray Tune 加速调优

[Ultralytics YOLO11](https://www.ultralytics.com/) 集成了 Ray Tune 进行超参数调优，简化了 YOLO11 模型超参数的优化。使用 Ray Tune，你可以利用高级搜索策略、并行处理和早停来加速调优过程。

### Ray Tune

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/ray-tune-overview.avif" alt="Ray Tune 概览">
</p>

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 是一个为效率和灵活性设计的超参数调优库。它支持各种搜索策略、并行处理和早停策略，并与流行的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)框架（包括 Ultralytics YOLO11）无缝集成。

### 与 Weights & Biases 集成

YOLO11 还允许与 [Weights & Biases](https://wandb.ai/site) 进行可选集成，以监控调优过程。

## 安装

要安装所需的包，运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装和更新 Ultralytics 和 Ray Tune 包
        pip install -U ultralytics "ray[tune]"

        # 可选安装 W&B 用于日志记录
        pip install wandb
        ```

## 用法

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 开始在 COCO8 数据集上调优 YOLO11n 训练的超参数
        result_grid = model.tune(data="coco8.yaml", use_ray=True)
        ```

## `tune()` 方法参数

YOLO11 中的 `tune()` 方法提供了一个易于使用的接口，用于使用 Ray Tune 进行超参数调优。它接受几个参数，允许你自定义调优过程。以下是每个参数的详细说明：

| 参数 | 类型 | 描述 | 默认值 |
| --------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `data` | `str` | 运行调优器的数据集配置文件（YAML 格式）。此文件应指定训练和[验证数据](https://www.ultralytics.com/glossary/validation-data)路径，以及其他数据集特定设置。 | |
| `space` | `dict, optional` | 定义 Ray Tune 超参数搜索空间的字典。每个键对应一个超参数名称，值指定调优期间要探索的值范围。如果未提供，YOLO11 使用包含各种超参数的默认搜索空间。 | |
| `grace_period` | `int, optional` | Ray Tune 中 [ASHA 调度器](https://docs.ray.io/en/latest/tune/api/schedulers.html)的宽限期（以 [epoch](https://www.ultralytics.com/glossary/epoch) 为单位）。调度器在此 epoch 数之前不会终止任何试验，允许模型在做出早停决定之前进行一些最小训练。 | 10 |
| `gpu_per_trial` | `int, optional` | 调优期间每个试验分配的 GPU 数量。这有助于管理 GPU 使用，特别是在多 GPU 环境中。如果未提供，调优器将使用所有可用的 GPU。 | `None` |
| `iterations` | `int, optional` | 调优期间运行的最大试验数。此参数有助于控制测试的超参数组合总数，确保调优过程不会无限期运行。 | 10 |
| `**train_args` | `dict, optional` | 调优期间传递给 `train()` 方法的额外参数。这些参数可以包括训练 epoch 数、[批量大小](https://www.ultralytics.com/glossary/batch-size)和其他训练特定配置等设置。 | {} |

通过自定义这些参数，你可以微调超参数优化过程以适应你的特定需求和可用的计算资源。

## 默认搜索空间描述

下表列出了使用 Ray Tune 进行 YOLO11 超参数调优的默认搜索空间参数。每个参数都有一个由 `tune.uniform()` 定义的特定值范围。

| 参数 | 范围 | 描述 |
| ----------------- | -------------------------- | -------------------------------------------------------------------------------- |
| `lr0` | `tune.uniform(1e-5, 1e-1)` | 控制优化期间步长的初始学习率。较高的值加速训练但可能导致不稳定。 |
| `lrf` | `tune.uniform(0.01, 1.0)` | 确定训练结束时学习率降低多少的最终学习率因子。 |
| `momentum` | `tune.uniform(0.6, 0.98)` | 优化器的动量因子，有助于加速训练并克服局部最小值。 |
| `weight_decay` | `tune.uniform(0.0, 0.001)` | 通过惩罚大权重值来防止过拟合的正则化参数。 |
| `warmup_epochs` | `tune.uniform(0.0, 5.0)` | 学习率逐渐增加以稳定早期训练的 epoch 数。 |
| `warmup_momentum` | `tune.uniform(0.0, 0.95)` | 在预热期间逐渐增加的初始动量值。 |
| `box` | `tune.uniform(0.02, 0.2)` | 边界框损失组件的权重，平衡模型中的定位准确性。 |
| `cls` | `tune.uniform(0.2, 4.0)` | 分类损失组件的权重，平衡模型中的类别预测准确性。 |
| `hsv_h` | `tune.uniform(0.0, 0.1)` | 引入颜色变化以帮助模型泛化的色调增强范围。 |
| `hsv_s` | `tune.uniform(0.0, 0.9)` | 改变颜色强度以提高鲁棒性的饱和度增强范围。 |
| `hsv_v` | `tune.uniform(0.0, 0.9)` | 帮助模型在各种光照条件下表现的亮度增强范围。 |
| `degrees` | `tune.uniform(0.0, 45.0)` | 以度为单位的旋转增强范围，提高对旋转对象的识别。 |
| `translate` | `tune.uniform(0.0, 0.9)` | 水平和垂直移动图像的平移增强范围。 |
| `scale` | `tune.uniform(0.0, 0.9)` | 模拟不同距离对象的缩放增强范围。 |
| `shear` | `tune.uniform(0.0, 10.0)` | 以度为单位的剪切增强范围，模拟透视变化。 |
| `perspective` | `tune.uniform(0.0, 0.001)` | 模拟 3D 视角变化的透视增强范围。 |
| `flipud` | `tune.uniform(0.0, 1.0)` | 垂直翻转增强概率，增加数据集多样性。 |
| `fliplr` | `tune.uniform(0.0, 1.0)` | 水平翻转增强概率，对对称对象有用。 |
| `mosaic` | `tune.uniform(0.0, 1.0)` | 将四张图像组合成一个训练样本的马赛克增强概率。 |
| `mixup` | `tune.uniform(0.0, 1.0)` | 将两张图像及其标签混合在一起的 Mixup 增强概率。 |
| `cutmix` | `tune.uniform(0.0, 1.0)` | 组合图像区域同时保持局部特征的 Cutmix 增强概率，提高对部分遮挡对象的检测。 |
| `copy_paste` | `tune.uniform(0.0, 1.0)` | 在图像之间转移对象以增加实例多样性的复制粘贴增强概率。 |

## 自定义搜索空间示例

在此示例中，我们演示如何使用 Ray Tune 和 YOLO11 的自定义搜索空间进行超参数调优。通过提供自定义搜索空间，你可以将调优过程集中在感兴趣的特定超参数上。

!!! example "用法"

    ```python
    from ray import tune

    from ultralytics import YOLO

    # 定义 YOLO 模型
    model = YOLO("yolo11n.pt")

    # 在模型上运行 Ray Tune
    result_grid = model.tune(
        data="coco8.yaml",
        space={"lr0": tune.uniform(1e-5, 1e-1)},
        epochs=50,
        use_ray=True,
    )
    ```

在上面的代码片段中，我们使用 "yolo11n.pt" 预训练权重创建一个 YOLO 模型。然后，我们调用 `tune()` 方法，使用 "coco8.yaml" 指定数据集配置。我们使用字典为初始学习率 `lr0` 提供自定义搜索空间，键为 "lr0"，值为 `tune.uniform(1e-5, 1e-1)`。最后，我们将额外的训练参数（如 epoch 数）直接传递给 tune 方法，如 `epochs=50`。

## 恢复中断的超参数调优会话

你可以通过传递 `resume=True` 来恢复中断的 Ray Tune 会话。你可以选择传递 Ray Tune 在 `runs/{task}` 下使用的目录 `name` 来恢复。否则，它将恢复最后一个中断的会话。你不需要再次提供 `iterations` 和 `space`，但需要再次提供其余的训练参数，包括 `data` 和 `epochs`。

!!! example "使用 `resume=True` 与 `model.tune()`"

    ```python
    from ultralytics import YOLO

    # 定义 YOLO 模型
    model = YOLO("yolo11n.pt")

    # 恢复之前的运行
    results = model.tune(use_ray=True, data="coco8.yaml", epochs=50, resume=True)

    # 恢复名为 'tune_exp_2' 的 Ray Tune 运行
    results = model.tune(use_ray=True, data="coco8.yaml", epochs=50, name="tune_exp_2", resume=True)
    ```

## 处理 Ray Tune 结果

运行 Ray Tune 超参数调优实验后，你可能想对获得的结果进行各种分析。本指南将带你了解处理和分析这些结果的常见工作流程。

### 从目录加载调优实验结果

使用 `tuner.fit()` 运行调优实验后，你可以从目录加载结果。这很有用，特别是如果你在初始训练脚本退出后执行分析。

```python
experiment_path = f"{storage_path}/{exp_name}"
print(f"从 {experiment_path} 加载结果...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
result_grid = restored_tuner.get_results()
```

### 基本实验级分析

获取试验执行情况的概览。你可以快速检查试验期间是否有任何错误。

```python
if result_grid.errors:
    print("一个或多个试验失败！")
else:
    print("没有错误！")
```

### 基本试验级分析

访问单个试验的超参数配置和最后报告的指标。

```python
for i, result in enumerate(result_grid):
    print(f"试验 #{i}: 配置: {result.config}, 最后报告的指标: {result.metrics}")
```

## 总结

在本指南中，我们介绍了使用 Ultralytics 和 Ray Tune 分析实验结果的常见工作流程。关键步骤包括从目录加载实验结果、执行基本实验级和试验级分析以及绘制指标。

通过查看 Ray Tune 的[分析结果](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html)文档页面进一步探索，以充分利用你的超参数调优实验。

## 常见问题

### 如何使用 Ray Tune 调优 YOLO11 模型的超参数？

要使用 Ray Tune 调优 Ultralytics YOLO11 模型的超参数，请按照以下步骤操作：

1. **安装所需的包：**

    ```bash
    pip install -U ultralytics "ray[tune]"
    pip install wandb # 可选用于日志记录
    ```

2. **加载 YOLO11 模型并开始调优：**

    ```python
    from ultralytics import YOLO

    # 加载 YOLO11 模型
    model = YOLO("yolo11n.pt")

    # 使用 COCO8 数据集开始调优
    result_grid = model.tune(data="coco8.yaml", use_ray=True)
    ```

这利用了 Ray Tune 的高级搜索策略和并行处理来高效优化模型的超参数。有关更多信息，请查看 [Ray Tune 文档](https://docs.ray.io/en/latest/tune/index.html)。

### 为什么应该使用 Ray Tune 进行 YOLO11 的超参数优化？

Ray Tune 为超参数优化提供了众多优势：

- **高级搜索策略**：利用[贝叶斯优化](https://www.ultralytics.com/glossary/bayesian-network)和 HyperOpt 等算法进行高效的参数搜索。
- **并行处理**：支持多个试验的并行执行，显著加速调优过程。
- **早停**：采用 ASHA 等策略提前终止表现不佳的试验，节省计算资源。

Ray Tune 与 Ultralytics YOLO11 无缝集成，提供易于使用的接口来有效调优超参数。要开始，请查看[超参数调优](../guides/hyperparameter-tuning.md)指南。

### 如何为 YOLO11 超参数调优定义自定义搜索空间？

要为 YOLO11 超参数调优定义自定义搜索空间：

```python
from ray import tune

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
search_space = {"lr0": tune.uniform(1e-5, 1e-1), "momentum": tune.uniform(0.6, 0.98)}
result_grid = model.tune(data="coco8.yaml", space=search_space, use_ray=True)
```

这自定义了调优过程中要探索的超参数（如初始学习率和动量）的范围。有关高级配置，请参阅[自定义搜索空间示例](#自定义搜索空间示例)部分。
