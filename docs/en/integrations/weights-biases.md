---
comments: true
description: 学习如何使用 Weights & Biases 增强 YOLO11 实验跟踪和可视化，以获得更好的模型性能和管理。
keywords: YOLO11, Weights & Biases, 模型训练, 实验跟踪, Ultralytics, 机器学习, 计算机视觉, 模型可视化
---

# 使用 Weights & Biases 进行 YOLO 实验跟踪和可视化

[目标检测](https://www.ultralytics.com/glossary/object-detection)模型如 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 已成为许多[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用的核心组成部分。然而，训练、评估和部署这些复杂模型会带来诸多挑战。跟踪关键训练指标、比较模型变体、分析模型行为和检测问题需要大量的工具和实验管理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EeDd5P4eS6A"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO11 与 Weights and Biases 配合使用
</p>

本指南展示了 Ultralytics YOLO11 与 Weights & Biases 的集成，用于增强实验跟踪、模型检查点保存和模型性能可视化。它还包括设置集成、训练、微调以及使用 Weights & Biases 交互式功能可视化结果的说明。

## Weights & Biases

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/wandb-demo-experiments.avif" alt="Weights & Biases 概览">
</p>

[Weights & Biases](https://wandb.ai/site) 是一个前沿的 [MLOps 平台](https://www.ultralytics.com/glossary/machine-learning-operations-mlops)，专为跟踪、可视化和管理[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)实验而设计。它具有自动记录训练指标以实现完整实验可重现性、用于简化数据分析的交互式 UI，以及用于在各种环境中部署的高效模型管理工具。

## 使用 Weights & Biases 进行 YOLO11 训练

您可以使用 Weights & Biases 为 YOLO11 训练过程带来效率和自动化。该集成允许您跟踪实验、比较模型，并做出数据驱动的决策来改进您的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目。

## 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 Ultralytics YOLO 和 Weights & Biases 所需的包
        pip install -U ultralytics wandb

        # 为 Ultralytics 启用 W&B 日志记录
        yolo settings wandb=True
        ```

有关安装过程的详细说明和最佳实践，请务必查看我们的 [YOLO11 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

## 配置 Weights & Biases

安装必要的包后，下一步是设置您的 Weights & Biases 环境。这包括创建 Weights & Biases 账户并获取必要的 API 密钥，以实现开发环境与 W&B 平台之间的顺畅连接。

首先在您的工作区中初始化 Weights & Biases 环境。您可以通过运行以下命令并按照提示的说明进行操作。

!!! tip "初始 SDK 设置"

    === "Python"

        ```python
        import wandb

        # 初始化您的 Weights & Biases 环境
        wandb.login(key="YOUR_API_KEY")
        ```

    === "CLI"

        ```bash
        # 初始化您的 Weights & Biases 环境
        wandb login
        ```

导航到 [Weights & Biases 授权页面](https://wandb.ai/authorize)创建并获取您的 API 密钥。在提示时使用此密钥来验证您的环境与 W&B 的连接。

## 使用方法：使用 Weights & Biases 训练 YOLO11

在深入了解使用 Weights & Biases 进行 YOLO11 模型训练的使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型系列](../models/index.md)。这将帮助您选择最适合项目需求的模型。

!!! example "使用方法：使用 Weights & Biases 训练 YOLO11"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 训练和微调模型
        model.train(data="coco8.yaml", epochs=5, project="ultralytics", name="yolo11n")
        ```

    === "CLI"

        ```bash
        # 使用 Weights & Biases 训练 YOLO11 模型
        yolo train data=coco8.yaml epochs=5 project=ultralytics name=yolo11n
        ```

### W&B 参数

| 参数    | 默认值 | 描述                                                                                                        |
| ------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| project | `None` | 指定本地和 W&B 中记录的项目名称。这样您可以将多个运行分组在一起。        |
| name    | `None` | 训练运行的名称。这决定了用于创建子文件夹的名称和 W&B 日志记录使用的名称 |

!!! tip "启用或禁用 Weights & Biases"

    如果您想在 Ultralytics 中启用或禁用 Weights & Biases 日志记录，可以使用 `yolo settings` 命令。默认情况下，Weights & Biases 日志记录是禁用的。

    === "CLI"

        ```bash
        # 启用 Weights & Biases 日志记录
        yolo settings wandb=True

        # 禁用 Weights & Biases 日志记录
        yolo settings wandb=False
        ```

### 理解输出

运行上述使用代码片段后，您可以期望以下关键输出：

- 设置具有唯一 ID 的新运行，表示训练过程的开始。
- 模型结构的简要摘要，包括层数和参数数量。
- 在每个训练[轮次](https://www.ultralytics.com/glossary/epoch)期间定期更新重要指标，如 box loss、cls loss、dfl loss、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [mAP 分数](https://www.ultralytics.com/glossary/mean-average-precision-map)。
- 训练结束时，显示详细指标，包括模型的推理速度和整体[精度](https://www.ultralytics.com/glossary/accuracy)指标。
- 指向 Weights & Biases 仪表板的链接，用于深入分析和可视化训练过程，以及本地日志文件位置的信息。

### 查看 Weights & Biases 仪表板

运行使用代码片段后，您可以通过输出中提供的链接访问 Weights & Biases（W&B）仪表板。此仪表板提供了使用 YOLO11 进行模型训练过程的全面视图。

## Weights & Biases 仪表板的主要功能

- **实时指标跟踪**：观察损失、精度和验证分数等指标在训练过程中的演变，为模型调优提供即时洞察。[查看如何使用 Weights & Biases 跟踪实验](https://imgur.com/D6NVnmN)。

- **超参数优化**：Weights & Biases 有助于微调关键参数，如[学习率](https://www.ultralytics.com/glossary/learning-rate)、[批量大小](https://www.ultralytics.com/glossary/batch-size)等，从而提高 YOLO11 的性能。这有助于您为特定数据集和任务找到最佳配置。

- **比较分析**：该平台允许对不同训练运行进行并排比较，这对于评估各种模型配置的影响和了解哪些更改可以提高性能至关重要。

- **训练进度可视化**：关键指标的图形表示提供了对模型在各轮次中性能的直观理解。[查看 Weights & Biases 如何帮助您可视化验证结果](https://imgur.com/a/kU5h7W4)。

- **资源监控**：跟踪 CPU、GPU 和内存使用情况，以优化训练过程的效率并识别工作流程中的潜在瓶颈。

- **模型工件管理**：访问和共享模型检查点，便于在复杂项目中轻松部署和与团队成员协作。

- **使用图像叠加查看推理结果**：使用 Weights & Biases 中的交互式叠加在图像上可视化预测结果，提供模型在真实世界数据上性能的清晰详细视图。有关更多详细信息，请参阅 Weights & Biases 的[图像叠加功能](https://docs.wandb.ai/models/track/log/media#image-overlays)。

通过使用这些功能，您可以有效地跟踪、分析和优化 YOLO11 模型的训练，确保[目标检测](https://www.ultralytics.com/glossary/object-detection)任务获得最佳性能和效率。

## 总结

本指南帮助您探索了 Ultralytics YOLO 与 Weights & Biases 的集成。它展示了此集成高效跟踪和可视化模型训练及预测结果的能力。通过利用 W&B 的强大功能，您可以简化[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作流程，做出数据驱动的决策，并提高模型性能。

有关使用详情的更多信息，请访问 [Weights & Biases 官方文档](https://docs.wandb.ai/models/integrations/ultralytics)或探索 [Soumik Rakshit 在 YOLO VISION 2023 上关于此集成的演讲](https://www.ultralytics.com/blog/supercharging-ultralytics-with-weights-biases)。

此外，请务必查看 [Ultralytics 集成指南页面](../integrations/index.md)，了解更多关于不同精彩集成的信息，如 [MLflow](../integrations/mlflow.md) 和 [Comet ML](../integrations/comet.md)。

## 常见问题

### 如何将 Weights & Biases 与 Ultralytics YOLO11 集成？

要将 Weights & Biases 与 Ultralytics YOLO11 集成：

1. 安装所需的包：

    ```bash
    pip install -U ultralytics wandb
    yolo settings wandb=True
    ```

2. 登录您的 Weights & Biases 账户：

    ```python
    import wandb

    wandb.login(key="YOUR_API_KEY")
    ```

3. 启用 W&B 日志记录训练您的 YOLO11 模型：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(data="coco8.yaml", epochs=5, project="ultralytics", name="yolo11n")
    ```

这将自动将指标、超参数和模型工件记录到您的 W&B 项目中。

### Weights & Biases 与 YOLO11 集成的主要功能是什么？

主要功能包括：

- 训练期间的实时指标跟踪
- 超参数优化工具
- 不同训练运行的比较分析
- 通过图表可视化训练进度
- 资源监控（CPU、GPU、内存使用情况）
- 模型工件管理和共享
- 使用图像叠加查看推理结果

这些功能有助于跟踪实验、优化模型，并在 YOLO11 项目上更有效地协作。

### 如何查看 YOLO11 训练的 Weights & Biases 仪表板？

运行启用 W&B 集成的训练脚本后：

1. 控制台输出中将提供指向您的 W&B 仪表板的链接。
2. 点击链接或访问 [wandb.ai](https://wandb.ai/) 并登录您的账户。
3. 导航到您的项目以查看详细的指标、可视化和模型性能数据。

仪表板提供了对模型训练过程的洞察，使您能够有效地分析和改进 YOLO11 模型。

### 我可以禁用 YOLO11 训练的 Weights & Biases 日志记录吗？

是的，您可以使用以下命令禁用 W&B 日志记录：

```bash
yolo settings wandb=False
```

要重新启用日志记录，请使用：

```bash
yolo settings wandb=True
```

这允许您在不修改训练脚本的情况下控制何时使用 W&B 日志记录。

### Weights & Biases 如何帮助优化 YOLO11 模型？

Weights & Biases 通过以下方式帮助优化 YOLO11 模型：

1. 提供训练指标的详细可视化
2. 轻松比较不同模型版本
3. 提供[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)工具
4. 允许协作分析模型性能
5. 便于轻松共享模型工件和结果

这些功能帮助研究人员和开发人员更快地迭代，并做出数据驱动的决策来改进他们的 YOLO11 模型。
