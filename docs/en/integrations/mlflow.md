---
comments: true
description: 学习如何设置和使用 MLflow 日志记录与 Ultralytics YOLO，以增强实验跟踪、模型可重复性和性能改进。
keywords: MLflow, Ultralytics YOLO, 机器学习, 实验跟踪, 指标记录, 参数记录, 工件记录
---

# Ultralytics YOLO 的 MLflow 集成

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/mlflow-integration-ultralytics-yolo.avif" alt="MLflow 生态系统">

## 简介

实验日志记录是[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作流程的关键方面，它能够跟踪各种指标、参数和工件。它有助于增强模型可重复性、调试问题和改进模型性能。[Ultralytics](https://www.ultralytics.com/) YOLO 以其实时[目标检测](https://www.ultralytics.com/glossary/object-detection)能力而闻名，现在提供与 [MLflow](https://mlflow.org/) 的集成，MLflow 是一个用于完整机器学习生命周期管理的开源平台。

本文档页面是设置和使用 Ultralytics YOLO 项目的 MLflow 日志记录功能的综合指南。

## 什么是 MLflow？

[MLflow](https://mlflow.org/) 是由 [Databricks](https://www.databricks.com/) 开发的开源平台，用于管理端到端的机器学习生命周期。它包括用于跟踪实验、将代码打包成可重复运行以及共享和部署模型的工具。MLflow 设计为可与任何机器学习库和编程语言一起使用。

## 功能

- **指标记录**：在每个 epoch 结束时和训练结束时记录指标。
- **参数记录**：记录训练中使用的所有参数。
- **工件记录**：在训练结束时记录模型工件，包括权重和配置文件。

## 设置和先决条件

确保已安装 MLflow。如果没有，使用 pip 安装：

```bash
pip install mlflow
```

确保在 Ultralytics 设置中启用了 MLflow 日志记录。通常，这由设置中的 `mlflow` 键控制。有关更多信息，请参阅[设置](../quickstart.md#ultralytics-settings)页面。

!!! example "更新 Ultralytics MLflow 设置"

    === "Python"

        在 Python 环境中，调用 `settings` 对象的 `update` 方法来更改设置：
        ```python
        from ultralytics import settings

        # 更新设置
        settings.update({"mlflow": True})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        如果你更喜欢使用命令行界面，以下命令将允许你修改设置：
        ```bash
        # 更新设置
        yolo settings mlflow=True

        # 将设置重置为默认值
        yolo settings reset
        ```

## 如何使用

### 命令

1. **设置项目名称**：你可以通过环境变量设置项目名称：

    ```bash
    export MLFLOW_EXPERIMENT_NAME=YOUR_EXPERIMENT_NAME
    ```

    或者在训练 YOLO 模型时使用 `project=<project>` 参数，即 `yolo train project=my_project`。

2. **设置运行名称**：与设置项目名称类似，你可以通过环境变量设置运行名称：

    ```bash
    export MLFLOW_RUN=YOUR_RUN_NAME
    ```

    或者在训练 YOLO 模型时使用 `name=<name>` 参数，即 `yolo train project=my_project name=my_name`。

3. **启动本地 MLflow 服务器**：要开始跟踪，使用：

    ```bash
    mlflow server --backend-store-uri runs/mlflow
    ```

    这将在 `http://127.0.0.1:5000` 启动本地服务器，并将所有 mlflow 日志保存到 'runs/mlflow' 目录。要指定不同的 URI，设置 `MLFLOW_TRACKING_URI` 环境变量。

4. **终止 MLflow 服务器实例**：要停止所有运行的 MLflow 实例，运行：

    ```bash
    ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
    ```

### 日志记录

日志记录由 `on_pretrain_routine_end`、`on_fit_epoch_end` 和 `on_train_end` [回调函数](../reference/utils/callbacks/mlflow.md)处理。这些函数在训练过程的相应阶段自动调用，并处理参数、指标和工件的日志记录。

## 示例

1. **记录自定义指标**：你可以通过在调用 `on_fit_epoch_end` 之前修改 `trainer.metrics` 字典来添加要记录的自定义指标。

2. **查看实验**：要查看日志，导航到你的 MLflow 服务器（通常是 `http://127.0.0.1:5000`）并选择你的实验和运行。<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/yolo-mlflow-experiment.avif" alt="YOLO MLflow 实验">

3. **查看运行**：运行是实验中的单个模型。点击运行查看运行详情，包括上传的工件和模型权重。<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/yolo-mlflow-run.avif" alt="YOLO MLflow 运行">

## 禁用 MLflow

要关闭 MLflow 日志记录：

```bash
yolo settings mlflow=False
```

## 结论

MLflow 日志记录与 Ultralytics YOLO 的集成提供了一种简化的方式来跟踪你的[机器学习实验](https://www.ultralytics.com/blog/log-ultralytics-yolo-experiments-using-mlflow-integration)。它使你能够有效地监控性能指标和管理工件，从而有助于稳健的模型开发和部署。有关更多详细信息，请访问 MLflow [官方文档](https://mlflow.org/docs/latest/index.html)。

## 常见问题

### 如何设置 MLflow 日志记录与 Ultralytics YOLO？

要设置 MLflow 日志记录与 Ultralytics YOLO，你首先需要确保已安装 MLflow。你可以使用 pip 安装：

```bash
pip install mlflow
```

接下来，在 Ultralytics 设置中启用 MLflow 日志记录。这可以使用 `mlflow` 键控制。有关更多信息，请参阅[设置指南](../quickstart.md#ultralytics-settings)。

!!! example "更新 Ultralytics MLflow 设置"

    === "Python"

        ```python
        from ultralytics import settings

        # 更新设置
        settings.update({"mlflow": True})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        ```bash
        # 更新设置
        yolo settings mlflow=True

        # 将设置重置为默认值
        yolo settings reset
        ```

最后，启动本地 MLflow 服务器进行跟踪：

```bash
mlflow server --backend-store-uri runs/mlflow
```

### 使用 MLflow 与 Ultralytics YOLO 可以记录哪些指标和参数？

Ultralytics YOLO 与 MLflow 支持在整个训练过程中记录各种指标、参数和工件：

- **指标记录**：在每个 [epoch](https://www.ultralytics.com/glossary/epoch) 结束时和训练完成时跟踪指标。
- **参数记录**：记录训练过程中使用的所有参数。
- **工件记录**：在训练后保存模型工件，如权重和配置文件。

有关更详细的信息，请访问 [Ultralytics YOLO 跟踪文档](#功能)。

### 启用后可以禁用 MLflow 日志记录吗？

是的，你可以通过更新设置来禁用 Ultralytics YOLO 的 MLflow 日志记录。以下是使用 CLI 的方法：

```bash
yolo settings mlflow=False
```

有关进一步自定义和重置设置，请参阅[设置指南](../quickstart.md#ultralytics-settings)。

### 如何启动和停止用于 Ultralytics YOLO 跟踪的 MLflow 服务器？

要启动 MLflow 服务器以跟踪 Ultralytics YOLO 中的实验，使用以下命令：

```bash
mlflow server --backend-store-uri runs/mlflow
```

此命令默认在 `http://127.0.0.1:5000` 启动本地服务器。如果你需要停止运行的 MLflow 服务器实例，使用以下 bash 命令：

```bash
ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
```

有关更多命令选项，请参阅[命令部分](#命令)。

### 将 MLflow 与 Ultralytics YOLO 集成进行实验跟踪有什么好处？

将 MLflow 与 Ultralytics YOLO 集成为管理机器学习实验提供了几个好处：

- **增强实验跟踪**：轻松跟踪和比较不同运行及其结果。
- **改进模型可重复性**：通过记录所有参数和工件确保实验可重复。
- **性能监控**：随时间可视化性能指标，为模型改进做出数据驱动的决策。
- **简化工作流程**：自动化日志记录过程，让你更专注于模型开发而不是手动跟踪。
- **协作开发**：与团队成员分享实验结果，以便更好地协作和知识共享。

有关设置和利用 MLflow 与 Ultralytics YOLO 的深入了解，请探索 [Ultralytics YOLO 的 MLflow 集成](#简介)文档。
