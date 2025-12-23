---
comments: true
description: 了解如何将 YOLO11 与 ClearML 集成，简化您的 MLOps 工作流程，自动化实验并轻松增强模型管理。
keywords: YOLO11, ClearML, MLOps, Ultralytics, 机器学习, 目标检测, 模型训练, 自动化, 实验管理
---

# 使用 ClearML 训练 YOLO11：简化您的 MLOps 工作流程

MLOps 弥合了在实际环境中创建和部署[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型之间的差距。它专注于高效部署、可扩展性和持续管理，以确保模型在实际应用中表现良好。

[Ultralytics YOLO11](https://www.ultralytics.com/) 与 ClearML 轻松集成，简化和增强您的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的训练和管理过程。本指南将引导您完成集成过程，详细说明如何设置 ClearML、管理实验、自动化模型管理和有效协作。

## ClearML

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/clearml-overview.avif" alt="ClearML 概述">
</p>

[ClearML](https://clear.ml/) 是一个创新的开源 MLOps 平台，专门设计用于自动化、监控和编排机器学习工作流程。其主要功能包括自动记录所有训练和推理数据以实现完全的实验可重复性、直观的 Web UI 便于[数据可视化](https://www.ultralytics.com/glossary/data-visualization)和分析、高级超参数[优化算法](https://www.ultralytics.com/glossary/optimization-algorithm)，以及强大的模型管理以便在各种平台上高效部署。

## 使用 ClearML 训练 YOLO11

通过将 YOLO11 与 ClearML 集成，您可以为机器学习工作流程带来自动化和效率，以改进训练过程。

## 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "命令行"

        ```bash
        # 安装 YOLO11 和 ClearML 所需的包
        pip install ultralytics clearml
        ```

## 配置 ClearML

安装必要的包后，下一步是初始化和配置您的 ClearML SDK。这涉及设置您的 ClearML 账户并获取必要的凭据，以便在开发环境和 ClearML 服务器之间建立无缝连接。

!!! tip "初始 SDK 设置"

    === "命令行"

        ```bash
        # 初始化您的 ClearML SDK 设置过程
        clearml-init
        ```

执行此命令后，访问 [ClearML 设置页面](https://app.clear.ml/settings/workspace-configuration)。导航到右上角并选择"设置"。转到"工作区"部分并点击"创建新凭据"。使用"创建凭据"弹出窗口中提供的凭据完成设置。

## 使用

!!! example "使用"

    === "Python"

        ```python
        from clearml import Task

        from ultralytics import YOLO

        # 步骤 1：创建 ClearML 任务
        task = Task.init(project_name="my_project", task_name="my_yolo11_task")

        # 步骤 2：选择 YOLO11 模型
        model_variant = "yolo11n"
        task.set_parameter("model_variant", model_variant)

        # 步骤 3：加载 YOLO11 模型
        model = YOLO(f"{model_variant}.pt")

        # 步骤 4：设置训练参数
        args = dict(data="coco8.yaml", epochs=16)
        task.connect(args)

        # 步骤 5：启动模型训练
        results = model.train(**args)
        ```

### 理解代码

**步骤 1：创建 ClearML 任务**：在 ClearML 中初始化一个新任务，指定您的项目和任务名称。此任务将跟踪和管理您的模型训练。

**步骤 2：选择 YOLO11 模型**：`model_variant` 变量设置为 'yolo11n'，这是 YOLO11 模型之一。然后将此变体记录在 ClearML 中以进行跟踪。

**步骤 3：加载 YOLO11 模型**：使用 Ultralytics 的 YOLO 类加载所选的 YOLO11 模型，为训练做准备。

**步骤 4：设置训练参数**：关键训练参数如数据集（`coco8.yaml`）和[训练周期](https://www.ultralytics.com/glossary/epoch)数（`16`）组织在字典中并连接到 ClearML 任务。

**步骤 5：启动模型训练**：使用指定的参数开始模型训练。训练过程的结果捕获在 `results` 变量中。

### ClearML 结果页面的主要功能

- **实时指标跟踪**：跟踪损失、[准确率](https://www.ultralytics.com/glossary/accuracy)和验证分数等关键指标。
- **实验比较**：并排比较不同的训练运行。
- **详细日志和输出**：访问全面的日志、指标的图形表示和控制台输出。
- **资源利用率监控**：监控 CPU、GPU 和内存等计算资源的利用率。
- **模型工件管理**：查看、下载和共享训练模型和检查点等模型工件。

### ClearML 中的高级功能

#### 远程执行

ClearML 的远程执行功能便于在不同机器上重现和操作实验。它记录已安装的包和未提交的更改等重要细节。当任务入队时，[ClearML Agent](https://clear.ml/docs/latest/docs/clearml_agent/) 会拉取它、重新创建环境并运行实验，报告详细结果。

#### 克隆、编辑和入队

ClearML 的用户友好界面允许轻松克隆、编辑和入队任务。用户可以克隆现有实验，通过 UI 调整参数或其他细节，并将任务入队执行。

## 数据集版本管理

ClearML 还提供强大的[数据集版本管理](https://clear.ml/docs/latest/docs/hyperdatasets/dataset/)功能，与 YOLO11 训练工作流程无缝集成。

## 总结

本指南引导您完成了将 ClearML 与 Ultralytics 的 YOLO11 集成的过程。涵盖从初始设置到高级模型管理的所有内容，您已经了解了如何利用 ClearML 进行高效训练、实验跟踪和机器学习项目中的工作流程优化。

有关使用的更多详细信息，请访问 [ClearML 的官方 YOLOv8 集成指南](https://clear.ml/docs/latest/docs/integrations/yolov8/)，该指南也适用于 YOLO11 工作流程。

## 常见问题

### 将 Ultralytics YOLO11 与 ClearML 集成的过程是什么？

将 Ultralytics YOLO11 与 ClearML 集成涉及一系列步骤来简化您的 MLOps 工作流程。首先，安装必要的包，然后使用 `clearml-init` 初始化 ClearML SDK，接着在 ClearML 设置页面配置您的凭据。

### 为什么应该将 ClearML 与 Ultralytics YOLO11 一起用于我的机器学习项目？

将 ClearML 与 Ultralytics YOLO11 一起使用可以通过自动化实验跟踪、简化工作流程和实现强大的模型管理来增强您的机器学习项目。ClearML 提供实时指标跟踪、资源利用率监控和用于比较实验的用户友好界面。
