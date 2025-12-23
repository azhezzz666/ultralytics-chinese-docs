---
comments: true
description: 了解启动成功计算机视觉项目的基本步骤，从定义目标到模型部署和维护。
keywords: 计算机视觉, AI, 目标检测, 图像分类, 实例分割, 数据标注, 模型训练, 模型评估, 模型部署
---

# 理解计算机视觉项目的关键步骤

## 简介

计算机视觉是[人工智能](https://www.ultralytics.com/glossary/artificial-intelligence-ai)（AI）的一个子领域，帮助计算机像人类一样看到和理解世界。它处理和分析图像或视频以提取信息、识别模式并根据该数据做出决策。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/CfbHwPG01cE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何进行<a href="https://www.ultralytics.com/glossary/computer-vision-cv">计算机视觉</a>项目 | 分步指南
</p>

[目标检测](../tasks/detect.md)、[图像分类](../tasks/classify.md)和[实例分割](../tasks/segment.md)等计算机视觉技术可以应用于从[自动驾驶](https://www.ultralytics.com/solutions/ai-in-automotive)到[医学成像](https://www.ultralytics.com/solutions/ai-in-healthcare)的各个行业，以获得有价值的洞察。

进行自己的计算机视觉项目是理解和学习更多计算机视觉知识的好方法。然而，计算机视觉项目可能包含许多步骤，一开始可能看起来令人困惑。在本指南结束时，您将熟悉计算机视觉项目涉及的步骤。我们将从头到尾介绍项目的所有内容，解释为什么每个部分都很重要。

## 计算机视觉项目概述

在讨论计算机视觉项目涉及的每个步骤的细节之前，让我们先看看整体流程。如果您今天开始一个计算机视觉项目，您将采取以下步骤：

- 您的首要任务是了解项目的需求。
- 然后，您将收集并准确标注有助于训练模型的图像。
- 接下来，您将清理数据并应用增强技术为模型训练做准备。
- 模型训练后，您将彻底测试和评估模型，以确保它在不同条件下表现一致。
- 最后，您将把模型部署到现实世界中，并根据新的洞察和反馈进行更新。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/five-stages-of-ml-development-lifecycle.avif" alt="计算机视觉项目步骤概述">
</p>

现在我们知道了要期待什么，让我们深入了解这些步骤并推进您的项目。

## 步骤 1：定义项目目标

任何计算机视觉项目的第一步是清楚地定义您要解决的问题。了解最终目标有助于您开始构建解决方案。这在计算机视觉中尤其重要，因为您的项目目标将直接影响您需要关注的计算机视觉任务。

以下是一些项目目标和可用于实现这些目标的计算机视觉任务的示例：

- **目标：** 开发一个系统，可以监控和管理高速公路上不同类型车辆的流量，改善交通管理和安全。
    - **计算机视觉任务：** 目标检测非常适合交通监控，因为它可以高效地定位和识别多辆车辆。它比图像分割计算要求更低，后者为此任务提供了不必要的细节，确保更快的实时分析。

- **目标：** 开发一个工具，通过在医学成像扫描中提供肿瘤的精确像素级轮廓来协助放射科医生。
    - **计算机视觉任务：** 图像分割适用于医学成像，因为它提供了对评估大小、形状和治疗计划至关重要的肿瘤的准确和详细边界。

- **目标：** 创建一个数字系统，对各种文档（如发票、收据、法律文件）进行分类，以提高组织效率和文档检索。
    - **计算机视觉任务：** [图像分类](https://www.ultralytics.com/glossary/image-classification)在这里是理想的，因为它一次处理一个文档，无需考虑文档在图像中的位置。这种方法简化并加速了分类过程。

### 步骤 1.5：选择正确的模型和训练方法

在了解项目目标和合适的计算机视觉任务后，定义项目目标的一个重要部分是[选择正确的模型](../models/index.md)和训练方法。

根据目标，您可能会选择先选择模型，或者在步骤 2 中查看您能够收集的数据后再选择。例如，假设您的项目高度依赖于特定类型数据的可用性。在这种情况下，先收集和分析数据然后再选择模型可能更实际。另一方面，如果您对模型要求有清晰的了解，您可以先选择模型，然后收集符合这些规格的数据。

选择从头开始训练还是使用[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)会影响您准备数据的方式。从头开始训练需要多样化的数据集来从头构建模型的理解。另一方面，迁移学习允许您使用预训练模型并用更小、更具体的数据集对其进行调整。此外，选择要训练的特定模型将决定您需要如何准备数据，例如根据模型的特定要求调整图像大小或添加标注。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/training-from-scratch-vs-transfer-learning.avif" alt="从头训练与使用迁移学习">
</p>

注意：选择模型时，请考虑其[部署](./model-deployment-options.md)以确保兼容性和性能。例如，轻量级模型由于在资源受限设备上的效率而非常适合[边缘计算](https://www.ultralytics.com/glossary/edge-computing)。要了解更多关于定义项目的关键点，请阅读我们关于[定义项目目标和选择正确模型](./defining-project-goals.md)的指南。

在进入计算机视觉项目的实际工作之前，清楚地了解这些细节很重要。在进入步骤 2 之前，请仔细检查您是否已考虑以下内容：

- 清楚地定义您要解决的问题。
- 确定项目的最终目标。
- 确定所需的特定计算机视觉任务（例如目标检测、图像分类、图像分割）。
- 决定是从头开始训练模型还是使用迁移学习。
- 为您的任务和部署需求选择合适的模型。

## 步骤 2：数据收集和数据标注

计算机视觉模型的质量取决于数据集的质量。您可以从互联网收集图像、拍摄自己的照片或使用现有数据集。以下是一些下载高质量数据集的优秀资源：[Google 数据集搜索引擎](https://datasetsearch.research.google.com/)、[UC Irvine 机器学习仓库](https://archive.ics.uci.edu/)和 [Kaggle 数据集](https://www.kaggle.com/datasets)。

一些库（如 Ultralytics）提供[对各种数据集的内置支持](../datasets/index.md)，使您更容易从高质量数据开始。这些库通常包括无缝使用流行数据集的实用程序，可以在项目初期为您节省大量时间和精力。

但是，如果您选择收集图像或拍摄自己的照片，则需要标注数据。数据标注是为数据添加标签以向模型传授知识的过程。您将使用的数据标注类型取决于您的特定计算机视觉技术。以下是一些示例：

- **图像分类：** 您将整个图像标记为单个类别。
- **[目标检测](https://www.ultralytics.com/glossary/object-detection)：** 您将在图像中的每个对象周围绘制边界框并标记每个框。
- **[图像分割](https://www.ultralytics.com/glossary/image-segmentation)：** 您将根据图像中每个像素所属的对象对其进行标记，创建详细的对象边界。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/different-types-of-image-annotation.avif" alt="不同类型的图像标注">
</p>

[数据收集和标注](./data-collection-and-annotation.md)可能是一项耗时的手动工作。标注工具可以帮助简化此过程。以下是一些有用的开源标注工具：[Label Studio](https://github.com/HumanSignal/label-studio)、[CVAT](https://github.com/cvat-ai/cvat) 和 [Labelme](https://github.com/wkentaro/labelme)。

## 步骤 3：数据增强和拆分数据集

收集和标注图像数据后，重要的是在执行[数据增强](https://www.ultralytics.com/glossary/data-augmentation)之前先将数据集拆分为训练集、验证集和测试集。在增强之前拆分数据集对于在原始、未更改的数据上测试和验证模型至关重要。它有助于准确评估模型对新的、未见过的数据的泛化能力。

以下是如何拆分数据：

- **训练集：** 这是数据的最大部分，通常占总数据的 70-80%，用于训练模型。
- **验证集：** 通常约占数据的 10-15%；此集用于在训练期间调整超参数和验证模型，有助于防止[过拟合](https://www.ultralytics.com/glossary/overfitting)。
- **测试集：** 剩余的 10-15% 数据作为测试集保留。它用于在训练完成后评估模型在未见过数据上的性能。

拆分数据后，您可以通过应用旋转、缩放和翻转图像等变换来执行数据增强，以人为增加数据集的大小。数据增强使您的模型对变化更加稳健，并提高其在未见过图像上的性能。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/examples-of-data-augmentations.avif" alt="数据增强示例">
</p>

[OpenCV](https://www.ultralytics.com/glossary/opencv)、[Albumentations](../integrations/albumentations.md) 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 等库提供了灵活的增强功能供您使用。此外，一些库（如 Ultralytics）在其模型训练功能中直接具有[内置增强设置](../modes/train.md)，简化了流程。

为了更好地理解数据，您可以使用 [Matplotlib](https://matplotlib.org/) 或 [Seaborn](https://seaborn.pydata.org/) 等工具来可视化图像并分析其分布和特征。可视化数据有助于识别模式、异常和增强技术的有效性。您还可以使用 [Ultralytics Explorer](../datasets/explorer/index.md)，这是一个用于通过语义搜索、SQL 查询和向量相似性搜索探索计算机视觉数据集的工具。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-1.avif" alt="Ultralytics Explorer 工具">
</p>

通过正确[理解、拆分和增强数据](./preprocessing_annotated_data.md)，您可以开发一个经过良好训练、验证和测试的模型，在实际应用中表现良好。

## 步骤 4：模型训练

一旦数据集准备好进行训练，您就可以专注于设置必要的环境、管理数据集和训练模型。

首先，您需要确保环境配置正确。通常，这包括以下内容：

- 安装 TensorFlow、[PyTorch](https://www.ultralytics.com/glossary/pytorch) 或 [Ultralytics](../quickstart.md) 等基本库和框架。
- 如果您使用 GPU，安装 CUDA 和 cuDNN 等库将有助于启用 GPU 加速并加快训练过程。

然后，您可以将训练和验证数据集加载到环境中。通过调整大小、格式转换或增强来归一化和预处理数据。选择模型后，配置层并指定超参数。通过设置[损失函数](https://www.ultralytics.com/glossary/loss-function)、优化器和性能指标来编译模型。

Ultralytics 等库简化了训练过程。您可以用最少的代码[开始训练](../modes/train.md)，将数据输入模型。这些库自动处理权重调整、[反向传播](https://www.ultralytics.com/glossary/backpropagation)和验证。它们还提供工具来轻松监控进度和调整超参数。训练后，只需几个命令即可保存模型及其权重。

重要的是要记住，正确的数据集管理对于高效训练至关重要。使用数据集的版本控制来跟踪更改并确保可重复性。[DVC（数据版本控制）](../integrations/dvc.md)等工具可以帮助管理大型数据集。

## 步骤 5：模型评估和模型微调

使用各种指标评估模型性能并对其进行优化以提高[准确性](https://www.ultralytics.com/glossary/accuracy)非常重要。[评估](../modes/val.md)有助于识别模型表现出色的领域和可能需要改进的领域。[微调](https://www.ultralytics.com/glossary/fine-tuning)确保模型针对最佳性能进行优化。

- **[性能指标](./yolo-performance-metrics.md)：** 使用准确率、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 F1 分数等指标来评估模型性能。这些指标提供了关于模型预测效果的洞察。
- **[超参数调优](./hyperparameter-tuning.md)：** 调整超参数以优化模型性能。网格搜索或随机搜索等技术可以帮助找到最佳超参数值。
- **微调：** 对模型架构或训练过程进行小调整以提高性能。这可能涉及调整[学习率](https://www.ultralytics.com/glossary/learning-rate)、[批量大小](https://www.ultralytics.com/glossary/batch-size)或其他模型参数。

要深入了解模型评估和微调技术，请查看我们的[模型评估洞察指南](./model-evaluation-insights.md)。

## 步骤 6：模型测试

在此步骤中，您可以确保模型在完全未见过的数据上表现良好，确认其已准备好部署。模型测试和模型评估的区别在于，它侧重于验证最终模型的性能，而不是迭代改进它。

彻底测试和调试可能出现的任何常见问题非常重要。在训练或验证期间未使用的单独测试数据集上测试模型。此数据集应代表真实场景，以确保模型的性能一致且可靠。

此外，解决过拟合、[欠拟合](https://www.ultralytics.com/glossary/underfitting)和数据泄漏等常见问题。使用[交叉验证](https://www.ultralytics.com/glossary/cross-validation)和[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)等技术来识别和修复这些问题。有关全面的测试策略，请参阅我们的[模型测试指南](./model-testing.md)。

## 步骤 7：模型部署

一旦模型经过彻底测试，就可以部署它了。[模型部署](https://www.ultralytics.com/glossary/model-deployment)涉及使模型在生产环境中可用。以下是部署计算机视觉模型的步骤：

- **设置环境：** 为您选择的部署选项配置必要的基础设施，无论是基于云的（AWS、Google Cloud、Azure）还是基于边缘的（本地设备、IoT）。
- **[导出模型](../modes/export.md)：** 将模型导出为适当的格式（例如 YOLO11 的 ONNX、TensorRT、CoreML），以确保与部署平台的兼容性。
- **部署模型：** 通过设置 API 或端点并将其与应用程序集成来部署模型。
- **确保可扩展性：** 实施负载均衡器、自动扩展组和监控工具来管理资源并处理不断增加的数据和用户请求。

有关部署策略和最佳实践的更详细指导，请查看我们的[模型部署实践指南](./model-deployment-practices.md)。

## 步骤 8：监控、维护和文档

一旦模型部署完成，持续监控其性能、维护它以处理任何问题，并记录整个过程以供将来参考和改进非常重要。

监控工具可以帮助您跟踪关键性能指标（KPI）并检测准确性的异常或下降。通过监控模型，您可以了解模型漂移，即由于输入数据的变化，模型性能随时间下降。定期使用更新的数据重新训练模型以保持准确性和相关性。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/model-monitoring-maintenance-loop.avif" alt="模型监控">
</p>

除了监控和维护外，文档也是关键。彻底记录整个过程，包括模型架构、训练程序、超参数、数据预处理步骤以及部署和维护期间所做的任何更改。良好的文档确保可重复性，并使将来的更新或故障排除更容易。通过有效地[监控、维护和记录模型](./model-monitoring-and-maintenance.md)，您可以确保它在整个生命周期中保持准确、可靠且易于管理。

## 与社区互动

与计算机视觉爱好者社区建立联系可以帮助您自信地解决在计算机视觉项目中遇到的任何问题。以下是一些有效学习、故障排除和建立联系的方法。

### 社区资源

- **GitHub Issues：** 查看 [YOLO11 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)并使用 Issues 标签提问、报告错误和建议新功能。活跃的社区和维护者会帮助解决特定问题。
- **Ultralytics Discord 服务器：** 加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者互动，获得支持并分享见解。

### 官方文档

- **Ultralytics YOLO11 文档：** 探索[官方 YOLO11 文档](./index.md)获取关于不同计算机视觉任务和项目的详细指南和有用技巧。

使用这些资源将帮助您克服挑战并了解计算机视觉社区的最新趋势和最佳实践。

## 下一步

进行计算机视觉项目可能既令人兴奋又有回报。通过遵循本指南中的步骤，您可以为成功奠定坚实的基础。每个步骤对于开发满足您目标并在实际场景中运行良好的解决方案都至关重要。随着经验的积累，您将发现改进项目的高级技术和工具。

## 常见问题

### 如何为我的项目选择正确的计算机视觉任务？

选择正确的计算机视觉任务取决于项目的最终目标。例如，如果您想监控交通，**目标检测**是合适的，因为它可以实时定位和识别多种车辆类型。对于医学成像，**图像分割**是理想的，因为它提供肿瘤的详细边界，有助于诊断和治疗计划。了解更多关于特定任务的信息，如[目标检测](../tasks/detect.md)、[图像分类](../tasks/classify.md)和[实例分割](../tasks/segment.md)。

### 为什么数据标注在计算机视觉项目中至关重要？

数据标注对于教会模型识别模式至关重要。标注类型因任务而异：

- **图像分类**：整个图像标记为单个类别。
- **目标检测**：在对象周围绘制边界框。
- **图像分割**：根据每个像素所属的对象进行标记。

[Label Studio](https://github.com/HumanSignal/label-studio)、[CVAT](https://github.com/cvat-ai/cvat) 和 [Labelme](https://github.com/wkentaro/labelme) 等工具可以协助此过程。有关更多详细信息，请参阅我们的[数据收集和标注指南](./data-collection-and-annotation.md)。

### 我应该遵循哪些步骤来有效地增强和拆分数据集？

在增强之前拆分数据集有助于在原始、未更改的数据上验证模型性能。请遵循以下步骤：

- **训练集**：70-80% 的数据。
- **验证集**：10-15% 用于[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)。
- **测试集**：剩余的 10-15% 用于最终评估。

拆分后，应用旋转、缩放和翻转等数据增强技术来增加数据集多样性。[Albumentations](../integrations/albumentations.md) 和 OpenCV 等库可以提供帮助。Ultralytics 还提供[内置增强设置](../modes/train.md)以方便使用。

### 如何导出训练好的计算机视觉模型进行部署？

导出模型可确保与不同部署平台的兼容性。Ultralytics 提供多种格式，包括 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md) 和 [CoreML](../integrations/coreml.md)。要导出 YOLO11 模型，请遵循以下指南：

- 使用带有所需格式参数的 `export` 函数。
- 确保导出的模型符合部署环境的规格（例如边缘设备、云）。

有关更多信息，请查看[模型导出指南](../modes/export.md)。

### 监控和维护已部署的计算机视觉模型的最佳实践是什么？

持续监控和维护对于模型的长期成功至关重要。实施工具来跟踪关键性能指标（KPI）并检测异常。定期使用更新的数据重新训练模型以对抗模型漂移。记录整个过程，包括模型架构、超参数和更改，以确保可重复性和便于将来更新。在我们的[监控和维护指南](./model-monitoring-and-maintenance.md)中了解更多信息。
