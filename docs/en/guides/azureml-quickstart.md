---
comments: true
description: 学习如何在 AzureML 上运行 YOLO11。包含终端和笔记本的快速入门说明，利用 Azure 的云计算进行高效模型训练。
keywords: YOLO11, AzureML, 机器学习, 云计算, 快速入门, 终端, 笔记本, 模型训练, Python SDK, AI, Ultralytics
---

# YOLO11 🚀 在 AzureML 上运行

## 什么是 Azure？

[Azure](https://azure.microsoft.com/) 是微软的[云计算](https://www.ultralytics.com/glossary/cloud-computing)平台，旨在帮助组织将工作负载从本地数据中心迁移到云端。Azure 提供全方位的云服务，包括计算、数据库、分析、[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和网络，用户可以从这些服务中选择来开发和扩展新应用程序，或在公有云中运行现有应用程序。

## 什么是 Azure 机器学习（AzureML）？

Azure 机器学习，通常称为 AzureML，是一个完全托管的云服务，使数据科学家和开发人员能够高效地将预测分析嵌入到他们的应用程序中，帮助组织利用海量数据集并将云的所有优势带入机器学习。AzureML 提供各种服务和功能，旨在使机器学习变得易于访问、易于使用和可扩展。它提供自动化机器学习、拖放式模型训练以及强大的 Python SDK 等功能，使开发人员能够充分利用他们的机器学习模型。

## AzureML 如何使 YOLO 用户受益？

对于 YOLO（You Only Look Once）用户，AzureML 提供了一个强大、可扩展且高效的平台来训练和部署机器学习模型。无论您是想运行快速原型还是扩展以处理更大规模的数据，AzureML 灵活且用户友好的环境都提供各种工具和服务来满足您的需求。您可以利用 AzureML：

- 轻松管理用于训练的大型数据集和计算资源。
- 利用内置工具进行数据预处理、特征选择和模型训练。
- 通过 MLOps（机器学习运维）功能更高效地协作，包括但不限于模型和数据的监控、审计和版本控制。

在接下来的部分中，您将找到一个快速入门指南，详细介绍如何使用 AzureML 运行 YOLO11 目标检测模型，可以从计算终端或笔记本运行。

## 前提条件

在开始之前，请确保您有权访问 AzureML 工作区。如果没有，您可以按照 Azure 的官方文档创建一个新的 [AzureML 工作区](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2)。此工作区作为管理所有 AzureML 资源的集中位置。

## 创建计算实例

从您的 AzureML 工作区，选择 Compute > Compute instances > New，选择具有所需资源的实例。

<p align="center">
  <img width="1280" src="https://github.com/ultralytics/docs/releases/download/0/create-compute-arrow.avif" alt="创建 Azure 计算实例">
</p>

## 从终端快速入门

启动您的计算实例并打开终端：

<p align="center">
  <img width="480" src="https://github.com/ultralytics/docs/releases/download/0/open-terminal.avif" alt="打开终端">
</p>

### 创建虚拟环境

使用您首选的 Python 版本创建 conda 虚拟环境并在其中安装 pip。Python 3.13.1 目前在 AzureML 中存在依赖问题，因此请使用 Python 3.12。

```bash
conda create --name yolo11env -y python=3.12
conda activate yolo11env
conda install pip -y
```

安装所需的依赖项：

```bash
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx
```

### 执行 YOLO11 任务

预测：

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

使用初始学习率 0.01 训练检测模型 10 个[轮次](https://www.ultralytics.com/glossary/epoch)：

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

您可以在[这里找到更多使用 Ultralytics CLI 的说明](../quickstart.md#use-ultralytics-with-cli)。

## 从笔记本快速入门

### 创建新的 IPython 内核

打开计算终端。

<p align="center">
  <img width="480" src="https://github.com/ultralytics/docs/releases/download/0/open-terminal.avif" alt="打开终端">
</p>

从计算终端，使用 Python 3.12 创建一个新的 ipykernel，笔记本将使用它来管理依赖项：

```bash
conda create --name yolo11env -y python=3.12
conda activate yolo11env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name yolo11env --display-name "yolo11env"
```

关闭终端并创建一个新笔记本。从笔记本中选择新创建的内核。

然后打开笔记本单元格并安装所需的依赖项：

```bash
%%bash
source activate yolo11env
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx
```

请注意，您需要在每个 `%%bash` 单元格中运行 `source activate yolo11env`，以确保单元格使用预期的环境。

使用 [Ultralytics CLI](../quickstart.md#use-ultralytics-with-cli) 运行一些预测：

```bash
%%bash
source activate yolo11env
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

或者使用 [Ultralytics Python 接口](../quickstart.md#use-ultralytics-with-python)，例如训练模型：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")  # 加载官方 YOLO11n 模型

# 使用模型
model.train(data="coco8.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
path = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

您可以使用 Ultralytics CLI 或 Python 接口来运行 YOLO11 任务，如上面终端部分所述。

按照这些步骤，您应该能够在 AzureML 上快速运行 YOLO11 进行快速试验。对于更高级的用途，您可以参考本指南开头链接的完整 AzureML 文档。

## 深入探索 AzureML

本指南作为入门介绍，帮助您在 AzureML 上运行 YOLO11。然而，这只是 AzureML 所能提供功能的冰山一角。要深入了解并充分发挥 AzureML 在机器学习项目中的潜力，请考虑探索以下资源：

- [创建数据资产](https://learn.microsoft.com/azure/machine-learning/how-to-create-data-assets)：了解如何在 AzureML 环境中有效设置和管理数据资产。
- [启动 AzureML 作业](https://learn.microsoft.com/azure/machine-learning/how-to-train-model)：全面了解如何在 AzureML 上启动机器学习训练作业。
- [注册模型](https://learn.microsoft.com/azure/machine-learning/how-to-manage-models)：熟悉模型管理实践，包括注册、版本控制和部署。
- [使用 AzureML Python SDK 训练 YOLO11](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azure-machine-learning-python-sdk-8268696be8ba)：探索使用 AzureML Python SDK 训练 YOLO11 模型的分步指南。
- [使用 AzureML CLI 训练 YOLO11](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azureml-and-the-az-cli-73d3c870ba8e)：了解如何利用命令行界面在 AzureML 上简化 YOLO11 模型的训练和管理。

## 常见问题

### 如何在 AzureML 上运行 YOLO11 进行模型训练？

在 AzureML 上运行 YOLO11 进行模型训练涉及以下几个步骤：

1. **创建计算实例**：从您的 AzureML 工作区，导航到 Compute > Compute instances > New，选择所需的实例。

2. **设置环境**：启动计算实例，打开终端，创建 Conda 环境。设置 Python 版本（Python 3.13.1 尚不支持）：

    ```bash
    conda create --name yolo11env -y python=3.12
    conda activate yolo11env
    conda install pip -y
    pip install ultralytics onnx
    ```

3. **运行 YOLO11 任务**：使用 Ultralytics CLI 训练模型：
    ```bash
    yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
    ```

有关更多详细信息，您可以参考[使用 Ultralytics CLI 的说明](../quickstart.md#use-ultralytics-with-cli)。

### 使用 AzureML 进行 YOLO11 训练有什么好处？

AzureML 为训练 YOLO11 模型提供了强大且高效的生态系统：

- **可扩展性**：随着数据和模型复杂性的增长，轻松扩展计算资源。
- **MLOps 集成**：利用版本控制、监控和审计等功能来简化机器学习运维。
- **协作**：在团队内共享和管理资源，增强协作工作流程。

这些优势使 AzureML 成为从快速原型到大规模部署项目的理想平台。有关更多提示，请查看 [AzureML 作业](https://learn.microsoft.com/azure/machine-learning/how-to-train-model)。

### 如何排查在 AzureML 上运行 YOLO11 时的常见问题？

排查 AzureML 上 YOLO11 的常见问题可能涉及以下步骤：

- **依赖问题**：确保安装了所有必需的包。参考 `requirements.txt` 文件了解依赖项。
- **环境设置**：在运行命令之前，验证 conda 环境是否正确激活。
- **资源分配**：确保计算实例有足够的资源来处理训练工作负载。

有关更多指导，请查看我们的 [YOLO 常见问题](https://docs.ultralytics.com/guides/yolo-common-issues/)文档。

### 我可以在 AzureML 上同时使用 Ultralytics CLI 和 Python 接口吗？

是的，AzureML 允许您无缝使用 Ultralytics CLI 和 Python 接口：

- **CLI**：适合快速任务和直接从终端运行标准脚本。

    ```bash
    yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
    ```

- **Python 接口**：适用于需要自定义编码和在笔记本中集成的更复杂任务。

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(data="coco8.yaml", epochs=3)
    ```

有关分步说明，请参考 [CLI 快速入门指南](../quickstart.md#use-ultralytics-with-cli)和 [Python 快速入门指南](../quickstart.md#use-ultralytics-with-python)。

### 与其他[目标检测](https://www.ultralytics.com/glossary/object-detection)模型相比，使用 Ultralytics YOLO11 有什么优势？

Ultralytics YOLO11 相比其他目标检测模型具有几个独特优势：

- **速度**：与 Faster R-CNN 和 SSD 等模型相比，推理和训练时间更快。
- **[准确率](https://www.ultralytics.com/glossary/accuracy)**：在检测任务中具有高准确率，具有无锚点设计和增强的数据增强策略等特性。
- **易用性**：直观的 API 和 CLI 可快速设置，对初学者和专家都很友好。

要了解更多关于 YOLO11 的功能，请访问 [Ultralytics YOLO](https://www.ultralytics.com/yolo) 页面获取详细信息。
