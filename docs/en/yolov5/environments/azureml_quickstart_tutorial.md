---
comments: true
description: 学习如何在 AzureML 上设置和运行 Ultralytics YOLOv5。按照本快速入门指南在 AzureML 计算实例上轻松配置和训练模型。
keywords: YOLOv5, AzureML, 机器学习, 计算实例, 快速入门, 模型训练, 虚拟环境, Python, AI, 深度学习, Ultralytics
---

# Ultralytics YOLOv5 🚀 在 AzureML 上的快速入门

欢迎阅读 Microsoft Azure 机器学习 (AzureML) 上的 Ultralytics [YOLOv5](../../models/yolov5.md) 快速入门指南！本指南将引导您在 AzureML 计算实例上设置 YOLOv5，涵盖从创建虚拟环境到训练和运行模型推理的所有内容。

## 什么是 Azure？

[Azure](https://azure.microsoft.com/) 是 Microsoft 的综合[云计算](https://www.ultralytics.com/glossary/cloud-computing)平台。它提供广泛的服务，包括计算能力、数据库、分析工具、[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)功能和网络解决方案。Azure 使组织能够通过 Microsoft 管理的数据中心构建、部署和管理应用程序和服务，促进工作负载从本地基础设施迁移到云端。

## 什么是 Azure 机器学习 (AzureML)？

[Azure 机器学习](https://azure.microsoft.com/products/machine-learning) (AzureML) 是专为开发、训练和部署机器学习模型而设计的专业云服务。它为各种技能水平的数据科学家和开发人员提供协作环境和工具。主要功能包括[自动化机器学习 (AutoML)](https://www.ultralytics.com/glossary/automated-machine-learning-automl)、用于模型创建的拖放界面，以及用于更精细控制 ML 生命周期的强大 [Python](https://www.python.org/) SDK。AzureML 简化了将[预测建模](https://www.ultralytics.com/glossary/predictive-modeling)嵌入应用程序的过程。

## 先决条件

要遵循本指南，您需要一个有效的 [Azure 订阅](https://azure.microsoft.com/free/)和访问 [AzureML 工作区](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2)的权限。如果您还没有设置工作区，请参阅官方 [Azure 文档](https://learn.microsoft.com/azure/machine-learning/quickstart-create-resources?view=azureml-api-2)创建一个。

## 创建计算实例

AzureML 中的计算实例为数据科学家提供托管的基于云的工作站。

1.  导航到您的 AzureML 工作区。
2.  在左侧窗格中，选择 **计算**。
3.  转到 **计算实例** 选项卡并点击 **新建**。
4.  根据您的训练或推理需求选择适当的 CPU 或 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 资源来配置您的实例。

<img width="1741" alt="create-compute-arrow" src="https://github.com/ultralytics/docs/releases/download/0/create-compute-arrow.avif">

## 打开终端

一旦您的计算实例运行，您可以直接从 AzureML 工作室访问其终端。

1.  转到左侧窗格中的 **笔记本** 部分。
2.  在顶部下拉菜单中找到您的计算实例。
3.  点击文件浏览器下方的 **终端** 选项，打开到您实例的命令行界面。

![open-terminal-arrow](https://github.com/ultralytics/docs/releases/download/0/open-terminal-arrow.avif)

## 设置和运行 YOLOv5

现在，让我们设置环境并运行 Ultralytics YOLOv5。

### 1. 创建虚拟环境

使用虚拟环境管理依赖项是最佳实践。我们将使用 [Conda](https://docs.conda.io/en/latest/)，它预装在 AzureML 计算实例上。有关详细的 Conda 设置指南，请参阅 Ultralytics [Conda 快速入门指南](../../guides/conda-quickstart.md)。

创建具有特定 Python 版本的 Conda 环境（例如 `yolov5env`）并激活它：

```bash
conda create --name yolov5env -y python=3.10 # 创建新的 Conda 环境
conda activate yolov5env                     # 激活环境
conda install pip -y                         # 确保安装了 pip
```

### 2. 克隆 YOLOv5 仓库

使用 [Git](https://git-scm.com/) 从 [GitHub](https://github.com/) 克隆官方 Ultralytics YOLOv5 仓库：

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆仓库
cd yolov5                                       # 进入目录
# 初始化子模块（如果有的话，尽管 YOLOv5 通常不需要此步骤）
# git submodule update --init --recursive
```

### 3. 安装依赖项

安装 `requirements.txt` 文件中列出的必要 Python 包。我们还安装 [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange) 以获得模型导出功能。

```bash
pip install -r requirements.txt # 安装核心依赖
pip install "onnx>=1.12.0"      # 安装 ONNX 用于导出
```

### 4. 执行 YOLOv5 任务

设置完成后，您现在可以训练、验证、执行推理和导出您的 YOLOv5 模型。

- **训练**模型在 [COCO128](../../datasets/detect/coco128.md) 等数据集上。查看[训练模式](../../modes/train.md)文档了解更多详情。

    ```bash
    # 使用 yolov5s 预训练权重在 COCO128 数据集上开始训练
    python train.py --data coco128.yaml --weights yolov5s.pt --img 640 --epochs 10 --batch 16
    ```

- **验证**训练模型的性能，使用[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) 等指标。查看[验证模式](../../modes/val.md)指南了解选项。

    ```bash
    # 在 COCO128 验证集上验证 yolov5s 模型
    python val.py --weights yolov5s.pt --data coco128.yaml --img 640
    ```

- 对新图像或视频**运行推理**。探索[预测模式](../../modes/predict.md)文档了解各种推理来源。

    ```bash
    # 使用 yolov5s 对示例图像运行推理
    python detect.py --weights yolov5s.pt --source data/images --img 640
    ```

- **导出**模型为不同格式，如 ONNX、[TensorRT](https://www.ultralytics.com/glossary/tensorrt) 或 [CoreML](https://docs.ultralytics.com/integrations/coreml/) 以进行部署。参考[导出模式](../../modes/export.md)指南和 [ONNX 集成](../../integrations/onnx.md)页面。

    ```bash
    # 将 yolov5s 导出为 ONNX 格式
    python export.py --weights yolov5s.pt --include onnx --img 640
    ```

## 使用笔记本

如果您更喜欢交互式体验，可以在 AzureML 笔记本中运行这些命令。您需要创建一个链接到 Conda 环境的自定义 [IPython 内核](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)。

### 创建新的 IPython 内核

在计算实例终端中运行以下命令：

```bash
# 确保您的 Conda 环境已激活
# conda activate yolov5env

# 如果尚未安装，安装 ipykernel
conda install ipykernel -y

# 创建链接到您环境的新内核
python -m ipykernel install --user --name yolov5env --display-name "Python (yolov5env)"
```

创建内核后，刷新浏览器。当您打开或创建 `.ipynb` 笔记本文件时，从右上角的内核下拉菜单中选择您的新内核（"Python (yolov5env)"）。

### 在笔记本单元格中运行命令

- **Python 单元格：** Python 单元格中的代码将自动使用所选的 `yolov5env` 内核执行。

- **Bash 单元格：** 要运行 shell 命令，请在单元格开头使用 `%%bash` 魔术命令。请记住在每个 bash 单元格中激活您的 Conda 环境，因为它们不会自动继承笔记本的内核环境上下文。

    ```bash
    %%bash
    source activate yolov5env # 在单元格内激活环境

    # 示例：使用激活的环境运行验证
    python val.py --weights yolov5s.pt --data coco128.yaml --img 640
    ```

恭喜！您已成功在 AzureML 上设置和运行 Ultralytics YOLOv5。如需进一步探索，请考虑查看其他 [Ultralytics 集成](../../integrations/index.md)或详细的 [YOLOv5 文档](../index.md)。您可能还会发现 [AzureML 文档](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2)对于分布式训练或将模型部署为端点等高级场景很有用。
