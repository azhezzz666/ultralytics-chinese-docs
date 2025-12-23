---
comments: true
description: 学习如何在自定义数据集上训练 YOLOv5，包含简单易懂的步骤。详细介绍数据集准备、模型选择和训练过程。
keywords: YOLOv5, 自定义数据集, 模型训练, 目标检测, 机器学习, AI, YOLO 模型, PyTorch, 数据集准备, Ultralytics
---

# 在自定义数据上训练 YOLOv5

📚 本指南介绍如何使用 [YOLOv5](https://github.com/ultralytics/yolov5) 模型 🚀 训练你自己的**自定义数据集**。训练自定义模型是将[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)解决方案定制到特定实际应用的基本步骤，超越了通用[目标检测](https://docs.ultralytics.com/tasks/detect/)的范畴。

## 开始之前

首先，确保你已设置好必要的环境。克隆 YOLOv5 仓库并从 `requirements.txt` 安装所需依赖。需要 [**Python>=3.8.0**](https://www.python.org/) 环境和 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。如果本地没有找到模型和数据集，它们会自动从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆仓库
cd yolov5
pip install -r requirements.txt # 安装依赖
```

## 在自定义数据上训练

<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-active-learning-loop.avif" alt="Ultralytics 主动学习循环图，展示数据收集、标注、训练、部署和边缘案例收集"></a>
<br>
<br>

开发自定义[目标检测](https://docs.ultralytics.com/tasks/detect/)模型是一个迭代过程：

1.  **收集和整理图像**：收集与你特定任务相关的图像。高质量、多样化的数据至关重要。请参阅我们的[数据收集和标注](https://docs.ultralytics.com/guides/data-collection-and-annotation/)指南。
2.  **标注对象**：准确标注图像中感兴趣的对象。
3.  **训练模型**：使用标注数据[训练](https://docs.ultralytics.com/modes/train/)你的 YOLOv5 模型。利用[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)从预训练权重开始。
4.  **部署和预测**：使用训练好的模型对新的、未见过的数据进行[推理](https://docs.ultralytics.com/modes/predict/)。
5.  **收集边缘案例**：识别模型表现不佳的场景（[边缘案例](https://en.wikipedia.org/wiki/Edge_case)），并将类似数据添加到数据集中以提高鲁棒性。重复这个循环。

[Ultralytics HUB](https://docs.ultralytics.com/hub/) 为整个[机器学习运维（MLOps）](https://www.ultralytics.com/glossary/machine-learning-operations-mlops)周期提供了简化的无代码解决方案，包括数据集管理、模型训练和部署。

!!! question "许可证"

    Ultralytics 提供两种许可选项以适应不同的使用场景：

    - **AGPL-3.0 许可证**：这个 [OSI 批准](https://opensource.org/license/agpl-v3)的开源许可证非常适合热衷于开放协作和知识共享的学生、研究人员和爱好者。它要求衍生作品也在相同许可证下共享。完整详情请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
    - **企业许可证**：专为商业应用设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中，无需遵守 AGPL-3.0 的开源规定。如果你的项目需要商业部署，请申请[企业许可证](https://www.ultralytics.com/license)。

    在 [Ultralytics 许可](https://www.ultralytics.com/license)页面进一步了解我们的许可选项。

在开始训练之前，数据集准备是必不可少的。

## 1. 创建数据集

YOLOv5 模型需要标注数据来学习对象类别的视觉特征。正确组织数据集是关键。

### 1.1 创建 `dataset.yaml`

数据集配置文件（如 `coco128.yaml`）概述了数据集的结构、类别名称和图像目录路径。[COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) 是一个小型示例数据集，包含来自大型 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 数据集的前 128 张图像。它对于快速测试训练流程和诊断潜在问题（如[过拟合](https://www.ultralytics.com/glossary/overfitting)）非常有用。

`dataset.yaml` 文件结构包括：

- `path`：包含数据集的根目录。
- `train`、`val`、`test`：从 `path` 到包含图像或列出图像路径的文本文件的相对路径，分别用于训练集、验证集和测试集。
- `names`：将类别索引（从 0 开始）映射到相应类别名称的字典。

你可以将 `path` 设置为绝对目录（如 `/home/user/datasets/coco128`）或相对路径（如从 YOLOv5 仓库根目录启动训练时的 `../datasets/coco128`）。

以下是 `coco128.yaml` 的结构（[在 GitHub 上查看](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)）：

```yaml
# 相对于 yolov5 目录的数据集根目录
path: coco128

# 训练/验证/测试集：指定目录、*.txt 文件或列表
train: images/train2017 # 128 张训练图像
val: images/train2017 # 128 张验证图像
test: # 测试图像的可选路径

# 类别（使用 80 个 COCO 类别的示例）
names:
    0: person
    1: bicycle
    2: car
    # ...（其余 COCO 类别）
    77: teddy bear
    78: hair drier
    79: toothbrush
```

### 1.2 利用模型进行自动标注

虽然使用工具进行手动标注是常见方法，但这个过程可能很耗时。基础模型的最新进展为自动化或半自动化标注过程提供了可能，可以显著加快数据集创建速度。以下是一些可以帮助生成标注的模型示例：

- **[Google Gemini](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-google-gemini-models-for-object-detection-image-captioning-and-ocr.ipynb)**：像 Gemini 这样的大型多模态模型具有强大的图像理解能力。它们可以被提示识别和定位图像中的对象，生成可以转换为 YOLO 格式标注的边界框或描述。在提供的教程笔记本中探索其潜力。
- **[SAM2（Segment Anything Model 2）](https://docs.ultralytics.com/models/sam-2/)**：专注于分割的基础模型（如 SAM2）可以高精度地识别和描绘对象。虽然主要用于分割，但生成的掩码通常可以转换为适合目标检测任务的边界框标注。
- **[YOLOWorld](https://docs.ultralytics.com/models/yolo-world/)**：该模型提供开放词汇检测能力。你可以提供感兴趣对象的文本描述，YOLOWorld 可以在图像中定位它们，_无需_事先针对这些特定类别进行训练。这可以作为生成初始标注的起点，然后进行细化。

使用这些模型可以提供"预标注"步骤，减少所需的手动工作量。然而，审查和细化自动生成的标注以确保准确性和一致性至关重要，因为质量直接影响训练后的 YOLOv5 模型的性能。生成（并可能细化）标注后，确保它们符合 **YOLO 格式**：每张图像一个 `*.txt` 文件，每行代表一个对象，格式为 `class_index x_center y_center width height`（归一化坐标，类别索引从零开始）。如果图像中没有感兴趣的对象，则不需要相应的 `*.txt` 文件。

YOLO 格式 `*.txt` 文件的规范非常精确：

- 每个对象[边界框](https://www.ultralytics.com/glossary/bounding-box)一行。
- 每行必须包含：`class_index x_center y_center width height`。
- 坐标必须**归一化**到 0 到 1 之间的范围。为此，将 `x_center` 和 `width` 的像素值除以图像的总宽度，将 `y_center` 和 `height` 除以图像的总高度。
- 类别索引从零开始（即第一个类别用 `0` 表示，第二个用 `1` 表示，依此类推）。

<p align="center"><img width="750" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif" alt="标注了两个人和一条领带的示例图像"></p>

上图对应的标注文件，包含两个"person"对象（类别索引 `0`）和一个"tie"对象（类别索引 `27`），如下所示：

<p align="center"><img width="428" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-1.avif" alt="标注图像的 YOLO 格式标注文件内容示例"></p>

### 1.3 组织目录

按照下图所示结构组织你的[数据集](https://docs.ultralytics.com/datasets/)目录。默认情况下，YOLOv5 期望数据集目录（如 `/coco128`）位于 `/yolov5` 仓库目录**旁边**的 `/datasets` 文件夹中。

YOLOv5 通过将图像路径中最后一个 `/images/` 替换为 `/labels/` 来自动定位每张图像的标注。例如：

```bash
../datasets/coco128/images/im0.jpg # 图像文件路径
../datasets/coco128/labels/im0.txt # 对应标注文件路径
```

推荐的目录结构是：

```
/datasets/
└── coco128/  # 数据集根目录
    ├── images/
    │   ├── train2017/  # 训练图像
    │   │   ├── 000000000009.jpg
    │   │   └── ...
    │   └── val2017/    # 验证图像（如果训练/验证使用相同集合则可选）
    │       └── ...
    └── labels/
        ├── train2017/  # 训练标注
        │   ├── 000000000009.txt
        │   └── ...
        └── val2017/    # 验证标注（如果训练/验证使用相同集合则可选）
            └── ...
```

<p align="center"><img width="700" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-dataset-structure.avif" alt="推荐的 YOLOv5 数据集目录结构图"></p>

## 2. 选择模型

选择一个[预训练模型](https://docs.ultralytics.com/models/)来启动训练过程。从预训练权重开始可以显著加速学习并提高性能，相比从头开始训练效果更好。YOLOv5 提供各种模型大小，每种在速度和精度之间有不同的平衡。例如，[YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml) 是第二小和最快的模型，适合资源受限的环境。请参阅 [README 表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)以获取所有可用[模型](https://docs.ultralytics.com/models/)的详细比较。

<p align="center"><img width="800" alt="YOLOv5 模型比较图，显示大小、速度和精度" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

## 3. 训练

使用 `train.py` 脚本开始[模型训练](https://docs.ultralytics.com/modes/train/)。基本参数包括：

- `--img`：定义输入[图像尺寸](https://docs.ultralytics.com/usage/cfg/#image-size)（如 `--img 640`）。较大的尺寸通常产生更好的精度，但需要更多 GPU 内存。
- `--batch`：确定[批量大小](https://www.ultralytics.com/glossary/batch-size)（如 `--batch 16`）。选择 GPU 能处理的最大尺寸。
- `--epochs`：指定总训练[轮数](https://www.ultralytics.com/glossary/epoch)（如 `--epochs 100`）。一轮代表对整个训练数据集的一次完整遍历。
- `--data`：数据集 `dataset.yaml` 文件的路径（如 `--data coco128.yaml`）。
- `--weights`：初始权重文件的路径。强烈建议使用预训练权重（如 `--weights yolov5s.pt`）以获得更快的收敛和更好的结果。要从头开始训练（除非你有非常大的数据集和特定需求，否则不建议），使用 `--weights '' --cfg yolov5s.yaml`。

如果本地没有找到预训练权重，它们会自动从[最新 YOLOv5 发布版本](https://github.com/ultralytics/yolov5/releases)下载。

```bash
# 示例：在 COCO128 数据集上训练 YOLOv5s 3 轮
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

!!! tip "优化训练速度"

    💡 使用 `--cache ram` 或 `--cache disk` 将数据集图像分别缓存到 [RAM](https://en.wikipedia.org/wiki/Random-access_memory) 或本地磁盘。这可以显著加速训练，特别是当数据集 I/O（输入/输出）操作成为瓶颈时。注意这需要大量 RAM 或磁盘空间。

!!! tip "本地数据存储"

    💡 始终使用本地存储的数据集进行训练。从网络驱动器（如 Google Drive）或远程存储访问数据可能会显著变慢并影响训练性能。将数据集复制到本地 SSD 通常是最佳实践。

所有训练输出，包括权重和日志，都保存在 `runs/train/` 目录中。每次训练会话都会创建一个新的子目录（如 `runs/train/exp`、`runs/train/exp2` 等）。要获得交互式的实践体验，请在我们的官方教程笔记本中探索训练部分：<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>

## 4. 可视化

YOLOv5 与各种工具无缝集成，用于可视化训练进度、评估结果和实时监控性能。

### Comet 日志记录和可视化 🌟 新功能

[Comet](https://docs.ultralytics.com/integrations/comet/) 已完全集成，用于全面的实验跟踪。实时可视化指标，保存超参数，管理数据集和模型检查点，并使用交互式 [Comet 自定义面板](https://bit.ly/yolov5-colab-comet-panels)分析模型预测。

入门非常简单：

```bash
pip install comet_ml                                                          # 1. 安装 Comet 库
export COMET_API_KEY=YOUR_API_KEY_HERE                                        # 2. 设置你的 Comet API 密钥（在 Comet.ml 创建免费账户）
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt # 3. 训练模型 - Comet 自动记录一切！
```

在我们的 [Comet 集成指南](https://docs.ultralytics.com/integrations/comet/)中深入了解支持的功能。从官方[文档](https://bit.ly/yolov5-colab-comet-docs)了解更多 Comet 的功能。尝试 Comet Colab 笔记本进行实时演示：[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

<img width="1920" alt="Comet UI 显示 YOLOv5 训练指标和可视化" src="https://github.com/ultralytics/docs/releases/download/0/yolo-ui.avif">

### ClearML 日志记录和自动化 🌟 新功能

[ClearML](https://docs.ultralytics.com/integrations/clearml/) 集成支持详细的实验跟踪、数据集版本管理，甚至远程执行训练运行。通过以下简单步骤激活 ClearML：

- 安装包：`pip install clearml`
- 初始化 ClearML：运行一次 `clearml-init` 以连接到你的 ClearML 服务器（自托管或[免费层](https://clear.ml/)）。

ClearML 自动捕获实验详情、模型上传、比较、未提交的代码更改和已安装的包，确保完全可重现性。你可以轻松地在远程代理上调度训练任务，并使用 ClearML Data 管理数据集版本。探索 [ClearML 集成指南](https://docs.ultralytics.com/integrations/clearml/)以获取全面详情。

<a href="https://clear.ml/">
<img alt="ClearML 实验管理 UI 显示 YOLOv5 训练运行的图表和日志" src="https://github.com/ultralytics/docs/releases/download/0/clearml-experiment-management-ui.avif" width="1280"></a>

### 本地日志记录

训练结果使用 [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) 自动记录，并作为 [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) 文件保存在特定实验目录中（如 `runs/train/exp`）。记录的数据包括：

- 训练和验证损失及性能指标。
- 显示应用增强（如马赛克）的示例图像。
- 用于视觉检查的真实标注和模型预测。
- 关键评估指标，如[精确率](https://www.ultralytics.com/glossary/precision)-[召回率](https://www.ultralytics.com/glossary/recall)（PR）曲线。
- 用于详细类别性能分析的[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)。

<img alt="YOLOv5 训练的本地日志记录结果示例，包括图表和图像马赛克" src="https://github.com/ultralytics/docs/releases/download/0/local-logging-results.avif" width="1280">

`results.csv` 文件在每轮后更新，训练结束后绘制为 `results.png`。你也可以使用提供的实用函数手动绘制任何 `results.csv` 文件：

```python
from utils.plots import plot_results

# 从特定训练运行目录绘制结果
plot_results("runs/train/exp/results.csv")  # 这将在同一目录中生成 'results.png'
```

<p align="center"><img width="800" alt="results.png 绘图示例，显示训练指标如 mAP、精确率、召回率和损失随轮数变化" src="https://github.com/ultralytics/docs/releases/download/0/results.avif"></p>

## 5. 后续步骤

训练成功完成后，性能最佳的模型检查点（`best.pt`）已保存并准备好进行部署或进一步优化。可能的后续步骤包括：

- 通过 [CLI](https://github.com/ultralytics/yolov5#quick-start-examples) 或 [Python](./pytorch_hub_model_loading.md) 使用训练好的模型对新图像或视频运行[推理](https://docs.ultralytics.com/modes/predict/)。
- 执行[验证](https://docs.ultralytics.com/modes/val/)以评估模型在不同数据划分（如保留的测试集）上的[准确性](https://www.ultralytics.com/glossary/accuracy)和泛化能力。
- [导出](https://docs.ultralytics.com/modes/export/)模型到各种部署格式，如 [ONNX](https://docs.ultralytics.com/integrations/onnx/)、[TensorFlow SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/) 或 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/)，以便在不同平台上进行优化推理。
- 使用[超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning/)技术来挖掘额外的性能提升。
- 按照我们的[最佳训练效果技巧](https://docs.ultralytics.com/guides/model-training-tips/)继续改进模型，并根据性能分析迭代添加更多样化和具有挑战性的数据。

## 支持的环境

Ultralytics 提供配备了必要依赖项（如 [CUDA](https://developer.nvidia.com/cuda)、[cuDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/)）的即用环境，便于顺利开始。

- **免费 GPU 笔记本**：
    - <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a>
    - <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>
    - <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **云平台**：
    - **Google Cloud**：[GCP 快速入门指南](https://docs.ultralytics.com/integrations/google-colab/)
    - **Amazon AWS**：[AWS 快速入门指南](https://docs.ultralytics.com/integrations/amazon-sagemaker/)
    - **Microsoft Azure**：[AzureML 快速入门指南](https://docs.ultralytics.com/guides/azureml-quickstart/)
- **本地设置**：
    - **Docker**：[Docker 快速入门指南](https://docs.ultralytics.com/guides/docker-quickstart/) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 持续集成状态徽章"></a>

此徽章表示所有 YOLOv5 [GitHub Actions](https://github.com/ultralytics/yolov5/actions) [持续集成（CI）](https://www.ultralytics.com/glossary/continuous-integration-ci)测试均已成功通过。这些严格的 CI 测试涵盖核心功能，包括[训练](https://docs.ultralytics.com/modes/train/)、[验证](https://docs.ultralytics.com/modes/val/)、[推理](https://docs.ultralytics.com/modes/predict/)、[导出](https://docs.ultralytics.com/modes/export/)和[基准测试](https://docs.ultralytics.com/modes/benchmark/)，跨 macOS、Windows 和 Ubuntu 操作系统运行。测试每 24 小时自动执行一次，并在每次代码提交时执行，确保一致的稳定性和最佳性能。

## 常见问题

### 如何在自定义数据集上训练 YOLOv5？

在自定义数据集上训练 YOLOv5 涉及几个关键步骤：

1.  **准备数据集**：收集图像并进行标注。确保标注符合所需的 [YOLO 格式](https://docs.ultralytics.com/datasets/detect/)。将图像和标注组织到 `train/` 和 `val/`（以及可选的 `test/`）目录中。考虑使用 [Google Gemini](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-google-gemini-models-for-object-detection-image-captioning-and-ocr.ipynb)、[SAM2](https://docs.ultralytics.com/models/sam-2/) 或 [YOLOWorld](https://docs.ultralytics.com/models/yolo-world/) 等模型来辅助或自动化标注过程（参见第 1.2 节）。
2.  **设置环境**：克隆 YOLOv5 仓库并使用 `pip install -r requirements.txt` 安装依赖。
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```
3.  **创建数据集配置**：在 `dataset.yaml` 文件中定义数据集路径、类别数量和类别名称。
4.  **开始训练**：执行 `train.py` 脚本，提供 `dataset.yaml` 路径、所需的预训练权重（如 `yolov5s.pt`）、图像尺寸、批量大小和训练轮数。
    ```bash
    python train.py --img 640 --batch 16 --epochs 100 --data path/to/your/dataset.yaml --weights yolov5s.pt
    ```

### 为什么应该使用 Ultralytics HUB 训练 YOLO 模型？

[Ultralytics HUB](https://docs.ultralytics.com/hub/) 是一个综合平台，旨在简化整个 YOLO 模型开发生命周期，通常无需编写任何代码。主要优势包括：

- **简化训练**：使用预配置环境和直观的用户界面轻松训练模型。
- **集成数据管理**：在平台内高效上传、版本控制和管理数据集。
- **实时监控**：使用集成工具（如 [Comet](https://docs.ultralytics.com/integrations/comet/) 或 TensorBoard）跟踪训练进度和可视化性能指标。
- **协作功能**：通过共享资源、项目管理工具和轻松的模型共享促进团队合作。
- **无代码部署**：直接将训练好的模型部署到各种目标。

有关实践演练，请查看我们的博客文章：[如何使用 Ultralytics HUB 训练自定义模型](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub)。

### 如何将标注数据转换为 YOLOv5 格式？

无论你是手动标注还是使用自动化工具（如第 1.2 节中提到的那些），最终标注必须符合 YOLOv5 所需的特定 **YOLO 格式**：

- 为每张图像创建一个 `.txt` 文件。文件名应与图像文件名匹配（如 `image1.jpg` 对应 `image1.txt`）。将这些文件放在与 `images/` 目录平行的 `labels/` 目录中（如 `../datasets/mydataset/labels/train/`）。
- `.txt` 文件中的每一行代表一个对象标注，格式为：`class_index center_x center_y width height`。
- 坐标（`center_x`、`center_y`、`width`、`height`）必须相对于图像尺寸**归一化**（值在 0.0 到 1.0 之间）。
- 类别索引**从零开始**（第一个类别是 `0`，第二个是 `1`，等等）。

许多手动标注工具提供直接导出为 YOLO 格式的功能。如果使用自动化模型，你需要脚本或流程将其输出（如边界框坐标、分割掩码）转换为这种特定的归一化文本格式。确保最终数据集结构符合指南中提供的示例。有关更多详情，请参阅我们的[数据收集和标注指南](https://docs.ultralytics.com/guides/data-collection-and-annotation/)。

### 在商业应用中使用 YOLOv5 有哪些许可选项？

Ultralytics 提供灵活的许可以满足不同需求：

- **AGPL-3.0 许可证**：此开源许可证适用于学术研究、个人项目以及可接受开源合规性的情况。它要求修改和衍生作品也在 AGPL-3.0 下开源。查看 [AGPL-3.0 许可证详情](https://www.ultralytics.com/legal/agpl-3-0-software-license)。
- **企业许可证**：为将 YOLOv5 集成到专有产品或服务中的企业设计的商业许可证。此许可证消除了 AGPL-3.0 的开源义务，允许闭源分发。访问我们的[许可页面](https://www.ultralytics.com/license)了解更多详情或申请[企业许可证](https://www.ultralytics.com/legal/enterprise-software-license)。

选择最符合你项目需求和分发模式的许可证。
