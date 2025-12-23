---
comments: true
description: 学习如何使用 Roboflow 强大的工具收集、标注和部署自定义 Ultralytics YOLO 模型的数据。轻松优化您的计算机视觉流水线。
keywords: Roboflow, Ultralytics YOLO, 数据标注, 计算机视觉, 模型训练, 模型部署, 数据集管理, 自动图像标注, AI 工具
---

# Roboflow 集成

[Roboflow](https://roboflow.com/?ref=ultralytics) 提供了一套专为构建和部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型设计的工具。您可以使用其 API 和 SDK 在开发流水线的各个阶段集成 Roboflow，或利用其端到端界面管理从图像收集到推理的整个过程。Roboflow 提供[数据标注](https://www.ultralytics.com/glossary/data-labeling)、[模型训练](https://docs.ultralytics.com/modes/train/)和[模型部署](https://docs.ultralytics.com/guides/model-deployment-options/)功能，为与 Ultralytics 工具一起开发自定义计算机视觉解决方案提供组件。

!!! question "许可证"

    Ultralytics 提供两种许可选项以适应不同的使用场景：

    - **AGPL-3.0 许可证**：这个[经 OSI 批准的开源许可证](https://www.ultralytics.com/legal/agpl-3-0-software-license)非常适合学生和爱好者，促进开放协作和知识共享。详情请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
    - **企业许可证**：专为商业用途设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中。如果您的场景涉及商业应用，请通过 [Ultralytics 许可](https://www.ultralytics.com/license)联系我们。

    更多详情请参阅 [Ultralytics 许可页面](https://www.ultralytics.com/license)。

本指南演示如何使用 Roboflow 查找、标注和组织数据，以训练自定义 [Ultralytics YOLO11](../models/yolo11.md) 模型。

- [收集数据以训练自定义 YOLO11 模型](#收集数据以训练自定义-yolo11-模型)
- [上传、转换和标注 YOLO11 格式的数据](#上传转换和标注-yolo11-格式的数据)
- [预处理和增强数据以提高模型鲁棒性](#预处理和增强数据以提高模型鲁棒性)
- [YOLO11 的数据集管理](#yolo11-的数据集管理)
- [导出 40+ 种格式的数据用于模型训练](#导出-40-种格式的数据用于模型训练)
- [上传自定义 YOLO11 模型权重用于测试和部署](#上传自定义-yolo11-模型权重用于测试和部署)
- [如何评估 YOLO11 模型](#如何评估-yolo11-模型)
- [学习资源](#学习资源)
- [项目展示](#项目展示)
- [常见问题](#常见问题)

## 收集数据以训练自定义 YOLO11 模型

Roboflow 提供两项主要服务来协助 Ultralytics [YOLO 模型](../models/index.md)的数据收集：Universe 和 Collect。有关数据收集策略的更多一般信息，请参阅我们的[数据收集和标注指南](../guides/data-collection-and-annotation.md)。

### Roboflow Universe

Roboflow Universe 是一个在线存储库，包含大量视觉[数据集](../datasets/index.md)。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe.avif" alt="Roboflow Universe" width="800">
</p>

使用 Roboflow 账户，您可以导出 Universe 上可用的数据集。要导出数据集，请使用相关数据集页面上的"Download this Dataset"按钮。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export.avif" alt="Roboflow Universe 数据集导出" width="800">
</p>

为了与 Ultralytics [YOLO11](../models/yolo11.md) 兼容，选择"YOLO11"作为导出格式：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export-1.avif" alt="Roboflow Universe 数据集导出格式选择" width="800">
</p>

Universe 还提供一个页面，汇集了上传到 Roboflow 的公开微调 YOLO 模型。这对于探索用于测试或自动数据标注的预训练模型非常有用。

### Roboflow Collect

如果您更喜欢自己收集图像，Roboflow Collect 是一个开源项目，可以通过边缘设备上的网络摄像头自动收集图像。您可以使用文本或图像提示来指定要收集的数据，帮助仅捕获视觉模型所需的图像。

## 上传、转换和标注 YOLO11 格式的数据

Roboflow Annotate 是一个在线工具，用于为各种计算机视觉任务标注图像，包括[目标检测](../tasks/detect.md)、[分类](../tasks/classify.md)和[分割](../tasks/segment.md)。

要为 Ultralytics [YOLO](../models/index.md) 模型（支持检测、实例分割、分类、姿态估计和 OBB）标注数据，首先在 Roboflow 中创建一个项目。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/create-roboflow-project.avif" alt="创建 Roboflow 项目" width="400">
</p>

接下来，将您的图像和其他工具的现有标注上传到 Roboflow。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/upload-images-to-roboflow.avif" alt="上传图像到 Roboflow" width="800">
</p>

上传后，您将被引导到标注页面。选择上传的图像批次并点击"Start Annotating"开始标注。

### 标注工具

- **边界框标注**：按 `B` 或点击框图标。点击并拖动以创建[边界框](https://www.ultralytics.com/glossary/bounding-box)。弹出窗口将提示您为标注选择类别。
- **多边形标注**：用于[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)。按 `P` 或点击多边形图标。点击对象周围的点来绘制多边形。

### 标注助手（SAM 集成）

Roboflow 集成了基于 [Segment Anything Model (SAM)](../models/sam.md) 的标注助手，可能加速标注过程。

要使用标注助手，点击侧边栏中的光标图标。SAM 将为您的项目启用。

将鼠标悬停在对象上，SAM 可能会建议标注。点击接受标注。您可以通过点击建议区域内部或外部来细化标注的精确度。

### 标签

您可以使用侧边栏中的标签面板为图像添加标签。标签可以表示位置、相机来源等属性。这些标签允许您搜索特定图像并生成包含具有特定标签图像的数据集版本。

### 标注辅助（基于模型）

托管在 Roboflow 上的模型可以与 Label Assist 一起使用，这是一个自动标注工具，利用您训练的 [YOLO11](../models/yolo11.md) 模型来建议标注。首先，将您的 YOLO11 模型权重上传到 Roboflow（见下面的说明）。然后，通过点击左侧边栏中的魔术棒图标并选择您的模型来激活 Label Assist。

选择您的模型并点击"Continue"以启用 Label Assist：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="在 Roboflow 中启用 Label Assist" width="800">
</p>

当您打开新图像进行标注时，Label Assist 可能会根据您模型的预测自动建议标注。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="Label Assist 根据训练模型推荐标注" width="800">
</p>

## YOLO11 的数据集管理

Roboflow 提供多种工具来理解和管理您的计算机视觉[数据集](../datasets/index.md)。

### 数据集搜索

使用数据集搜索根据语义文本描述（例如"查找所有包含人的图像"）或特定标签/标记来查找图像。通过点击侧边栏中的"Dataset"并使用搜索栏和过滤器来访问此功能。

例如，搜索包含人的图像：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/searching-for-an-image.avif" alt="在 Roboflow 数据集中搜索图像" width="800">
</p>

您可以通过"Tags"选择器使用标签细化搜索：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/filter-images-by-tag.avif" alt="在 Roboflow 中按标签过滤图像" width="350">
</p>

### 健康检查

在训练之前，使用 Roboflow 健康检查来获取数据集的洞察并识别潜在改进。通过"Health Check"侧边栏链接访问。它提供图像大小、类别平衡、标注热图等统计信息。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-dataset-health-check.avif" alt="Roboflow 健康检查分析仪表板" width="800">
</p>

健康检查可能会建议更改以提高性能，例如解决类别平衡功能中识别的类别不平衡问题。理解数据集健康状况对于有效的[模型训练](../modes/train.md)至关重要。

## 预处理和增强数据以提高模型鲁棒性

要导出数据，您需要创建数据集版本，这是数据集在特定时间点的快照。点击侧边栏中的"Versions"，然后点击"Create New Version"。在这里，您可以应用预处理步骤和[数据增强](https://www.ultralytics.com/glossary/data-augmentation)来潜在地提高模型鲁棒性。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/creating-dataset-version-on-roboflow.avif" alt="在 Roboflow 上创建带有预处理和增强选项的数据集版本" width="800">
</p>

对于每个选定的增强，弹出窗口允许您微调其参数，如亮度。适当的增强可以显著提高模型泛化能力，这是我们[模型训练技巧指南](../guides/model-training-tips.md)中讨论的关键概念。

## 导出 40+ 种格式的数据用于模型训练

生成数据集版本后，您可以将其导出为适合模型训练的各种格式。点击版本页面上的"Export Dataset"按钮。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/exporting-dataset.avif" alt="从 Roboflow 导出数据集" width="800">
</p>

选择"YOLO11"格式以与 Ultralytics 训练流水线兼容。现在您可以开始训练自定义 [YOLO11](../models/yolo11.md) 模型了。有关使用导出数据集启动训练的详细说明，请参阅 [Ultralytics 训练模式文档](../modes/train.md)。

## 上传自定义 YOLO11 模型权重用于测试和部署

Roboflow 为部署的模型提供可扩展的 API，以及与 [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing)、[Luxonis OAK](https://www.luxonis.com/)、[Raspberry Pi](../guides/raspberry-pi.md) 和基于 GPU 的系统等设备兼容的 SDK。在我们的指南中探索各种[模型部署选项](../guides/model-deployment-options.md)。

您可以使用简单的 [Python](https://www.python.org/) 脚本将 YOLO11 模型权重上传到 Roboflow 来部署模型。

创建一个新的 Python 文件并添加以下代码：

```python
import roboflow  # 使用 'pip install roboflow' 安装

# 登录 Roboflow（需要 API 密钥）
roboflow.login()

# 初始化 Roboflow 客户端
rf = roboflow.Roboflow()

# 定义您的工作区和项目详情
WORKSPACE_ID = "your-workspace-id"  # 替换为您的实际工作区 ID
PROJECT_ID = "your-project-id"  # 替换为您的实际项目 ID
VERSION = 1  # 替换为您想要的数据集版本号
MODEL_PATH = "path/to/your/runs/detect/train/"  # 替换为您的 YOLO11 训练结果目录路径

# 获取项目和版本
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION)

# 上传模型权重用于部署
# 确保 MODEL_PATH 指向包含 'best.pt' 的目录
dataset.deploy(
    model_type="yolov8",
    model_path=MODEL_PATH,
)  # 注意：在 Roboflow 部署中使用 "yolov8" 作为 model_type 以兼容 YOLO11

print(f"来自 {MODEL_PATH} 的模型已上传到 Roboflow 项目 {PROJECT_ID}，版本 {VERSION}。")
print("部署可能需要最多 30 分钟。")
```

在此代码中，将 `your-workspace-id`、`your-project-id`、`VERSION` 号和 `MODEL_PATH` 替换为您的 Roboflow 账户、项目和本地训练结果目录的特定值。确保 `MODEL_PATH` 正确指向包含训练好的 `best.pt` 权重文件的目录。

运行上述代码时，系统会要求您进行身份验证（通常通过 API 密钥）。然后，您的模型将被上传，并为您的项目创建一个 API 端点。此过程可能需要最多 30 分钟才能完成。

要测试您的模型并查找支持的 SDK 的部署说明，请转到 Roboflow 侧边栏中的"Deploy"选项卡。在此页面顶部，将出现一个小部件，允许您使用网络摄像头或上传图像或视频来测试您的模型。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/running-inference-example-image.avif" alt="使用 Roboflow 部署小部件在示例图像上运行推理" width="800">
</p>

您上传的模型还可以用作标注助手，根据其训练在新图像上建议标注。

## 如何评估 YOLO11 模型

Roboflow 提供评估模型性能的功能。理解[性能指标](../guides/yolo-performance-metrics.md)对于模型迭代至关重要。

上传模型后，通过 Roboflow 仪表板上的模型页面访问模型评估工具。点击"View Detailed Evaluation"。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-model-evaluation.avif" alt="启动 Roboflow 模型评估" width="800">
</p>

此工具显示一个[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)来说明模型性能，以及使用 [CLIP](https://openai.com/research/clip) 嵌入的交互式向量分析图。这些功能有助于识别模型改进的领域。

混淆矩阵弹出窗口：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/confusion-matrix.avif" alt="Roboflow 中显示的混淆矩阵" width="800">
</p>

将鼠标悬停在单元格上可查看值，点击单元格可查看带有模型预测和真实数据的相应图像。

点击"Vector Analysis"可查看基于 CLIP 嵌入可视化图像相似性的散点图。距离较近的图像在语义上相似。点表示图像，颜色从白色（性能良好）到红色（性能较差）。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/vector-analysis-plot.avif" alt="Roboflow 中使用 CLIP 嵌入的向量分析图" width="800">
</p>

向量分析有助于：

- 识别图像聚类。
- 精确定位模型表现不佳的聚类。
- 理解导致性能不佳的图像之间的共性。

## 学习资源

探索这些资源以了解更多关于将 Roboflow 与 Ultralytics YOLO11 一起使用的信息：

- **[在自定义数据集上训练 YOLO11（Colab）](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)**：一个交互式 [Google Colab](../integrations/google-colab.md) 笔记本，指导您在自己的数据上训练 YOLO11。
- **[YOLO11 文档](../models/yolo11.md)**：了解在 Ultralytics 框架内训练、导出和部署 YOLO11 模型。
- **[Ultralytics 博客](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai)**：包含计算机视觉文章，包括 [YOLO11 训练](../modes/train.md)和标注最佳实践。
- **[Ultralytics YouTube 频道](https://www.youtube.com/@Ultralytics)**：提供计算机视觉主题的深入视频指南，从模型训练到自动标注和[部署](../guides/model-deployment-options.md)。

## 项目展示

结合 Ultralytics YOLO11 和 Roboflow 的用户反馈：

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-1.avif" alt="展示图像 1" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-2.avif" alt="展示图像 2" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-3.avif" alt="展示图像 3" width="500">
</p>

## 常见问题

## 常见问题

### 如何使用 Roboflow 为 YOLO11 模型标注数据？

使用 Roboflow Annotate。创建项目，上传图像，使用标注工具（`B` 用于[边界框](https://www.ultralytics.com/glossary/bounding-box)，`P` 用于多边形）或基于 SAM 的标注助手进行更快的标注。详细步骤请参阅[上传、转换和标注数据部分](#上传转换和标注-yolo11-格式的数据)。

### Roboflow 为收集 YOLO11 训练数据提供哪些服务？

Roboflow 提供 Universe（访问大量[数据集](../datasets/index.md)）和 Collect（通过网络摄像头自动收集图像）。这些可以帮助获取 YOLO11 模型所需的[训练数据](https://www.ultralytics.com/glossary/training-data)，补充我们[数据收集指南](../guides/data-collection-and-annotation.md)中概述的策略。

### 如何使用 Roboflow 管理和分析我的 YOLO11 数据集？

利用 Roboflow 的数据集搜索、标签和健康检查功能。搜索可通过文本或标签查找图像，而健康检查分析数据集质量（类别平衡、图像大小等）以指导训练前的改进。详情请参阅[数据集管理部分](#yolo11-的数据集管理)。

### 如何从 Roboflow 导出我的 YOLO11 数据集？

在 Roboflow 中创建数据集版本，应用所需的预处理和[增强](https://www.ultralytics.com/glossary/data-augmentation)，然后点击"Export Dataset"并选择 YOLO11 格式。该过程在[导出数据部分](#导出-40-种格式的数据用于模型训练)中有概述。这将准备好您的数据以供 Ultralytics [训练流水线](../modes/train.md)使用。

### 如何将 YOLO11 模型与 Roboflow 集成和部署？

使用提供的 Python 脚本将训练好的 YOLO11 权重上传到 Roboflow。这将创建一个可部署的 API 端点。有关脚本和说明，请参阅[上传自定义权重部分](#上传自定义-yolo11-模型权重用于测试和部署)。在我们的文档中探索更多[部署选项](../guides/model-deployment-options.md)。
