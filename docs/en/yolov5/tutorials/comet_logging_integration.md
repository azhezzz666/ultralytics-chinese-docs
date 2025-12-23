---
comments: true
description: 学习如何使用 Comet 跟踪、可视化和优化 YOLOv5 模型指标，实现无缝的机器学习工作流程。
keywords: YOLOv5, Comet, 机器学习, 模型跟踪, 超参数, 可视化, 深度学习, 日志记录, 指标
---

![Comet](https://cdn.comet.ml/img/notebook_logo.png)

# YOLOv5 与 Comet

本指南将介绍如何将 YOLOv5 与 [Comet](https://bit.ly/yolov5-readme-comet2) 结合使用，Comet 是一个用于跟踪、比较和优化机器学习实验的强大工具。

## 关于 Comet

[Comet](https://bit.ly/yolov5-readme-comet2) 构建的工具帮助数据科学家、工程师和团队负责人加速和优化[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。

实时跟踪和可视化模型指标，保存您的超参数、数据集和模型检查点，并使用 [Comet 自定义面板](https://www.comet.com/docs/v2/guides/comet-dashboard/code-panels/about-panels/?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github)可视化您的模型预测！Comet 确保您永远不会丢失工作，并使各种规模的团队轻松共享结果和协作！

## 入门

### 安装 Comet

```bash
pip install comet_ml
```

### 配置 Comet 凭据

有两种方式配置 Comet 与 YOLOv5。

您可以通过环境变量设置凭据：

**环境变量**

```bash
export COMET_API_KEY=YOUR_API_KEY
export COMET_PROJECT_NAME=YOUR_COMET_PROJECT_NAME # 默认为 'yolov5'
```

或者在工作目录中创建 `.comet.config` 文件并在其中设置凭据：

**Comet 配置文件**

```
[comet]
api_key=YOUR_API_KEY
project_name=YOUR_COMET_PROJECT_NAME # 默认为 'yolov5'
```

### 运行训练脚本

```bash
# 在 COCO128 上训练 YOLOv5s 5 个轮次
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```

就是这样！Comet 将自动记录您的超参数、命令行参数、训练和验证指标。您可以在 Comet UI 中可视化和分析您的运行。

![Comet UI 与 YOLOv5 训练](https://github.com/ultralytics/docs/releases/download/0/yolo-ui.avif)

## 试试示例！

查看[已完成运行的示例](https://www.comet.com/examples/comet-example-yolov5/a0e29e0e9b984e4a822db2a62d0cb357?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github)。

或者更好的是，在这个 Colab Notebook 中亲自尝试：

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

## 自动记录

默认情况下，Comet 将记录以下项目：

### 指标

- 训练和[验证数据](https://www.ultralytics.com/glossary/validation-data)的边界框损失、目标损失、分类损失
- 验证数据的 mAP_0.5、mAP_0.5:0.95 指标
- 验证数据的[精确率](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)

### 参数

- 模型超参数
- 通过命令行选项传递的所有参数

### 可视化

- 验证数据上模型预测的[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)
- 所有类别的 PR 和 F1 曲线图
- 类别标签的相关图


## 配置 Comet 日志记录

可以通过传递给训练脚本的命令行标志或通过环境变量配置 Comet 记录额外数据：

```bash
export COMET_MODE=online                           # 设置 Comet 以 'online' 或 'offline' 模式运行。默认为 online
export COMET_MODEL_NAME="yolov5"                   # 设置保存模型的名称。默认为 yolov5
export COMET_LOG_CONFUSION_MATRIX=false            # 设置为禁用记录 Comet 混淆矩阵。默认为 true
export COMET_MAX_IMAGE_UPLOADS=30                  # 控制记录到 Comet 的总图像预测数量。默认为 100
export COMET_LOG_PER_CLASS_METRICS=true            # 设置为在训练结束时记录每个检测类别的评估指标。默认为 false
export COMET_DEFAULT_CHECKPOINT_FILENAME="last.pt" # 如果您想从不同的检查点恢复训练，请设置此项。默认为 'last.pt'
export COMET_LOG_BATCH_LEVEL_METRICS=true          # 如果您想在批次级别记录训练指标，请设置此项。默认为 false
export COMET_LOG_PREDICTIONS=true                  # 设置为 false 以禁用记录模型预测
```

## 使用 Comet 记录检查点

默认情况下，将模型记录到 Comet 是禁用的。要启用它，请将 `save-period` 参数传递给训练脚本。这将根据 `save-period` 提供的间隔值将记录的检查点保存到 Comet：

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --save-period 1
```

## 记录模型预测

默认情况下，模型预测（图像、真实标签和边界框）将被记录到 Comet。

您可以通过传递 `bbox_interval` 命令行参数来控制记录预测的频率和相关图像。可以使用 Comet 的[目标检测](https://www.ultralytics.com/glossary/object-detection)自定义面板可视化预测。此频率对应于每个[轮次](https://www.ultralytics.com/glossary/epoch)的每第 N 批数据。在下面的示例中，我们记录每个轮次的每第 2 批数据。

**注意：** YOLOv5 验证数据加载器默认[批次大小](https://www.ultralytics.com/glossary/batch-size)为 32，因此您需要相应地设置记录频率。

这是一个[使用面板的示例项目](https://www.comet.com/examples/comet-example-yolov5?shareable=YcwMiJaZSXfcEXpGOHDD12vA1&utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github)

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --bbox_interval 2
```

### 控制记录到 Comet 的预测图像数量

从 YOLOv5 记录预测时，Comet 将记录与每组预测相关的图像。默认情况下，最多记录 100 张验证图像。您可以使用 `COMET_MAX_IMAGE_UPLOADS` 环境变量增加或减少此数量：

```bash
env COMET_MAX_IMAGE_UPLOADS=200 python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --bbox_interval 1
```

### 记录类别级别指标

使用 `COMET_LOG_PER_CLASS_METRICS` 环境变量记录每个类别的 mAP、精确率、召回率、f1：

```bash
env COMET_LOG_PER_CLASS_METRICS=true python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt
```

## 将数据集上传到 Comet Artifacts

如果您想使用 [Comet Artifacts](https://www.comet.com/docs/v2/guides/data-management/using-artifacts/#learn-more?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github) 存储数据，可以使用 `upload_dataset` 标志。

数据集应按照 [YOLOv5 文档](./train_custom_data.md)中描述的方式组织。数据集配置 `yaml` 文件必须遵循与 `coco128.yaml` 文件相同的格式。

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data coco128.yaml \
  --weights yolov5s.pt \
  --upload_dataset
```

您可以在 Comet 工作区的 Artifacts 选项卡中找到上传的数据集：

![Comet Artifacts 选项卡](https://github.com/ultralytics/docs/releases/download/0/artifact-1.avif)

您可以直接在 Comet UI 中预览数据：

![Comet 数据预览](https://github.com/ultralytics/docs/releases/download/0/artifact-2.avif)

Artifacts 支持版本控制，还支持添加有关数据集的元数据。Comet 将自动从您的数据集 `yaml` 文件记录元数据：

![Comet Artifact 元数据](https://github.com/ultralytics/docs/releases/download/0/artifact-metadata-logging.avif)

### 使用保存的 Artifact

如果您想使用 Comet Artifacts 中的数据集，请将数据集 `yaml` 文件中的 `path` 变量设置为指向以下 Artifact 资源 URL：

```yaml
# artifact.yaml 文件内容
path: "comet://WORKSPACE_NAME/ARTIFACT_NAME:ARTIFACT_VERSION_OR_ALIAS"
```

然后按以下方式将此文件传递给训练脚本：

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data artifact.yaml \
  --weights yolov5s.pt
```

Artifacts 还允许您跟踪数据在实验工作流程中流动时的血缘关系。在这里，您可以看到一个图表，显示所有使用您上传数据集的实验：

![Comet Artifact 血缘图](https://github.com/ultralytics/docs/releases/download/0/artifact-lineage-graph.avif)

## 恢复训练运行

如果您的训练运行因任何原因中断，例如网络连接中断，您可以使用 `resume` 标志和 Comet 运行路径恢复运行。

运行路径格式为 `comet://WORKSPACE_NAME/PROJECT_NAME/EXPERIMENT_ID`。

这将把运行恢复到中断前的状态，包括从检查点恢复模型、恢复所有超参数和训练参数，以及下载原始运行中使用的 Comet 数据集 Artifacts。恢复的运行将继续记录到 Comet UI 中的现有实验：

```bash
python train.py \
  --resume "comet://YOUR_RUN_PATH"
```

## 使用 Comet Optimizer 进行超参数搜索

YOLOv5 还与 [Comet 的 Optimizer](https://www.comet.com/docs/v2/guides/optimizer/configure-optimizer/) 集成，使在 Comet UI 中可视化超参数扫描变得简单。

### 配置 Optimizer 扫描

要配置 Comet Optimizer，您需要创建一个包含扫描信息的 JSON 文件。`utils/loggers/comet/optimizer_config.json` 中提供了一个示例文件：

```bash
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json"
```

`hpo.py` 脚本接受与 `train.py` 相同的参数。如果您希望向扫描传递额外参数，只需在脚本后添加它们：

```bash
python utils/loggers/comet/hpo.py \
  --comet_optimizer_config "utils/loggers/comet/optimizer_config.json" \
  --save-period 1 \
  --bbox_interval 1
```

## 可视化结果

Comet 提供多种方式来可视化扫描结果。查看[已完成扫描的项目](https://www.comet.com/examples/comet-example-yolov5/view/PrlArHGuuhDTKC1UuBmTtOSXD/panels?utm_source=yolov5&utm_medium=partner&utm_campaign=partner_yolov5_2022&utm_content=github)。

![Comet 超参数可视化](https://github.com/ultralytics/docs/releases/download/0/hyperparameter-yolo.avif)
