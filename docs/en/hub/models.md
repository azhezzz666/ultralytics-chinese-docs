---
comments: true
description: 探索 Ultralytics HUB，轻松训练、分析、预览、部署和共享使用 YOLO11 的自定义视觉 AI 模型。立即开始训练！
keywords: Ultralytics HUB, YOLO11, 自定义 AI 模型, 模型训练, 模型部署, 模型分析, 视觉 AI
---

# Ultralytics HUB 模型

[Ultralytics HUB](https://www.ultralytics.com/hub) 模型为在自定义数据集上训练视觉 AI 模型提供了简化的解决方案。

该过程用户友好且高效，涉及简单的三步创建和由 Ultralytics YOLO11 驱动的加速训练。在训练期间，可以实时更新模型指标，以便您监控每个步骤的进度。训练完成后，您可以预览模型并轻松将其部署到实际应用中。因此，[Ultralytics HUB](https://www.ultralytics.com/hub) 提供了一个全面而简单的模型创建、训练、评估和部署系统。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YVlkq5H2tAQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics HUB 训练和验证概述
</p>

## 训练模型

点击侧边栏中的**模型**按钮导航到[模型](https://hub.ultralytics.com/models)页面，然后点击页面右上角的**训练模型**按钮。

![Ultralytics HUB 模型页面截图，箭头指向侧边栏中的模型按钮和训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-page.avif)

??? tip "提示"

    您可以直接从[主页](https://hub.ultralytics.com/home)训练模型。

    ![Ultralytics HUB 主页截图，箭头指向训练模型卡片](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-card.avif)

此操作将触发**训练模型**对话框，包含三个简单步骤：

### 1. 数据集

在此步骤中，您需要选择要训练模型的数据集。选择数据集后，点击**继续**。

![Ultralytics HUB 训练模型对话框截图，箭头指向数据集和继续按钮](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-dialog-dataset-step.avif)

??? tip "提示"

    如果您直接从数据集页面训练模型，可以跳过此步骤。

    ![Ultralytics HUB 数据集页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-page-train-model-button.avif)

### 2. 模型

在此步骤中，您需要选择要创建模型的项目、模型名称和模型架构。

![Ultralytics HUB 训练模型对话框截图，箭头指向项目下拉菜单、模型名称和继续按钮](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-dialog.avif)

??? note "注意"

    Ultralytics HUB 会尝试预选项目。

    如果您按上述方式打开**训练模型**对话框，[Ultralytics HUB](https://www.ultralytics.com/hub) 将预选您上次使用的项目。

    如果您从项目页面打开**训练模型**对话框，[Ultralytics HUB](https://www.ultralytics.com/hub) 将预选您所在的项目。

    ![Ultralytics HUB 项目页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-button.avif)

    如果您还没有创建项目，可以在此步骤中设置项目名称，它将与您的模型一起创建。

!!! info "信息"

    您可以在我们的文档中阅读更多关于可用 [YOLO 模型](https://docs.ultralytics.com/models/)和架构的信息。

默认情况下，您的模型将使用预训练模型（在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 数据集上训练）以减少训练时间。您可以通过打开**高级模型配置**折叠面板来更改此行为并调整模型配置。

![Ultralytics HUB 训练模型对话框截图，箭头指向高级模型配置折叠面板](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-2.avif)

!!! note "注意"

    您可以轻松更改最常见的模型配置选项（如轮次数量），但您也可以使用**自定义**选项访问与 [Ultralytics HUB](https://www.ultralytics.com/hub) 相关的所有[训练设置](https://docs.ultralytics.com/modes/train/#train-settings)。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Unt4Lfid7aY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Ultralytics HUB 中配置 Ultralytics YOLOv8 训练参数
</p>

或者，您可以点击**自定义**选项卡从之前训练的模型开始训练。

![Ultralytics HUB 训练模型对话框截图，箭头指向自定义选项卡](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-3.avif)

当您对模型配置满意时，点击**继续**。

### 3. 训练

在此步骤中，您将开始训练模型。

??? note "注意"

    在此步骤，您可以选择关闭**训练模型**对话框，稍后从模型页面开始训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向开始训练卡片](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-model-page-start-training.avif)

[Ultralytics HUB](https://www.ultralytics.com/hub) 提供三种训练选项：

- [Ultralytics 云](./cloud-training.md)
- Google Colab
- 自带代理

#### a. Ultralytics 云

您需要[升级](./pro.md#如何升级)到 [Pro 计划](./pro.md)才能访问 [Ultralytics 云](./cloud-training.md)。

![Ultralytics HUB 训练模型对话框截图](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-4.avif)

要使用我们的[云训练](./cloud-training.md)解决方案训练模型，请阅读 [Ultralytics 云训练](./cloud-training.md)文档。

#### b. Google Colab

要使用 [Google Colab](https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb) 开始训练模型，请按照 [Ultralytics HUB](https://www.ultralytics.com/hub) **训练模型**对话框或 [Google Colab](https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb) 笔记本中显示的说明操作。

<a href="https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

![Ultralytics HUB 训练模型对话框截图，箭头指向说明](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-instructions.avif)

训练开始后，您可以点击**完成**并在模型页面上监控训练进度。

![Ultralytics HUB 训练模型对话框截图，箭头指向完成按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-done-button.avif)

![Ultralytics HUB 正在训练的模型页面截图](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-progress.avif)

!!! note "注意"

    如果训练停止并保存了检查点，您可以从模型页面恢复训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向恢复训练卡片](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-resume-training.avif)

#### c. 自带代理

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/S_J-Dyw15i0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics HUB 进行自带代理模型训练
</p>

要使用您自己的代理开始训练模型，请按照 [Ultralytics HUB](https://www.ultralytics.com/hub) **训练模型**对话框中显示的说明操作。

![Ultralytics HUB 训练模型对话框截图，箭头指向说明](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-instructions-1.avif)

从 [PyPI](https://pypi.org/project/ultralytics/) 安装 `ultralytics` 包。

```bash
pip install -U ultralytics
```

接下来，使用提供的 Python 代码开始训练模型。

训练开始后，您可以点击**完成**并在模型页面上监控训练进度。

![Ultralytics HUB 训练模型对话框截图，箭头指向完成按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-done-button-1.avif)

![Ultralytics HUB 正在训练的模型页面截图](https://github.com/ultralytics/docs/releases/download/0/model-training-progress.avif)

!!! note "注意"

    如果训练停止并保存了检查点，您可以从模型页面恢复训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向恢复训练卡片](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-resume-training-1.avif)

## 分析模型

[训练模型](#训练模型)后，您可以分析模型指标。

**训练**选项卡根据任务精心分组展示最重要的指标。

![Ultralytics HUB 已训练模型页面截图](https://github.com/ultralytics/docs/releases/download/0/hub-analyze-model.avif)

要访问所有模型指标，请点击**图表**选项卡。

![Ultralytics HUB 模型页面预览选项卡截图，箭头指向图表选项卡](https://github.com/ultralytics/docs/releases/download/0/hub-analyze-model-2.avif)

??? tip "提示"

    每个图表都可以放大以获得更好的可视化效果。

    ![Ultralytics HUB 模型页面训练选项卡截图，箭头指向其中一个图表的展开图标](https://github.com/ultralytics/docs/releases/download/0/hub-analyze-model-train-tab-expand-icon.avif)

    ![Ultralytics HUB 模型页面训练选项卡截图，其中一个图表已展开](https://github.com/ultralytics/docs/releases/download/0/hub-analyze-model-train-tab-expanded-chart.avif)

    此外，为了正确分析数据，您可以使用缩放功能。

    ![Ultralytics HUB 模型页面训练选项卡截图，其中一个图表已展开并缩放](https://github.com/ultralytics/docs/releases/download/0/hub-analyze-model-zoomed-chart.avif)

## 预览模型

[训练模型](#训练模型)后，您可以点击**预览**选项卡预览模型。

在**测试**卡片中，您可以从训练期间使用的数据集中选择预览图像，或从您的设备上传图像。

![Ultralytics HUB 模型页面预览选项卡截图，箭头指向图表选项卡和测试卡片](https://github.com/ultralytics/docs/releases/download/0/hub-preview-model-charts-test-card.avif)

!!! note "注意"

    您也可以使用相机拍照并直接在其上运行推理。

    ![Ultralytics HUB 模型页面预览选项卡截图，箭头指向测试卡片中的相机选项卡](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-preview-camera-tab.avif)

此外，您可以通过[下载](https://www.ultralytics.com/app-install)我们的 [Ultralytics HUB App](app/index.md)，直接在您的 [iOS](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240) 或 [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) 移动设备上实时预览模型。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向实时预览卡片](https://github.com/ultralytics/docs/releases/download/0/deploy-tab-real-time-preview-card.avif)

## 部署模型

[训练模型](#训练模型)后，您可以将其导出为 13 种不同格式，包括 ONNX、OpenVINO、CoreML、[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)、Paddle 等。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/K69DUpSBNdA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics HUB 将 Ultralytics YOLO11 导出为 ONNX、OpenVINO 和其他格式 🚀
</p>

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向导出卡片和所有已导出的格式](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-deploy-export-formats.avif)

??? tip "提示"

    如果打开导出操作下拉菜单并点击**高级**选项，您可以自定义每种格式的导出选项。

    ![Ultralytics HUB 模型页面部署选项卡截图，箭头指向其中一种格式的高级选项](https://github.com/ultralytics/docs/releases/download/0/hub-deploy-model-advanced-option.avif)

!!! note "注意"

    如果打开导出操作下拉菜单并点击**高级**选项，您可以重新导出每种格式。

您还可以在生产环境中使用我们的[推理 API](./inference-api.md)。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向 Ultralytics 推理 API 卡片](https://github.com/ultralytics/docs/releases/download/0/hub-inference-api-card.avif)

阅读 [Ultralytics 推理 API](./inference-api.md) 文档了解更多信息。

## 共享模型

!!! info "信息"

    [Ultralytics HUB](https://www.ultralytics.com/hub) 的共享功能提供了一种方便的方式与他人共享模型。此功能旨在同时满足现有 [Ultralytics HUB](https://www.ultralytics.com/hub) 用户和尚未创建账户的用户。

??? note "注意"

    您可以控制模型的通用访问权限。

    您可以选择将通用访问权限设置为"私有"，在这种情况下，只有您可以访问它。或者，您可以将通用访问权限设置为"未列出"，这将授予任何拥有模型直接链接的人查看权限，无论他们是否拥有 [Ultralytics HUB](https://www.ultralytics.com/hub) 账户。

导航到要共享的模型的模型页面，打开模型操作下拉菜单，点击**共享**选项。此操作将触发**共享模型**对话框。

![Ultralytics HUB 模型页面截图，箭头指向共享选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-model.avif)

??? tip "提示"

    您也可以直接从[模型](https://hub.ultralytics.com/models)页面或模型所在项目的项目页面共享模型。

    ![Ultralytics HUB 模型页面截图，箭头指向其中一个模型的共享选项](https://github.com/ultralytics/docs/releases/download/0/hub-share-model-2.avif)

将通用访问权限设置为"未列出"，然后点击**保存**。

![Ultralytics HUB 共享模型对话框截图，箭头指向下拉菜单和保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-share-model-dialog.avif)

现在，任何拥有您模型直接链接的人都可以查看它。

??? tip "提示"

    您可以轻松点击**共享模型**对话框中显示的模型链接来复制它。

    ![Ultralytics HUB 共享模型对话框截图，箭头指向模型链接](https://github.com/ultralytics/docs/releases/download/0/hub-share-model-link.avif)

## 编辑模型

导航到要编辑的模型的模型页面，打开模型操作下拉菜单，点击**编辑**选项。此操作将触发**更新模型**对话框。

![Ultralytics HUB 模型页面截图，箭头指向编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-model-1.avif)

??? tip "提示"

    您也可以直接从[模型](https://hub.ultralytics.com/models)页面或模型所在项目的项目页面编辑模型。

    ![Ultralytics HUB 模型页面截图，箭头指向其中一个模型的编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-model-2.avif)

对模型应用所需的修改，然后点击**保存**确认更改。

![Ultralytics HUB 更新模型对话框截图，箭头指向保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-edit-model-save-button.avif)

## 删除模型

导航到要删除的模型的模型页面，打开模型操作下拉菜单，点击**删除**选项。此操作将删除模型。

![Ultralytics HUB 模型页面截图，箭头指向删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-model-1.avif)

??? tip "提示"

    您也可以直接从[模型](https://hub.ultralytics.com/models)页面或模型所在项目的项目页面删除模型。

    ![Ultralytics HUB 模型页面截图，箭头指向其中一个模型的删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-model-2.avif)

!!! note "注意"

    如果您改变主意，可以从[回收站](https://hub.ultralytics.com/trash)页面恢复模型。

    ![Ultralytics HUB 回收站页面截图，箭头指向其中一个模型的恢复选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-trash-restore-option.avif)
