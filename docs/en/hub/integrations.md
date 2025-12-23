---
comments: true
description: 探索 Ultralytics HUB 与 Roboflow 等平台的无缝集成。了解如何导入数据集、训练模型并增强您的 AI 工作流程。
keywords: Ultralytics HUB, Roboflow 集成, 数据集导入, 模型训练, AI, 机器学习, 模型导出, ONNX, OpenVINO
---

# Ultralytics HUB 集成

了解 [Ultralytics HUB](https://www.ultralytics.com/hub) 与各种平台和格式的集成，以简化您的 [AI](https://www.ultralytics.com/glossary/artificial-intelligence-ai) 工作流程。

## 数据集

将您的数据集无缝导入 Ultralytics HUB 以进行高效的[模型训练](../modes/train.md)。

数据集导入后，您可以像使用原生 Ultralytics HUB 数据集一样在其上[训练模型](./models.md#训练模型)。

### Roboflow

您可以在 Ultralytics HUB **数据集**页面上轻松筛选 Roboflow 数据集。

![Ultralytics HUB 数据集页面截图，显示 Roboflow 提供商筛选器](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-page-roboflow-filter.avif)

Ultralytics HUB 支持两种与 Roboflow 的集成类型：**Universe** 和 **Workspace**。

#### Universe

Roboflow Universe 集成允许您一次从 Roboflow 导入一个[数据集](https://www.ultralytics.com/glossary/benchmark-dataset)到 Ultralytics HUB。

##### 导入

导出 Roboflow 数据集时，选择 Ultralytics HUB 格式。此操作会将您重定向到 Ultralytics HUB 并打开**数据集导入**对话框。

点击**导入**按钮导入您的 Roboflow 数据集。

![Ultralytics HUB 数据集导入对话框截图，箭头指向导入按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-import-dialog.avif)

接下来，您可以在新导入的数据集上训练模型。

![Ultralytics HUB Roboflow Universe 数据集页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-universe-import-2.avif)

##### 移除

导航到要移除的 Roboflow 数据集的数据集页面。打开数据集操作下拉菜单，点击**移除**选项。

![Ultralytics HUB Roboflow Universe 数据集页面截图，箭头指向移除选项](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-universe-remove.avif)

??? tip "提示"

    您也可以直接从主**数据集**页面移除导入的 Roboflow 数据集。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个 Roboflow Universe 数据集的移除选项](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-remove-option.avif)

#### Workspace

Roboflow Workspace 集成允许您一次将整个 Roboflow Workspace 导入 Ultralytics HUB。

##### 导入

点击侧边栏中的**集成**按钮导航到**集成**页面。

输入您的 Roboflow Workspace 私有 [API 密钥](https://en.wikipedia.org/wiki/API_key)，然后点击**添加**按钮。

??? tip "提示"

    点击**获取我的 API 密钥**按钮将重定向您到 Roboflow Workspace 设置，您可以在那里找到您的私有 API 密钥。

![Ultralytics HUB 集成页面截图，箭头指向侧边栏中的集成按钮和添加按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-integrations-page.avif)

这会将您的 Ultralytics HUB 账户与 Roboflow Workspace 连接，使您的 Roboflow 数据集在 Ultralytics HUB 中可用。

![Ultralytics HUB 集成页面截图，箭头指向其中一个已连接的工作区](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-import-2.avif)

接下来，您可以使用已连接工作区中的任何数据集训练模型。

![Ultralytics HUB Roboflow Workspace 数据集页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-train-model.avif)

##### 移除

通过侧边栏导航到**集成**页面。点击要断开连接的 Roboflow Workspace 的**取消链接**按钮。

![Ultralytics HUB 集成页面截图，箭头指向侧边栏中的集成按钮和其中一个已连接工作区的取消链接按钮](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-remove-1.avif)

??? tip "提示"

    您也可以直接从属于该工作区的任何数据集的数据集页面取消链接已连接的 Roboflow Workspace。

    ![Ultralytics HUB Roboflow Workspace 数据集页面截图，箭头指向移除选项](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-remove-2.avif)

??? tip "提示"

    或者，使用与该工作区中任何数据集关联的移除选项，直接从主**数据集**页面移除已连接的 Roboflow Workspace。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个 Roboflow Workspace 数据集的移除选项](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-remove-option.avif)

## 模型

### 导出

训练模型后，您可以使用[导出模式](../modes/export.md)将其[导出](./models.md#部署模型)为 13 种不同格式，包括流行的 [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange)、[OpenVINO](../integrations/openvino.md)、[CoreML](../integrations/coreml.md)、[TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 [PaddlePaddle](../integrations/paddlepaddle.md)。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向导出卡片和所有已导出的格式](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-deploy-export-formats.avif)

可用的导出格式详见下表。

{% include "macros/export-table.md" %}

## 即将推出的激动人心的新功能 🎉

我们正在不断努力扩展 Ultralytics HUB 的集成功能。即将推出的功能包括：

- 更多[数据集集成](../datasets/index.md)
- 详细的导出集成指南
- 每个集成的分步[教程](../guides/index.md)

## 保持更新 🚧

此页面是您获取最新集成更新和功能发布的首选资源。通过以下方式保持联系：

- **新闻通讯**：订阅[我们的 Ultralytics 新闻通讯](https://www.ultralytics.com/#newsletter)获取公告、发布和早期访问更新。
- **社交媒体**：关注 [Ultralytics LinkedIn](https://www.linkedin.com/company/ultralytics) 获取幕后内容、产品新闻和社区亮点。
- **博客**：深入阅读 [Ultralytics AI 博客](https://www.ultralytics.com/blog)获取深度文章、教程和用例聚焦。

## 我们重视您的意见 🗣️

通过我们的[官方联系表单](https://www.ultralytics.com/contact)分享您的想法、反馈和集成请求，帮助塑造 Ultralytics HUB 的未来。

## 感谢社区！🌍

您的[贡献](../help/contributing.md)和持续支持推动我们致力于突破 [AI 创新](https://github.com/ultralytics/ultralytics)的边界。敬请期待——激动人心的事情即将到来！

---

对即将到来的内容感到兴奋？收藏此页面并查看我们的[快速入门指南](https://docs.ultralytics.com/quickstart/)，在等待期间开始使用我们当前的工具。准备好与 Ultralytics 一起踏上变革性的 AI 和 ML 之旅！🛠️🤖
