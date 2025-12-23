---
comments: true
description: 使用 Ultralytics HUB 项目优化您的模型管理。轻松创建、共享、编辑和比较模型，实现高效开发。
keywords: Ultralytics HUB, 模型管理, 创建项目, 共享项目, 编辑项目, 删除项目, 比较模型, 重新排序模型, 转移模型
---

# Ultralytics HUB 项目

[Ultralytics HUB](https://www.ultralytics.com/hub) 项目为整合和管理您的模型提供了有效的解决方案。如果您正在处理多个执行类似任务或具有相关目的的模型，[Ultralytics HUB](https://www.ultralytics.com/hub) 项目允许您将这些模型分组在一起。

这创建了一个统一且有组织的工作空间，便于更轻松地进行模型管理、比较和开发。将相似的模型或各种迭代放在一起可以促进快速基准测试，因为您可以比较它们的有效性。这可以导致更快、更有洞察力的[迭代开发](https://docs.ultralytics.com/guides/model-training-tips/)和模型优化。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics HUB 在 Tiger-Pose 数据集上训练 YOLOv8 姿态模型
</p>

## 创建项目

点击侧边栏中的**项目**按钮导航到[项目](https://hub.ultralytics.com/projects)页面，然后点击页面右上角的**创建项目**按钮。

![Ultralytics HUB 项目页面截图，箭头指向侧边栏中的项目按钮和创建项目按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-create-project-page.avif)

??? tip "提示"

    您可以直接从[主页](https://hub.ultralytics.com/home)创建项目。

    ![Ultralytics HUB 主页截图，箭头指向创建项目卡片](https://github.com/ultralytics/docs/releases/download/0/hub-create-project-card.avif)

此操作将触发**创建项目**对话框，打开一系列选项以根据您的需求定制项目。

在_项目名称_字段中输入项目名称或保留默认名称，然后一键完成项目创建。

您还可以选择为项目添加描述和独特图像，增强其在[项目](https://hub.ultralytics.com/projects)页面上的可识别性。

当您对项目配置满意时，点击**创建**。

![Ultralytics HUB 创建项目对话框截图，箭头指向创建按钮](https://github.com/ultralytics/docs/releases/download/0/hub-create-project-dialog.avif)

项目创建后，您可以从[项目](https://hub.ultralytics.com/projects)页面访问它。

![Ultralytics HUB 项目页面截图，箭头指向其中一个项目](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-projects-page.avif)

接下来，在您的项目中[训练模型](./models.md#训练模型)。

![Ultralytics HUB 项目页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-button.avif)

## 共享项目

!!! info "信息"

    [Ultralytics HUB](https://www.ultralytics.com/hub) 的共享功能提供了一种方便的方式与他人共享项目。此功能旨在同时满足现有 [Ultralytics HUB](https://www.ultralytics.com/hub) 用户和尚未创建账户的用户。

??? note "注意"

    您可以控制项目的通用访问权限。

    您可以选择将通用访问权限设置为"私有"，在这种情况下，只有您可以访问它。或者，您可以将通用访问权限设置为"未列出"，这将授予任何拥有项目直接链接的人查看权限，无论他们是否拥有 [Ultralytics HUB](https://www.ultralytics.com/hub) 账户。

导航到要共享的项目的项目页面，打开项目操作下拉菜单，点击**共享**选项。此操作将触发**共享项目**对话框。

![Ultralytics HUB 项目页面截图，箭头指向共享选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-project-dialog.avif)

??? tip "提示"

    您可以直接从[项目](https://hub.ultralytics.com/projects)页面共享项目。

    ![Ultralytics HUB 项目页面截图，箭头指向其中一个项目的共享选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-project-option.avif)

将通用访问权限设置为"未列出"，然后点击**保存**。

![Ultralytics HUB 共享项目对话框截图，箭头指向下拉菜单和保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-share-project-dialog.avif)

!!! warning "警告"

    更改项目的通用访问权限时，项目内模型的通用访问权限也会随之更改。

现在，任何拥有您项目直接链接的人都可以查看它。

??? tip "提示"

    您可以轻松点击**共享项目**对话框中显示的项目链接来复制它。

    ![Ultralytics HUB 共享项目对话框截图，箭头指向项目链接](https://github.com/ultralytics/docs/releases/download/0/hub-share-project-dialog-arrow.avif)

## 编辑项目

导航到要编辑的项目的项目页面，打开项目操作下拉菜单，点击**编辑**选项。此操作将触发**更新项目**对话框。

![Ultralytics HUB 项目页面截图，箭头指向编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-1.avif)

??? tip "提示"

    您可以直接从[项目](https://hub.ultralytics.com/projects)页面编辑项目。

    ![Ultralytics HUB 项目页面截图，箭头指向其中一个项目的编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-2.avif)

对项目应用所需的修改，然后点击**保存**确认更改。

![Ultralytics HUB 更新项目对话框截图，箭头指向保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-save-button.avif)

## 删除项目

导航到要删除的项目的项目页面，打开项目操作下拉菜单，点击**删除**选项。此操作将删除项目。

![Ultralytics HUB 项目页面截图，箭头指向删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-option.avif)

??? tip "提示"

    您可以直接从[项目](https://hub.ultralytics.com/projects)页面删除项目。

    ![Ultralytics HUB 项目页面截图，箭头指向其中一个项目的删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-option-1.avif)

!!! warning "警告"

    删除项目时，项目内的模型也会被删除。

!!! note "注意"

    如果您改变主意，可以从[回收站](https://hub.ultralytics.com/trash)页面恢复项目。

    ![Ultralytics HUB 回收站页面截图，箭头指向侧边栏中的回收站按钮和其中一个项目的恢复选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-restore-option.avif)

## 比较模型

导航到要比较的模型所在项目的项目页面。要使用模型比较功能，请点击**图表**选项卡。

![Ultralytics HUB 项目页面截图，箭头指向图表选项卡](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-1.avif)

这将显示所有相关图表。每个图表对应不同的指标，包含每个模型在该指标上的性能。模型用不同颜色表示，您可以将鼠标悬停在每个数据点上以获取更多信息。

![Ultralytics HUB 项目页面图表选项卡截图](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-charts-tab.avif)

??? tip "提示"

    每个图表都可以放大以获得更好的可视化效果。

    ![Ultralytics HUB 项目页面图表选项卡截图，箭头指向展开图标](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-expand-icon.avif)

    ![Ultralytics HUB 项目页面图表选项卡截图，其中一个图表已展开](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-expanded-chart.avif)

    此外，为了正确分析数据，您可以使用缩放功能。

    ![Ultralytics HUB 项目页面图表选项卡截图，其中一个图表已展开并缩放](https://github.com/ultralytics/docs/releases/download/0/hub-charts-tab-expanded-zoomed.avif)

??? tip "提示"

    您可以灵活地通过选择性隐藏某些模型来自定义视图。此功能允许您专注于感兴趣的模型。

    ![Ultralytics HUB 项目页面图表选项卡截图，箭头指向其中一个模型的隐藏/显示图标](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-hide-icon.avif)

## 重新排序模型

??? note "注意"

    Ultralytics HUB 的重新排序功能仅在您拥有的项目内有效。

导航到要重新排序的模型所在项目的项目页面。点击要移动的模型的指定重新排序图标，并将其拖动到所需位置。

![Ultralytics HUB 项目页面截图，箭头指向重新排序图标](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-reorder-models.avif)

## 转移模型

导航到要移动的模型所在项目的项目页面，打开项目操作下拉菜单，点击**转移**选项。此操作将触发**转移模型**对话框。

![Ultralytics HUB 项目页面截图，箭头指向其中一个模型的转移选项](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-1.avif)

??? tip "提示"

    您也可以直接从[模型](https://hub.ultralytics.com/models)页面转移模型。

    ![Ultralytics HUB 模型页面截图，箭头指向其中一个模型的转移选项](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-2.avif)

选择要将模型转移到的项目，然后点击**保存**。

![Ultralytics HUB 转移模型对话框截图，箭头指向下拉菜单和保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-dialog.avif)
