---
comments: true
description: 在 Ultralytics HUB 上轻松管理、上传和共享您的自定义数据集，实现无缝模型训练集成。立即简化您的工作流程！
keywords: Ultralytics HUB, 数据集, 自定义数据集, 数据集管理, 模型训练, 上传数据集, 共享数据集, 数据集工作流程
---

# Ultralytics HUB 数据集

[Ultralytics HUB](https://www.ultralytics.com/hub) 数据集是管理和利用自定义数据集的实用解决方案。

上传后，数据集可以立即用于模型训练。这种集成方法促进了从数据集管理到模型训练的无缝过渡，显著简化了整个过程。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/R42s2zFtNIY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>上传数据集到 Ultralytics HUB | 数据集上传功能完整演练
</p>

## 上传数据集

[Ultralytics HUB](https://www.ultralytics.com/hub) 数据集与 YOLOv5 和 YOLOv8 🚀 数据集相同。它们使用相同的结构和相同的标签格式，以保持一切简单。

在将数据集上传到 [Ultralytics HUB](https://www.ultralytics.com/hub) 之前，请确保**将数据集 YAML 文件放在数据集根目录中**，并且**数据集 YAML、目录和 ZIP 文件具有相同的名称**，如下例所示，然后压缩数据集目录。

例如，如果您的数据集名为 "coco8"，就像我们的 [COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) 示例数据集一样，那么您应该在 `coco8/` 目录中有一个 `coco8.yaml`，压缩后将创建 `coco8.zip`：

```bash
zip -r coco8.zip coco8
```

您可以下载我们的 [COCO8](https://github.com/ultralytics/hub/blob/main/example_datasets/coco8.zip) 示例数据集并解压缩，以准确了解如何构建您的数据集。

<p align="center">
  <img  src="https://github.com/ultralytics/docs/releases/download/0/coco8-dataset-structure.avif" alt="COCO8 数据集结构" width="80%">
</p>

数据集 YAML 与标准 YOLOv5 和 YOLOv8 YAML 格式相同。

!!! example "coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

压缩数据集后，您应该在上传到 [Ultralytics HUB](https://www.ultralytics.com/hub) 之前[验证它](https://docs.ultralytics.com/reference/hub/__init__/#ultralytics.hub.check_dataset)。[Ultralytics HUB](https://www.ultralytics.com/hub) 在上传后进行数据集验证检查，因此通过提前确保数据集格式正确且无错误，您可以避免因数据集被拒绝而造成的任何挫折。

```python
from ultralytics.hub import check_dataset

check_dataset("path/to/dataset.zip", task="detect")
```

数据集 ZIP 准备好后，通过点击侧边栏中的**数据集**按钮导航到[数据集](https://hub.ultralytics.com/datasets)页面，然后点击页面右上角的**上传数据集**按钮。

![Ultralytics HUB 数据集页面截图，箭头指向侧边栏中的数据集按钮和上传数据集按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-upload.avif)

??? tip "提示"

    您可以直接从[主页](https://hub.ultralytics.com/home)上传数据集。

    ![Ultralytics HUB 主页截图，箭头指向上传数据集卡片](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-upload-dataset-card.avif)

此操作将触发**上传数据集**对话框。

选择数据集的任务类型，并在_数据集 .zip 文件_字段中上传。

您还可以选择为 [Ultralytics HUB](https://www.ultralytics.com/hub) 数据集设置自定义名称和描述。

当您对数据集配置满意时，点击**上传**。

![Ultralytics HUB 上传数据集对话框截图，箭头指向数据集任务、数据集文件和上传按钮](https://github.com/ultralytics/docs/releases/download/0/hub-upload-dataset-dialog.avif)

数据集上传和处理完成后，您可以从[数据集](https://hub.ultralytics.com/datasets)页面访问它。

![Ultralytics HUB 数据集页面截图，箭头指向其中一个数据集](https://github.com/ultralytics/docs/releases/download/0/hub-datasets-page.avif)

您可以按分割（训练、验证、测试）分组查看数据集中的图像。

![Ultralytics HUB 数据集页面截图，箭头指向图像选项卡](https://github.com/ultralytics/docs/releases/download/0/hub-dataset-page-images-tab.avif)

??? tip "提示"

    每张图像都可以放大以获得更好的可视化效果。

    ![Ultralytics HUB 数据集页面图像选项卡截图，箭头指向展开图标](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-images-tab-expand-icon.avif)

    ![Ultralytics HUB 数据集页面图像选项卡截图，其中一张图像已展开](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-page-expanded-image.avif)

此外，您可以点击**概览**选项卡分析数据集。

![Ultralytics HUB 数据集页面截图，箭头指向概览选项卡](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-overview-tab.avif)

接下来，在您的数据集上[训练模型](./models.md#训练模型)。

![Ultralytics HUB 数据集页面截图，箭头指向训练模型按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-page-train-model-button.avif)

## 下载数据集

导航到要下载的数据集的数据集页面，打开数据集操作下拉菜单，点击**下载**选项。此操作将开始下载您的数据集。

![Ultralytics HUB 数据集页面截图，箭头指向下载选项](https://github.com/ultralytics/docs/releases/download/0/hub-download-dataset-1.avif)

??? tip "提示"

    您可以直接从[数据集](https://hub.ultralytics.com/datasets)页面下载数据集。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个数据集的下载选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-download-option.avif)

## 共享数据集

!!! info "信息"

    [Ultralytics HUB](https://www.ultralytics.com/hub) 的共享功能提供了一种方便的方式与他人共享数据集。此功能旨在同时满足现有 [Ultralytics HUB](https://www.ultralytics.com/hub) 用户和尚未创建账户的用户。

!!! note "注意"

    您可以控制数据集的通用访问权限。

    您可以选择将通用访问权限设置为"私有"，在这种情况下，只有您可以访问它。或者，您可以将通用访问权限设置为"未列出"，这将授予任何拥有数据集直接链接的人查看权限，无论他们是否拥有 [Ultralytics HUB](https://www.ultralytics.com/hub) 账户。

导航到要共享的数据集的数据集页面，打开数据集操作下拉菜单，点击**共享**选项。此操作将触发**共享数据集**对话框。

![Ultralytics HUB 数据集页面截图，箭头指向共享选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-dataset.avif)

??? tip "提示"

    您可以直接从[数据集](https://hub.ultralytics.com/datasets)页面共享数据集。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个数据集的共享选项](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-2.avif)

将通用访问权限设置为"未列出"，然后点击**保存**。

![Ultralytics HUB 共享数据集对话框截图，箭头指向下拉菜单和保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-dialog.avif)

现在，任何拥有您数据集直接链接的人都可以查看它。

??? tip "提示"

    您可以轻松点击**共享数据集**对话框中显示的数据集链接来复制它。

    ![Ultralytics HUB 共享数据集对话框截图，箭头指向数据集链接](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-link.avif)

## 编辑数据集

导航到要编辑的数据集的数据集页面，打开数据集操作下拉菜单，点击**编辑**选项。此操作将触发**更新数据集**对话框。

![Ultralytics HUB 数据集页面截图，箭头指向编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-1.avif)

??? tip "提示"

    您可以直接从[数据集](https://hub.ultralytics.com/datasets)页面编辑数据集。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个数据集的编辑选项](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-page.avif)

对数据集应用所需的修改，然后点击**保存**确认更改。

![Ultralytics HUB 更新数据集对话框截图，箭头指向保存按钮](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-save-button.avif)

## 删除数据集

导航到要删除的数据集的数据集页面，打开数据集操作下拉菜单，点击**删除**选项。此操作将删除数据集。

![Ultralytics HUB 数据集页面截图，箭头指向删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-dataset-option.avif)

??? tip "提示"

    您可以直接从[数据集](https://hub.ultralytics.com/datasets)页面删除数据集。

    ![Ultralytics HUB 数据集页面截图，箭头指向其中一个数据集的删除选项](https://github.com/ultralytics/docs/releases/download/0/hub-delete-dataset-page.avif)

!!! note "注意"

    如果您改变主意，可以从[回收站](https://hub.ultralytics.com/trash)页面恢复数据集。

    ![Ultralytics HUB 回收站页面截图，箭头指向侧边栏中的回收站按钮和其中一个数据集的恢复选项](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-trash-restore.avif)
