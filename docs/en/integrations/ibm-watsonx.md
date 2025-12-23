---
comments: true
description: 深入了解我们关于使用 IBM Watson 训练 YOLO11 模型的详细集成指南。了解关键功能和模型训练的分步说明。
keywords: IBM Watsonx, IBM Watsonx AI, 什么是 Watson, IBM Watson 集成, IBM Watson 功能, YOLO11, Ultralytics, 模型训练, GPU, TPU, 云计算
---

# 使用 IBM Watsonx 训练 YOLO11 模型的分步指南

如今，可扩展的[计算机视觉解决方案](../guides/steps-of-a-cv-project.md)变得越来越普遍，并正在改变我们处理视觉数据的方式。一个很好的例子是 IBM Watsonx，这是一个先进的 AI 和数据平台，可简化 AI 模型的开发、部署和管理。它为整个 AI 生命周期提供完整的套件，并与 IBM Cloud 服务无缝集成。

您可以使用 IBM Watsonx 训练 [Ultralytics YOLO11 模型](https://github.com/ultralytics/ultralytics)。对于对高效[模型训练](../modes/train.md)、针对特定任务进行微调以及使用强大工具和用户友好设置改进[模型性能](../guides/model-evaluation-insights.md)感兴趣的企业来说，这是一个不错的选择。在本指南中，我们将引导您完成使用 IBM Watsonx 训练 YOLO11 的过程，涵盖从设置环境到评估训练模型的所有内容。让我们开始吧！

## 什么是 IBM Watsonx？

[Watsonx](https://www.ibm.com/products/watsonx) 是 IBM 的云端平台，专为商业[生成式 AI](https://www.ultralytics.com/glossary/generative-ai) 和科学数据设计。IBM Watsonx 的三个组件 - `watsonx.ai`、`watsonx.data` 和 `watsonx.governance` - 共同创建了一个端到端、可信赖的 AI 平台，可以加速旨在解决业务问题的 AI 项目。它提供了强大的工具来构建、训练和[部署机器学习模型](../guides/model-deployment-options.md)，并使连接各种数据源变得容易。

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/overview-of-ibm-watsonx.avif" alt="IBM Watsonx 概述">
</p>

其用户友好的界面和协作功能简化了开发过程，并有助于高效的模型管理和部署。无论是计算机视觉、预测分析、[自然语言处理](https://www.ultralytics.com/glossary/natural-language-processing-nlp)还是其他 AI 应用，IBM Watsonx 都提供推动创新所需的工具和支持。

## IBM Watsonx 的主要功能

IBM Watsonx 由三个主要组件组成：`watsonx.ai`、`watsonx.data` 和 `watsonx.governance`。每个组件都提供满足 AI 和数据管理不同方面的功能。让我们仔细看看它们。

### [Watsonx.ai](https://www.ibm.com/products/watsonx-ai)

Watsonx.ai 为 AI 开发提供强大的工具，并提供对 IBM 支持的自定义模型、第三方模型（如 [Llama 3](https://www.ultralytics.com/blog/getting-to-know-metas-llama-3)）和 IBM 自己的 Granite 模型的访问。它包括用于实验 AI 提示的 Prompt Lab、用于使用标记数据改进模型性能的 Tuning Studio，以及用于简化生成式 AI 应用程序开发的 Flows Engine。此外，它还提供用于自动化 AI 模型生命周期和连接各种 API 和库的综合工具。

### [Watsonx.data](https://www.ibm.com/products/watsonx-data)

Watsonx.data 通过 IBM Storage Fusion HCI 集成支持云和本地部署。其用户友好的控制台提供跨环境的集中数据访问，并使用通用 SQL 简化数据探索。它使用 Presto 和 Spark 等高效查询引擎优化工作负载，通过 AI 驱动的语义层加速数据洞察，包括用于 AI 相关性的向量数据库，并支持开放数据格式以便于共享分析和 AI 数据。

### [Watsonx.governance](https://www.ibm.com/products/watsonx-governance)

Watsonx.governance 通过自动识别监管变化和执行策略来简化合规性。它将需求与内部风险数据联系起来，并提供最新的 AI 事实表。该平台通过警报和工具帮助管理风险，以检测[偏差和漂移](../guides/model-monitoring-and-maintenance.md)等问题。它还自动化 AI 生命周期的监控和文档记录，使用模型清单组织 AI 开发，并通过用户友好的仪表板和报告工具增强协作。

## 如何使用 IBM Watsonx 训练 YOLO11

您可以使用 IBM Watsonx 加速您的 YOLO11 模型训练工作流程。

### 先决条件

您需要一个 [IBM Cloud 账户](https://cloud.ibm.com/registration)来创建 [watsonx.ai](https://www.ibm.com/products/watsonx-ai) 项目，您还需要一个 [Kaggle](./kaggle.md) 账户来加载数据集。

### 步骤 1：设置您的环境

首先，您需要设置一个 IBM 账户来使用 Jupyter Notebook。使用您的 IBM Cloud 账户登录 [watsonx.ai](https://eu-de.dataplatform.cloud.ibm.com/registration/stepone?preselect_region=true)。

然后，创建一个 [watsonx.ai 项目](https://www.ibm.com/docs/en/watsonx/saas?topic=projects-creating-project)和一个 [Jupyter Notebook](https://www.ibm.com/docs/en/watsonx/saas?topic=editor-creating-managing-notebooks)。

完成后，将打开一个笔记本环境供您加载数据集。您可以使用本教程中的代码来处理一个简单的目标检测模型训练任务。

### 步骤 2：安装和导入相关库

接下来，您可以安装和导入必要的 Python 库。

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装所需的包
        pip install torch torchvision torchaudio
        pip install opencv-contrib-python-headless
        pip install ultralytics==8.0.196
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在安装 YOLO11 所需的包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)以获取解决方案和提示。

然后，您可以导入所需的包。

!!! example "导入相关库"

    === "Python"

        ```python
        # 导入 ultralytics
        import ultralytics

        ultralytics.checks()

        # 导入用于检索和显示图像文件的包
        ```

### 步骤 3：加载数据

在本教程中，我们将使用 Kaggle 上提供的[海洋垃圾数据集](https://www.kaggle.com/datasets/atiqishrak/trash-dataset-icra19)。使用此数据集，我们将自定义训练 YOLO11 模型来检测和分类水下图像中的垃圾和生物对象。

我们可以使用 Kaggle API 将数据集直接加载到笔记本中。首先，创建一个免费的 Kaggle 账户。创建账户后，您需要生成一个 API 密钥。生成密钥的说明可以在 [Kaggle API 文档](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md)的"API 凭据"部分找到。

将您的 Kaggle 用户名和 API 密钥复制并粘贴到以下代码中。然后运行代码以安装 API 并将数据集加载到 Watsonx 中。

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 kaggle
        pip install kaggle
        ```

安装 Kaggle 后，我们可以将数据集加载到 Watsonx 中。

!!! example "加载数据"

    === "Python"

        ```python
        # 将 "username" 字符串替换为您的用户名
        os.environ["KAGGLE_USERNAME"] = "username"
        # 将 "apiKey" 字符串替换为您的密钥
        os.environ["KAGGLE_KEY"] = "apiKey"

        # 加载数据集
        os.system("kaggle datasets download atiqishrak/trash-dataset-icra19 --unzip")

        # 将工作目录路径存储为 work_dir
        work_dir = os.getcwd()

        # 打印 work_dir 路径
        print(os.getcwd())

        # 打印 work_dir 内容
        print(os.listdir(f"{work_dir}"))

        # 打印 trash_ICRA19 子目录内容
        print(os.listdir(f"{work_dir}/trash_ICRA19"))
        ```

加载数据集后，我们打印并保存了工作目录。我们还打印了工作目录的内容以确认"trash_ICRA19"数据集已正确加载。

如果您在目录内容中看到"trash_ICRA19"，则表示已成功加载。您应该看到三个文件/文件夹：一个 `config.yaml` 文件、一个 `videos_for_testing` 目录和一个 `dataset` 目录。我们将忽略 `videos_for_testing` 目录，因此可以随意删除它。

我们将使用 `config.yaml` 文件和 dataset 目录的内容来训练我们的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型。以下是我们海洋垃圾数据集中的示例图像。

<p align="center">
  <img width="400" src="https://github.com/ultralytics/docs/releases/download/0/marine-litter-bounding-box.avif" alt="带边界框的海洋垃圾">
</p>


### 步骤 4：预处理数据

幸运的是，海洋垃圾数据集中的所有标签已经格式化为 YOLO .txt 文件。但是，我们需要重新排列图像和标签目录的结构，以帮助我们的模型处理图像和标签。目前，我们加载的数据集目录遵循以下结构：

<p align="center">
  <img width="400" src="https://github.com/ultralytics/docs/releases/download/0/marine-litter-bounding-box-1.avif" alt="加载的数据集目录">
</p>

但是，YOLO 模型默认需要在 train/val/test 拆分中的子目录中分别存放图像和标签。我们需要将目录重新组织为以下结构：

<p align="center">
  <img width="400" src="https://github.com/ultralytics/docs/releases/download/0/yolo-directory-structure.avif" alt="YOLO 目录结构">
</p>

要重新组织数据集目录，我们可以运行以下脚本：

!!! example "预处理数据"

    === "Python"

        ```python
        # 重新组织目录的函数
        def organize_files(directory):
            for subdir in ["train", "test", "val"]:
                subdir_path = os.path.join(directory, subdir)
                if not os.path.exists(subdir_path):
                    continue

                images_dir = os.path.join(subdir_path, "images")
                labels_dir = os.path.join(subdir_path, "labels")

                # 如果不存在则创建图像和标签子目录
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                # 将图像和标签移动到各自的子目录
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".txt"):
                        shutil.move(os.path.join(subdir_path, filename), os.path.join(labels_dir, filename))
                    elif filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                        shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))
                    # 删除 .xml 文件
                    elif filename.endswith(".xml"):
                        os.remove(os.path.join(subdir_path, filename))


        if __name__ == "__main__":
            directory = f"{work_dir}/trash_ICRA19/dataset"
            organize_files(directory)
        ```

接下来，我们需要修改数据集的 .yaml 文件。这是我们将在 .yaml 文件中使用的设置。类别 ID 号从 0 开始：

```yaml
path: /path/to/dataset/directory # 数据集根目录
train: train/images # 训练图像子目录
val: train/images # 验证图像子目录
test: test/images # 测试图像子目录

# 类别
names:
    0: plastic
    1: bio
    2: rov
```

运行以下脚本删除 `config.yaml` 的当前内容，并将其替换为反映我们新数据集目录结构的配置。该脚本自动使用我们之前定义的 `work_dir` 变量，因此在执行前请确保它指向您的数据集，并保持 train、val 和 test 子目录定义不变。

!!! example "编辑 .yaml 文件"

    === "Python"

        ```python
        # 新 config.yaml 文件的内容
        def update_yaml_file(file_path):
            data = {
                "path": f"{work_dir}/trash_ICRA19/dataset",
                "train": "train/images",
                "val": "train/images",
                "test": "test/images",
                "names": {0: "plastic", 1: "bio", 2: "rov"},
            }

            # 确保 "names" 列表出现在子目录之后
            names_data = data.pop("names")
            with open(file_path, "w") as yaml_file:
                yaml.dump(data, yaml_file)
                yaml_file.write("\n")
                yaml.dump({"names": names_data}, yaml_file)


        if __name__ == "__main__":
            file_path = f"{work_dir}/trash_ICRA19/config.yaml"  # .yaml 文件路径
            update_yaml_file(file_path)
            print(f"{file_path} 更新成功。")
        ```

### 步骤 5：训练 YOLO11 模型

运行以下命令行代码来微调预训练的默认 YOLO11 模型。

!!! example "训练 YOLO11 模型"

    === "CLI"

        ```bash
        !yolo task=detect mode=train data={work_dir}/trash_ICRA19/config.yaml model=yolo11n.pt epochs=2 batch=32 lr0=.04 plots=True
        ```

以下是模型训练命令中参数的详细说明：

- **task**：指定您使用指定的 YOLO 模型和数据集进行的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务。
- **mode**：表示您加载指定模型和数据的目的。由于我们正在训练模型，因此设置为"train"。稍后，当我们测试模型性能时，我们将其设置为"predict"。
- **epochs**：这限定了 YOLO11 将通过我们整个数据集的次数。
- **batch**：数值规定了训练[批量大小](https://www.ultralytics.com/glossary/batch-size)。批量是模型在更新其参数之前处理的图像数量。
- **lr0**：指定模型的初始[学习率](https://www.ultralytics.com/glossary/learning-rate)。
- **plots**：指示 YOLO 生成并保存我们模型训练和评估指标的图表。

有关模型训练过程和最佳实践的详细了解，请参阅 [YOLO11 模型训练指南](../modes/train.md)。本指南将帮助您充分利用实验并确保您有效地使用 YOLO11。

### 步骤 6：测试模型

我们现在可以运行推理来测试微调模型的性能：

!!! example "测试 YOLO11 模型"

    === "CLI"

        ```bash
        !yolo task=detect mode=predict source={work_dir}/trash_ICRA19/dataset/test/images model={work_dir}/runs/detect/train/weights/best.pt conf=0.5 iou=.5 save=True save_txt=True
        ```

这个简短的脚本为测试集中的每张图像生成预测标签，以及将预测[边界框](https://www.ultralytics.com/glossary/bounding-box)叠加在原始图像上的新输出图像文件。

通过 `save_txt=True` 参数保存每张图像的预测 .txt 标签，通过 `save=True` 参数生成带有边界框叠加的输出图像。
参数 `conf=0.5` 通知模型忽略置信度低于 50% 的所有预测。

最后，`iou=.5` 指示模型忽略同一类别中重叠度为 50% 或更高的框。它有助于减少为同一对象生成的潜在重复框。
我们可以加载带有预测边界框叠加的图像，以查看我们的模型在少量图像上的表现。

!!! example "显示预测"

    === "Python"

        ```python
        # 显示前面预测任务中的前十张图像
        for pred_dir in glob.glob(f"{work_dir}/runs/detect/predict/*.jpg")[:10]:
            img = Image.open(pred_dir)
            display(img)
        ```

上面的代码显示了测试集中的十张图像及其预测的边界框，以及类名标签和置信度级别。

### 步骤 7：评估模型

我们可以为每个类别生成模型[精确度](https://www.ultralytics.com/glossary/precision)和召回率的可视化。这些可视化保存在主目录下的 train 文件夹中。精确度分数显示在 P_curve.png 中：

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/precision-confidence-curve.avif" alt="精确度-置信度曲线">
</p>

该图显示，随着模型对预测的置信度级别增加，精确度呈指数增长。但是，在两个[训练周期](https://www.ultralytics.com/glossary/epoch)后，模型精确度尚未在某个置信度级别趋于平稳。

[召回率](https://www.ultralytics.com/glossary/recall)图（R_curve.png）显示了相反的趋势：

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/recall-confidence-curve.avif" alt="召回率-置信度曲线">
</p>

与精确度不同，召回率朝相反方向移动，在低置信度实例中显示更高的召回率，在高置信度实例中显示更低的召回率。这是分类模型中精确度和召回率权衡的一个恰当例子。

### 步骤 8：计算[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)

您可以通过计算预测边界框和同一对象的真实边界框之间的 IoU 来衡量预测[准确率](https://www.ultralytics.com/glossary/accuracy)。查看 [IBM 关于训练 YOLO11 的教程](https://developer.ibm.com/tutorials/awb-train-yolo-object-detection-model-in-python/)了解更多详细信息。

## 总结

我们探索了 IBM Watsonx 的主要功能，以及如何使用 IBM Watsonx 训练 YOLO11 模型。我们还看到了 IBM Watsonx 如何通过用于模型构建、数据管理和合规性的高级工具增强您的 AI 工作流程。

有关使用的更多详细信息，请访问 [IBM Watsonx 官方文档](https://www.ibm.com/products/watsonx)。

另外，请务必查看 [Ultralytics 集成指南页面](./index.md)，了解更多关于不同令人兴奋的集成的信息。

## 常见问题

### 如何使用 IBM Watsonx 训练 YOLO11 模型？

要使用 IBM Watsonx 训练 YOLO11 模型，请按照以下步骤操作：

1. **设置您的环境**：创建一个 IBM Cloud 账户并设置一个 Watsonx.ai 项目。使用 Jupyter Notebook 作为您的编码环境。
2. **安装库**：安装必要的库，如 `torch`、`opencv` 和 `ultralytics`。
3. **加载数据**：使用 Kaggle API 将您的数据集加载到 Watsonx 中。
4. **预处理数据**：将数据集组织成所需的目录结构并更新 `.yaml` 配置文件。
5. **训练模型**：使用 YOLO 命令行界面使用特定参数（如 `epochs`、`batch size` 和 `learning rate`）训练您的模型。
6. **测试和评估**：运行推理来测试模型并使用精确度和召回率等指标评估其性能。

有关详细说明，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

### IBM Watsonx 用于 AI 模型训练的主要功能是什么？

IBM Watsonx 为 AI 模型训练提供了几个关键功能：

- **Watsonx.ai**：提供 AI 开发工具，包括访问 IBM 支持的自定义模型和第三方模型（如 Llama 3）。它包括 Prompt Lab、Tuning Studio 和 Flows Engine，用于全面的 AI 生命周期管理。
- **Watsonx.data**：支持云和本地部署，提供集中的数据访问、高效的查询引擎（如 Presto 和 Spark）以及 AI 驱动的语义层。
- **Watsonx.governance**：自动化合规性，通过警报管理风险，并提供用于检测偏差和漂移等问题的工具。它还包括用于协作的仪表板和报告工具。

有关更多信息，请访问 [IBM Watsonx 官方文档](https://www.ibm.com/products/watsonx)。

### 为什么应该使用 IBM Watsonx 训练 Ultralytics YOLO11 模型？

IBM Watsonx 是训练 Ultralytics YOLO11 模型的绝佳选择，因为它提供了一套全面的工具来简化 AI 生命周期。主要优势包括：

- **可扩展性**：使用 IBM Cloud 服务轻松扩展您的模型训练。
- **集成**：与各种数据源和 API 无缝集成。
- **用户友好的界面**：通过协作和直观的界面简化开发过程。
- **高级工具**：访问 Prompt Lab、Tuning Studio 和 Flows Engine 等强大工具以增强模型性能。

在我们的[集成指南](./index.md)中了解更多关于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 以及如何使用 IBM Watsonx 训练模型的信息。

### 如何为 IBM Watsonx 上的 YOLO11 训练预处理数据集？

要为 IBM Watsonx 上的 YOLO11 训练预处理数据集：

1. **组织目录**：确保您的数据集遵循 YOLO 目录结构，在 train/val/test 拆分中为图像和标签分别设置子目录。
2. **更新 .yaml 文件**：修改 `.yaml` 配置文件以反映新的目录结构和类名。
3. **运行预处理脚本**：使用 Python 脚本重新组织数据集并相应地更新 `.yaml` 文件。

以下是组织数据集的示例脚本：

```python
import os
import shutil


def organize_files(directory):
    for subdir in ["train", "test", "val"]:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue

        images_dir = os.path.join(subdir_path, "images")
        labels_dir = os.path.join(subdir_path, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for filename in os.listdir(subdir_path):
            if filename.endswith(".txt"):
                shutil.move(os.path.join(subdir_path, filename), os.path.join(labels_dir, filename))
            elif filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))


if __name__ == "__main__":
    directory = f"{work_dir}/trash_ICRA19/dataset"
    organize_files(directory)
```

有关更多详细信息，请参阅我们的[数据预处理指南](../guides/preprocessing_annotated_data.md)。

### 在 IBM Watsonx 上训练 YOLO11 模型的先决条件是什么？

在开始在 IBM Watsonx 上训练 YOLO11 模型之前，请确保您具备以下先决条件：

- **IBM Cloud 账户**：在 IBM Cloud 上创建一个账户以访问 Watsonx.ai。
- **Kaggle 账户**：要加载数据集，您需要一个 Kaggle 账户和一个 API 密钥。
- **Jupyter Notebook**：在 Watsonx.ai 中设置一个 Jupyter Notebook 环境用于编码和模型训练。

有关设置环境的更多信息，请访问我们的 [Ultralytics 安装指南](../quickstart.md)。
