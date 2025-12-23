---
comments: true
description: 学习如何使用 ClearML 进行 YOLOv5 实验跟踪、数据版本控制、超参数优化和远程执行。
keywords: ClearML, YOLOv5, 机器学习, 实验跟踪, 数据版本控制, 超参数优化, 远程执行, ML 流水线
---

# ClearML 集成

<img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_dark.png#gh-light-mode-only" alt="Clear|ML"><img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_light.png#gh-dark-mode-only" alt="Clear|ML">

## 关于 ClearML

[ClearML](https://clear.ml/) 是一个[开源](https://github.com/clearml/clearml) MLOps 平台，旨在简化您的机器学习工作流程并节省时间 ⏱️。

🔨 在<b>实验管理器</b>中跟踪每次 YOLOv5 训练运行

🔧 使用集成的 ClearML <b>数据版本控制工具</b>对您的自定义[训练数据](https://www.ultralytics.com/glossary/training-data)进行版本控制并轻松访问

🔦 使用 ClearML Agent <b>远程训练和监控</b>您的 YOLOv5 训练运行

🔬 使用 ClearML <b>超参数优化</b>获得最佳 mAP

🔭 只需几个命令，使用 ClearML Serving 将您新训练的 <b>YOLOv5 模型转换为 API</b>

<br>
还有更多功能。您可以决定使用多少这些工具，可以只使用实验管理器，或者将它们全部串联成令人印象深刻的流水线！
<br>
<br>

![ClearML 标量仪表板](https://github.com/ultralytics/docs/releases/download/0/clearml-scalars-dashboard.avif)

<br>
<br>

## 🦾 设置环境

要跟踪您的实验和/或数据，ClearML 需要与服务器通信。您有两个选项来获取服务器：

可以免费注册 [ClearML 托管服务](https://clear.ml/)，或者您可以设置自己的 [ClearML 服务器](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server)。服务器也是开源的，所以即使您处理敏感数据，也应该没问题！

- 安装 `clearml` Python 包：

    ```bash
    pip install clearml
    ```

- 通过[创建凭据](https://app.clear.ml/settings/workspace-configuration)（转到右上角 Settings -> Workspace -> Create new credentials）将 ClearML SDK 连接到服务器，然后执行以下命令并按照说明操作：

    ```bash
    clearml-init
    ```

就是这样！您已完成设置 😎

<br>

## 🚀 使用 ClearML 训练 YOLOv5

要启用 ClearML 实验跟踪，只需按照前面所示安装 ClearML pip 包（如果您跳过了该步骤，请运行以下命令）。

```bash
pip install clearml
```

这将启用与 YOLOv5 训练脚本的集成。从现在开始，每次训练运行都将被 ClearML [实验管理器](https://docs.ultralytics.com/integrations/clearml/)捕获和存储。

如果您想更改 `project_name` 或 `task_name`，请使用 `train.py` 脚本的 `--project` 和 `--name` 参数，默认情况下项目将被称为 `YOLOv5`，任务为 `Training`。请注意：ClearML 使用 `/` 作为子项目的分隔符，因此在项目名称中使用 `/` 时要小心！

```bash
python train.py --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

或使用自定义项目和任务名称：

```bash
python train.py --project my_project --name my_training --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

这将捕获：

- 源代码 + 未提交的更改
- 已安装的包
- （超）参数
- 模型文件（使用 `--save-period n` 每 n 个轮次保存一个检查点）
- 控制台输出
- 标量（mAP_0.5、mAP_0.5:0.95、精确率、召回率、损失、学习率等）
- 一般信息，如机器详情、运行时间、创建日期等
- 所有生成的图表，如标签相关图和[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)
- 每个[轮次](https://www.ultralytics.com/glossary/epoch)带边界框的图像
- 每个轮次的马赛克图
- 每个轮次的验证图像

内容很多对吧？🤯 现在，我们可以在 ClearML UI 中可视化所有这些信息，以获得训练进度的概览。向表格视图添加自定义列（例如 mAP_0.5），以便您可以轻松按最佳性能模型排序。或者选择多个实验并直接比较它们！

我们还可以利用所有这些信息做更多事情，比如[超参数优化](https://www.ultralytics.com/glossary/hyperparameter-tuning)和远程执行，如果您想了解其工作原理，请继续阅读！

### 🔗 数据集版本管理

将数据与代码分开进行版本控制通常是个好主意，也便于获取最新版本。此仓库支持提供数据集版本 ID，如果数据不存在，它将确保获取数据。此外，此工作流程还将使用的数据集 ID 保存为任务参数的一部分，因此您始终可以确切知道哪个实验使用了哪些数据！

![ClearML 数据集界面](https://github.com/ultralytics/docs/releases/download/0/clearml-dataset-interface.avif)

### 准备您的数据集

YOLOv5 仓库通过使用包含数据集信息的 YAML 文件支持多种不同的数据集。默认情况下，数据集下载到相对于仓库根文件夹的 `../datasets` 文件夹。因此，如果您使用 YAML 中的链接或 yolov5 提供的脚本下载了 `coco128` 数据集，您将获得以下文件夹结构：

```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ LICENSE
        |_ README.txt
```

但这可以是您想要的任何数据集。随意使用您自己的数据集，只要保持此文件夹结构即可。

接下来，⚠️**将相应的 YAML 文件复制到数据集文件夹的根目录**⚠️。此 YAML 文件包含 ClearML 正确使用数据集所需的信息。当然，您也可以自己创建，只需遵循示例 YAML 的结构即可。

基本上我们需要以下键：`path`、`train`、`test`、`val`、`nc`、`names`。

```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ coco128.yaml  # <---- 在这里！
        |_ LICENSE
        |_ README.txt
```

### 上传您的数据集

要将此数据集作为版本化数据集导入 ClearML，请转到数据集根文件夹（例如，从 YOLOv5 仓库工作时为 `../datasets/coco128`）并运行以下命令：

```bash
cd ../datasets/coco128
clearml-data sync --project YOLOv5 --name coco128 --folder .
```

命令 `clearml-data sync` 实际上是一个简写命令。您也可以依次运行以下命令：

```bash
# 如果您想基于另一个数据集版本创建此版本，可以选择添加 --parent <parent_dataset_id>，
# 这样就不会上传重复文件！
clearml-data create --name coco128 --project YOLOv5
clearml-data add --files .
clearml-data close
```

### 使用 ClearML 数据集运行训练

现在您有了 ClearML 数据集，可以非常简单地使用它来训练自定义 YOLOv5 🚀 模型！

```bash
python train.py --img 640 --batch 16 --epochs 3 --data clearml://YOUR_DATASET_ID --weights yolov5s.pt --cache
```

<br>

### 👀 超参数优化

现在我们已经对实验和数据进行了版本控制，是时候看看我们可以在此基础上构建什么了！

使用代码信息、已安装的包和环境详情，实验本身现在是**完全可重现的**。事实上，ClearML 允许您克隆实验甚至更改其参数。然后我们可以使用这些新参数自动重新运行它，这基本上就是 HPO 所做的！

要**在本地运行超参数优化**，我们为您提供了一个预制脚本。只需确保训练任务至少运行过一次，以便它在 ClearML 实验管理器中，我们将基本上克隆它并更改其超参数。

您需要在 `utils/loggers/clearml/hpo.py` 中的脚本中填写此 `template task` 的 ID，然后运行它。您可以将 `task.execute_locally()` 更改为 `task.execute()` 以将其放入 ClearML 队列，让远程代理处理它。

```bash
# 要使用 optuna，请先安装它，否则您可以将优化器更改为 RandomSearch
pip install optuna
python utils/loggers/clearml/hpo.py
```

![HPO](https://github.com/ultralytics/docs/releases/download/0/hpo-clearml-experiment.avif)

## 🤯 远程执行（高级）

在本地运行 HPO 非常方便，但如果我们想在远程机器上运行实验呢？也许您可以访问现场非常强大的 GPU 机器，或者您有一些预算使用云 GPU。这就是 [ClearML Agent](https://clear.ml/docs/latest/docs/clearml_agent) 发挥作用的地方。查看代理可以做什么：

- [YouTube 视频](https://youtu.be/MX3BrXnaULs)
- [文档](https://clear.ml/docs/latest/docs/clearml_agent)

简而言之：实验管理器跟踪的每个实验都包含足够的信息，可以在不同的机器上重现它（已安装的包、未提交的更改等）。因此，ClearML 代理就是这样做的：它监听队列中的传入任务，当找到一个任务时，它会重新创建环境并运行它，同时仍然向实验管理器报告标量、图表等。

您可以通过简单运行以下命令将任何机器（云虚拟机、本地 GPU 机器、您自己的笔记本电脑...）变成 ClearML 代理：

```bash
clearml-agent daemon --queue QUEUES_TO_LISTEN_TO [--docker]
```

### 克隆、编辑和入队

代理运行后，我们可以给它一些工作。还记得 HPO 部分我们可以克隆任务并编辑超参数吗？我们也可以从界面执行此操作！

🪄 右键单击实验进行克隆

🎯 将超参数编辑为您希望的值

⏳ 右键单击任务将其入队到任何队列

![从 UI 入队任务](https://github.com/ultralytics/docs/releases/download/0/enqueue-task-ui.avif)

### 远程执行任务

现在您可以像上面解释的那样克隆任务，或者只需通过添加 `task.execute_remotely()` 标记您当前的脚本，执行时它将被放入队列，供代理开始处理！

要远程运行 YOLOv5 训练脚本，您只需在 clearml 日志记录器实例化后将此行添加到 training.py 脚本中：

```python
# ...
# 日志记录器
data_dict = None
if RANK in {-1, 0}:
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 日志记录器实例
    if loggers.clearml:
        loggers.clearml.task.execute_remotely(queue="my_queue")  # <------ 添加此行
        # 如果用户没有选择 ClearML 数据集，data_dict 为 None，否则由 ClearML 填充
        data_dict = loggers.clearml.data_dict
# ...
```

在此更改后运行训练脚本时，Python 将运行脚本直到该行，之后它将打包代码并将其发送到队列！

### 自动扩展工作器

ClearML 还带有[自动扩展器](https://clear.ml/docs/latest/docs/guides/services/aws_autoscaler)！此工具将在您选择的云（AWS、GCP、Azure）中自动启动新的远程机器，并在检测到队列中有实验时将它们变成 ClearML 代理。任务处理完成后，自动扩展器将自动关闭远程机器，您就停止付费了！

查看下面的自动扩展器入门视频。

[![观看视频](https://github.com/ultralytics/docs/releases/download/0/clearml-autoscalers-video-thumbnail.avif)](https://youtu.be/j4XVMAaUt3E)

## 了解更多

有关将 ClearML 与 Ultralytics 模型集成的更多信息，请查看我们的 [ClearML 集成指南](https://docs.ultralytics.com/integrations/clearml/)，并探索如何使用其他实验跟踪工具增强您的 [MLOps 工作流程](https://www.ultralytics.com/blog/exploring-yolov8-ml-experiment-tracking-integrations)。
