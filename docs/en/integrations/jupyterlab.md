---
comments: true
description: 学习如何使用 JupyterLab 训练和实验 Ultralytics YOLO11 模型。了解关键功能、设置说明和常见问题解决方案。
keywords: JupyterLab, YOLO11, Ultralytics, 模型训练, 深度学习, 交互式编程, 数据科学, 机器学习, Jupyter Notebook, 模型开发
---

# 使用 JupyterLab 训练 YOLO11 模型指南

构建[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型可能很困难，特别是当你没有合适的工具或环境时。如果你面临这个问题，JupyterLab 可能是适合你的解决方案。JupyterLab 是一个用户友好的基于 Web 的平台，使编程更加灵活和交互式。你可以用它来处理大型数据集、创建复杂模型，甚至与他人协作，所有这些都在一个地方完成。

你可以使用 JupyterLab 来[处理项目](../guides/steps-of-a-cv-project.md)，这些项目与 [Ultralytics YOLO11 模型](https://github.com/ultralytics/ultralytics)相关。JupyterLab 是高效模型开发和实验的绝佳选择。它使你可以轻松地从计算机上开始实验和[训练 YOLO11 模型](../modes/train.md)。让我们深入了解 JupyterLab、其关键功能以及如何使用它来训练 YOLO11 模型。

## 什么是 JupyterLab？

JupyterLab 是一个开源的基于 Web 的平台，专为处理 Jupyter notebooks、代码和数据而设计。它是传统 Jupyter Notebook 界面的升级版，提供更通用和强大的用户体验。

JupyterLab 允许你在一个地方处理 notebooks、文本编辑器、终端和其他工具。其灵活的设计让你可以根据需要组织工作区，使执行数据分析、可视化和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)等任务变得更加容易。JupyterLab 还支持实时协作，非常适合研究和数据科学领域的团队项目。

## JupyterLab 的关键功能

以下是使 JupyterLab 成为模型开发和实验绝佳选择的一些关键功能：

- **一体化工作区**：JupyterLab 是满足所有数据科学需求的一站式商店。与经典的 Jupyter Notebook 不同，后者有单独的文本编辑、终端访问和 notebooks 界面，JupyterLab 将所有这些功能集成到一个统一的环境中。你可以直接在 JupyterLab 中查看和编辑各种文件格式，包括 JPEG、PDF 和 CSV。一体化工作区让你触手可及所需的一切，简化工作流程并节省时间。
- **灵活布局**：JupyterLab 的突出功能之一是其灵活的布局。你可以拖放和调整标签大小，创建个性化布局，帮助你更高效地工作。可折叠的左侧边栏使文件浏览器、运行中的内核和命令面板等基本标签触手可及。你可以同时打开多个窗口，让你能够多任务处理并更有效地管理项目。
- **交互式代码控制台**：JupyterLab 中的代码控制台提供了一个交互式空间来测试代码片段或函数。它们还可以作为 notebook 中计算的日志。为 notebook 创建新控制台并查看所有内核活动非常简单。当你在实验新想法或排除代码问题时，此功能特别有用。
- **Markdown 预览**：由于其同步预览功能，在 JupyterLab 中处理 Markdown 文件更加高效。当你编写或编辑 Markdown 文件时，可以实时看到格式化的输出。这使得仔细检查文档外观变得更容易，省去了在编辑和预览模式之间来回切换的麻烦。
- **从文本文件运行代码**：如果你正在共享包含代码的文本文件，JupyterLab 可以轻松地直接在平台内运行它。你可以高亮代码并按 Shift + Enter 执行它。这对于快速验证代码片段非常有用，有助于确保你共享的代码是功能性的且无错误的。

## 为什么应该在 YOLO11 项目中使用 JupyterLab？

有多个平台可用于开发和评估机器学习模型，那么是什么让 JupyterLab 脱颖而出呢？让我们探索 JupyterLab 为你的机器学习项目提供的一些独特方面：

- **轻松的单元格管理**：在 JupyterLab 中管理单元格非常简单。你可以简单地拖放单元格来重新排列它们，而不是使用繁琐的剪切粘贴方法。
- **跨 Notebook 单元格复制**：JupyterLab 使在不同 notebooks 之间复制单元格变得简单。你可以将单元格从一个 notebook 拖放到另一个。
- **轻松切换到经典 Notebook 视图**：对于那些怀念经典 Jupyter Notebook 界面的人，JupyterLab 提供了轻松切换回去的方式。只需将 URL 中的 `/lab` 替换为 `/tree` 即可返回熟悉的 notebook 视图。
- **多视图**：JupyterLab 支持同一 notebook 的多个视图，这对于长 notebooks 特别有用。你可以并排打开不同部分进行比较或探索，在一个视图中所做的任何更改都会反映在另一个视图中。
- **可自定义主题**：JupyterLab 包含内置的深色主题用于 notebook，非常适合深夜编程会话。文本编辑器和终端也有可用的主题，允许你自定义整个工作区的外观。

## 使用 JupyterLab 时的常见问题

在使用 JupyterLab 时，你可能会遇到一些常见问题。以下是一些帮助你顺利使用平台的提示：

- **管理内核**：内核至关重要，因为它们管理你在 JupyterLab 中编写的代码与其运行环境之间的连接。它们还可以在 notebooks 之间访问和共享数据。当你关闭 Jupyter Notebook 时，内核可能仍在运行，因为其他 notebooks 可能正在使用它。如果你想完全关闭内核，可以选择它，右键单击，然后从弹出菜单中选择"关闭内核"。
- **安装 Python 包**：有时，你可能需要服务器上未预装的额外 Python 包。你可以使用命令 `python -m pip install package-name` 轻松地将这些包安装到你的主目录或虚拟环境中。要查看所有已安装的包，使用 `python -m pip list`。
- **将 Flask/FastAPI API 部署到 Posit Connect**：你可以使用终端中的 [rsconnect-python](https://docs.posit.co/rsconnect-python/) 包将 Flask 和 FastAPI API 部署到 Posit Connect。这样做可以更轻松地将你的 Web 应用程序与 JupyterLab 集成并与他人共享。
- **安装 JupyterLab 扩展**：JupyterLab 支持各种扩展来增强功能。你可以根据需要安装和自定义这些扩展。有关详细说明，请参阅 [JupyterLab 扩展指南](https://jupyterlab.readthedocs.io/en/latest/user/extensions.html)了解更多信息。
- **使用多个 Python 版本**：如果你需要使用不同版本的 Python，可以使用配置了不同 Python 版本的 Jupyter 内核。

## 如何使用 JupyterLab 试用 YOLO11

JupyterLab 使实验 YOLO11 变得容易。要开始，请按照以下简单步骤操作。

### 步骤 1：安装 JupyterLab

首先，你需要安装 JupyterLab。打开终端并运行命令：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 JupyterLab 所需的包
        pip install jupyterlab
        ```

### 步骤 2：下载 YOLO11 教程 Notebook

接下来，从 Ultralytics GitHub 仓库下载 [tutorial.ipynb](https://github.com/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) 文件。将此文件保存到本地机器上的任何目录。

### 步骤 3：启动 JupyterLab

使用终端导航到保存 notebook 文件的目录。然后，运行以下命令启动 JupyterLab：

!!! example "用法"

    === "CLI"

        ```bash
        jupyter lab
        ```

运行此命令后，它将在默认 Web 浏览器中打开 JupyterLab，如下所示。

![显示 JupyterLab 如何在浏览器中打开的图像](https://github.com/ultralytics/docs/releases/download/0/jupyterlab-browser-launch.avif)

### 步骤 4：开始实验

在 JupyterLab 中，打开 tutorial.ipynb notebook。你现在可以开始运行单元格来探索和实验 YOLO11。

![显示在 JupyterLab 中打开的 YOLO11 Notebook 的图像](https://github.com/ultralytics/docs/releases/download/0/opened-yolov8-notebook-jupyterlab.avif)

JupyterLab 的交互式环境允许你修改代码、可视化输出并在一个地方记录你的发现。你可以尝试不同的配置并了解 YOLO11 的工作原理。

有关模型训练过程和最佳实践的详细理解，请参阅 [YOLO11 模型训练指南](../modes/train.md)。本指南将帮助你充分利用实验并确保你有效地使用 YOLO11。

## 继续学习 JupyterLab

如果你想了解更多关于 JupyterLab 的信息，这里有一些很好的资源可以帮助你入门：

- [**JupyterLab 文档**](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)：深入了解官方 JupyterLab 文档，探索其功能和能力。这是了解如何充分利用这个强大工具的好方法。
- [**使用 Binder 试用**](https://mybinder.org/v2/gh/jupyterlab/jupyterlab-demo/HEAD?urlpath=lab/tree/demo)：无需安装任何东西即可使用 Binder 实验 JupyterLab，它可以让你直接在浏览器中启动实时 JupyterLab 实例。这是立即开始实验的好方法。
- [**安装指南**](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)：有关在本地机器上安装 JupyterLab 的分步指南，请查看安装指南。
- [**使用 JupyterLab 训练 Ultralytics YOLO11**](https://www.ultralytics.com/blog/train-ultralytics-yolo11-using-the-jupyterlab-integration)：在这篇详细的博客文章中了解更多关于使用 JupyterLab 与 YOLO11 模型的实际应用。

## 总结

我们探索了 JupyterLab 如何成为实验 Ultralytics YOLO11 模型的强大工具。使用其灵活和交互式的环境，你可以轻松地在本地机器上设置 JupyterLab 并开始使用 YOLO11。JupyterLab 使[训练](../guides/model-training-tips.md)和[评估](../guides/model-testing.md)模型、可视化输出以及[记录发现](../guides/model-monitoring-and-maintenance.md)变得简单，所有这些都在一个地方完成。

与 [Google Colab](../integrations/google-colab.md) 等其他平台不同，JupyterLab 在本地机器上运行，让你对计算环境有更多控制，同时仍提供交互式 notebook 体验。这使得它对于需要一致访问开发环境而不依赖云资源的开发人员特别有价值。

有关更多详细信息，请访问 [JupyterLab 常见问题页面](https://jupyterlab.readthedocs.io/en/stable/getting_started/faq.html)。

对更多 YOLO11 集成感兴趣？查看 [Ultralytics 集成指南](./index.md)，探索用于机器学习项目的其他工具和功能。

## 常见问题

### 如何使用 JupyterLab 训练 YOLO11 模型？

要使用 JupyterLab 训练 YOLO11 模型：

1. 安装 JupyterLab 和 Ultralytics 包：

    ```bash
    pip install jupyterlab ultralytics
    ```

2. 启动 JupyterLab 并打开一个新 notebook。

3. 导入 YOLO 模型并加载预训练模型：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    ```

4. 在自定义数据集上训练模型：

    ```python
    results = model.train(data="path/to/your/data.yaml", epochs=100, imgsz=640)
    ```

5. 使用 JupyterLab 的内置绘图功能可视化训练结果：

    ```python
    import matplotlib

    from ultralytics.utils.plotting import plot_results

    matplotlib.use("inline")  # 或 'notebook' 用于交互式
    plot_results(results)
    ```

JupyterLab 的交互式环境允许你轻松修改参数、可视化结果并迭代模型训练过程。

### JupyterLab 的哪些关键功能使其适合 YOLO11 项目？

JupyterLab 提供了几个使其非常适合 YOLO11 项目的功能：

1. 交互式代码执行：实时测试和调试 YOLO11 代码片段。
2. 集成文件浏览器：轻松管理数据集、模型权重和配置文件。
3. 灵活布局：并排排列多个 notebooks、终端和输出窗口，实现高效工作流程。
4. 丰富的输出显示：内联可视化 YOLO11 检测结果、训练曲线和模型性能指标。
5. Markdown 支持：使用富文本和图像记录你的 YOLO11 实验和发现。
6. 扩展生态系统：通过版本控制、[远程计算](google-colab.md)等扩展增强功能。

这些功能允许在使用 YOLO11 模型时获得无缝的开发体验，从数据准备到[模型部署](https://www.ultralytics.com/glossary/model-deployment)。

### 如何使用 JupyterLab 优化 YOLO11 模型性能？

要在 JupyterLab 中优化 YOLO11 模型性能：

1. 使用 autobatch 功能确定最佳批量大小：

    ```python
    from ultralytics.utils.autobatch import autobatch

    optimal_batch_size = autobatch(model)
    ```

2. 使用 Ray Tune 等库实现[超参数调优](../guides/hyperparameter-tuning.md)：

    ```python
    from ultralytics.utils.tuner import run_ray_tune

    best_results = run_ray_tune(model, data="path/to/data.yaml")
    ```

3. 使用 JupyterLab 的绘图功能可视化和分析模型指标：

    ```python
    from ultralytics.utils.plotting import plot_results

    plot_results(results.results_dict)
    ```

4. 实验不同的模型架构和[导出格式](../modes/export.md)，为你的特定用例找到速度和[准确率](https://www.ultralytics.com/glossary/accuracy)的最佳平衡。

JupyterLab 的交互式环境允许快速迭代和实时反馈，使优化 YOLO11 模型更加高效。

### 使用 JupyterLab 和 YOLO11 时如何处理常见问题？

在使用 JupyterLab 和 YOLO11 时，你可能会遇到一些常见问题。以下是处理方法：

1. GPU 内存问题：
    - 使用 `torch.cuda.empty_cache()` 在运行之间清除 GPU 内存。
    - 调整[批量大小](https://www.ultralytics.com/glossary/batch-size)或图像大小以适应你的 GPU 内存。

2. 包冲突：
    - 为你的 YOLO11 项目创建单独的 conda 环境以避免冲突。
    - 在 notebook 单元格中使用 `!pip install package_name` 安装缺失的包。

3. 内核崩溃：
    - 重启内核并逐个运行单元格以识别有问题的代码。
    - 检查代码中的内存泄漏，特别是在处理大型数据集时。
