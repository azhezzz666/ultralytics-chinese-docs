---
comments: true
description: 概述 Ultralytics-Snippets Visual Studio Code 扩展如何帮助开发者加速使用 Ultralytics Python 包的工作。
keywords: Visual Studio Code, VS Code, 深度学习, 卷积神经网络, 计算机视觉, Python, 代码片段, Ultralytics, 开发者生产力, 机器学习, YOLO, 开发者, 生产力, 效率, 学习, 编程, IDE, 代码编辑器, 开发者工具, 编程工具
---

# Ultralytics VS Code 扩展

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EXIpyYVEjoI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics Visual Studio Code 扩展 | 即用型代码片段 | Ultralytics YOLO 🎉
</p>

## 功能和优势

✅ 您是使用 Ultralytics 构建计算机视觉应用程序的数据科学家或[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工程师吗？

✅ 您讨厌反复编写相同的代码块吗？

✅ 您总是忘记 [export](../modes/export.md)、[predict](../modes/predict.md)、[train](../modes/train.md)、[track](../modes/track.md) 或 [val](../modes/val.md) 方法的参数或默认值吗？

✅ 想要开始使用 Ultralytics 并希望有一种_更简单_的方式来引用或运行代码示例吗？

✅ 想要在使用 Ultralytics 时加速开发周期吗？

如果您使用 Visual Studio Code 并对上述任何问题回答"是"，那么 Ultralytics-snippets VS Code 扩展就是来帮助您的！继续阅读以了解更多关于该扩展、如何安装它以及如何使用它。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/snippet-prediction-preview.avif" alt="代码片段预测预览">
  <br>
  在 20 秒内使用 Ultralytics YOLO 运行示例代码！🚀
</p>

## 受 Ultralytics 社区启发

构建此扩展的灵感来自 Ultralytics 社区。社区围绕类似主题和示例的问题推动了该项目的开发。此外，Ultralytics 团队的许多成员使用 VS Code 来加速他们自己的工作 ⚡。

## 为什么选择 VS Code？

[Visual Studio Code](https://code.visualstudio.com/) 在全球开发者中非常受欢迎，在 Stack Overflow 开发者调查中连续多年（[2021](https://survey.stackoverflow.co/2021#section-most-popular-technologies-integrated-development-environment)、[2022](https://survey.stackoverflow.co/2022/#section-most-popular-technologies-integrated-development-environment)、[2023](https://survey.stackoverflow.co/2023/#section-most-popular-technologies-integrated-development-environment) 和 [2024](https://survey.stackoverflow.co/2024/technology#1-integrated-development-environment)）排名最受欢迎。由于 VS Code 的高度可定制性、内置功能、广泛兼容性和可扩展性，这么多开发者使用它并不奇怪。鉴于其在更广泛的开发者社区以及 Ultralytics [Discord](https://discord.com/invite/ultralytics)、[Discourse](https://community.ultralytics.com/)、[Reddit](https://www.reddit.com/r/ultralytics/) 和 [GitHub](https://github.com/ultralytics) 社区中的受欢迎程度，构建 VS Code 扩展来帮助简化您的工作流程并提高生产力是有意义的。

想让我们知道您使用什么来开发代码吗？前往我们的 Discourse [社区投票](https://community.ultralytics.com/t/what-do-you-use-to-write-code/89/1)告诉我们！在那里，也许可以查看我们最喜欢的计算机视觉、机器学习、AI 和开发者[表情包](https://community.ultralytics.com/c/off-topic/memes-jokes/11)，或者发布您最喜欢的！

## 安装扩展

!!! note

    任何允许安装 VS Code 扩展的代码环境_应该_与 Ultralytics-snippets 扩展兼容。发布扩展后，发现 [neovim](https://neovim.io/) 可以与 VS Code 扩展兼容。要了解更多信息，请参阅 [Ultralytics-Snippets 存储库](https://github.com/Burhan-Q/ultralytics-snippets)的 Readme 中的 [`neovim` 安装部分](https://github.com/Burhan-Q/ultralytics-snippets?tab=readme-ov-file#use-with-neovim)。

### 在 VS Code 中安装

1. 导航到 [VS Code 中的扩展菜单](https://code.visualstudio.com/docs/editor/extension-marketplace)或使用快捷键 <kbd>Ctrl</kbd>+<kbd>Shift ⇑</kbd>+<kbd>x</kbd>，然后搜索 Ultralytics-snippets。

2. 点击 <kbd>Install</kbd> 按钮。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vs-code-extension-menu.avif" alt="VS Code 扩展菜单">
  <br>
</p>

### 从 VS Code 扩展市场安装

1. 访问 [VS Code 扩展市场](https://marketplace.visualstudio.com/VSCode)并搜索 Ultralytics-snippets，或直接前往 [VS Code 市场上的扩展页面](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets)。

2. 点击 <kbd>Install</kbd> 按钮并允许浏览器启动 VS Code 会话。

3. 按照任何提示安装扩展。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vscode-marketplace-extension-install.avif" alt="VS Code 市场扩展安装">
  <br>
  <a href="https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets">Ultralytics-Snippets</a> 的 Visual Studio Code 扩展市场页面
</p>

## 使用 Ultralytics-Snippets 扩展

- 🧠 **智能代码补全**：使用针对 Ultralytics API 定制的高级代码补全建议，更快、更准确地编写代码。

- ⌛ **提高开发速度**：通过消除重复的编码任务并利用预构建的代码块片段来节省时间。

- 🔬 **提高代码质量**：通过智能代码补全编写更干净、更一致、无错误的代码。

- 💎 **简化工作流程**：通过自动化常见任务，专注于项目的核心逻辑。

### 概述

该扩展仅在[语言模式](https://code.visualstudio.com/docs/getstarted/tips-and-tricks#_change-language-mode)配置为 Python 🐍 时才会运行。这是为了避免在处理任何其他文件类型时插入代码片段。所有代码片段都有以 `ultra` 开头的前缀，在安装扩展后在编辑器中简单地输入 `ultra` 将显示可用代码片段的列表。您还可以使用 <kbd>Ctrl</kbd>+<kbd>Shift ⇑</kbd>+<kbd>p</kbd> 打开 VS Code [命令面板](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette)并运行命令 `Snippets: Insert Snippet`。

### 代码片段字段

许多代码片段具有带有默认占位符值或名称的"字段"。例如，[predict](../modes/predict.md) 方法的输出可以保存到名为 `r`、`results`、`detections`、`preds` 或开发者选择的任何其他名称的 Python 变量中，这就是代码片段包含"字段"的原因。在插入代码片段后使用键盘上的 <kbd>Tab ⇥</kbd>，您的光标将在字段之间快速移动。选择字段后，输入新的变量名将更改该实例，但也会更改代码片段代码中该变量的所有其他实例！

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/multi-update-field-and-options.avif" alt="多更新字段和选项">
  <br>
  插入代码片段后，将 <code>model</code> 重命名为 <code>world_model</code> 会更新所有实例。按 <kbd>Tab ⇥</kbd> 移动到下一个字段，这会打开一个下拉菜单并允许选择模型规模，移动到下一个字段提供另一个下拉菜单来选择 <code>world</code> 或 <code>worldv2</code> 模型变体。
</p>

### 代码片段补全

!!! tip "更_短_的快捷方式"

    **不**需要输入代码片段的完整前缀，甚至不需要从代码片段的开头开始输入。请参见下图中的示例。

代码片段以最具描述性的方式命名，但这意味着可能需要输入很多内容，如果目标是更_快_地移动，这将适得其反。幸运的是，VS Code 允许用户输入 `ultra.example-yolo-predict`、`example-yolo-predict`、`yolo-predict` 甚至 `ex-yolo-p` 仍然可以到达预期的代码片段选项！如果预期的代码片段_实际上_是 `ultra.example-yolo-predict-kwords`，那么只需使用键盘箭头 <kbd>↑</kbd> 或 <kbd>↓</kbd> 突出显示所需的代码片段并按 <kbd>Enter ↵</kbd> 或 <kbd>Tab ⇥</kbd> 将插入正确的代码块。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/incomplete-snippet-example.avif" alt="不完整代码片段示例">
  <br>
  输入 <code>ex-yolo-p</code> 仍然_会_到达正确的代码片段。
</p>

### 代码片段类别

这些是 Ultralytics-snippets 扩展当前可用的代码片段类别。未来将添加更多，因此请务必检查更新并为扩展启用自动更新。如果您觉得有任何遗漏，也可以[请求添加其他代码片段](#如何请求新的代码片段)。

| 类别      | 起始前缀         | 描述                                                                                                                                                                                                           |
| :-------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 示例      | `ultra.examples` | 帮助学习或开始使用 Ultralytics 的示例代码。示例是文档页面中代码的副本或类似内容。                                                                               |
| 关键字参数 | `ultra.kwargs`   | 通过添加包含所有关键字参数和默认值的 [train](../modes/train.md)、[track](../modes/track.md)、[predict](../modes/predict.md) 和 [val](../modes/val.md) 方法的代码片段来加速开发。 |
| 导入      | `ultra.imports`  | 快速导入常见 Ultralytics 对象的代码片段。                                                                                                                                                |
| 模型      | `ultra.yolo`     | 插入用于初始化各种[模型](../models/index.md)（`yolo`、`sam`、`rtdetr` 等）的代码块，包括下拉配置选项。                                                                   |
| 结果      | `ultra.result`   | [处理推理结果](../modes/predict.md#working-with-results)时常见操作的代码块。                                                                                                    |
| 工具      | `ultra.util`     | 提供对 Ultralytics 包中内置的常见工具的快速访问，在[简单工具页面](../usage/simple-utilities.md)上了解更多关于这些工具的信息。                                           |

### 通过示例学习

`ultra.examples` 代码片段对于任何想要学习如何开始使用 Ultralytics YOLO 基础知识的人都非常有用。示例代码片段旨在在插入后运行（有些还有下拉选项）。这方面的示例显示在本页[顶部](#ultralytics-vs-code-扩展)的动画中，在插入代码片段后，选择所有代码并使用 <kbd>Shift ⇑</kbd>+<kbd>Enter ↵</kbd> 交互式运行。

!!! example

    就像本页[顶部](#ultralytics-vs-code-扩展)的动画所示，您可以使用代码片段 `ultra.example-yolo-predict` 插入以下代码示例。插入后，唯一可配置的选项是模型规模，可以是：`n`、`s`、`m`、`l` 或 `x` 之一。

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolo11n.pt", task="detect")
    results = model(source=ASSETS / "bus.jpg")

    for result in results:
        print(result.boxes.data)
        # result.show()  # 取消注释以查看每个结果图像
    ```

### 加速开发

除 `ultra.examples` 之外的代码片段的目标是在使用 Ultralytics 时使开发更容易、更快。许多项目中使用的常见代码块是迭代使用模型 [predict](../modes/predict.md) 方法返回的 `Results` 列表。`ultra.result-loop` 代码片段可以帮助解决这个问题。

!!! example

    使用 `ultra.result-loop` 将插入以下默认代码（包括注释）。

    ```python
    # 参考 https://docs.ultralytics.com/modes/predict/#working-with-results

    for result in results:
        result.boxes.data  # torch.Tensor 数组
    ```

然而，由于 Ultralytics 支持众多[任务](../tasks/index.md)，在[处理推理结果](../modes/predict.md#working-with-results)时，您可能希望访问其他 `Results` 属性，这就是[代码片段字段](#代码片段字段)发挥作用的地方。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/results-loop-options.avif" alt="结果循环选项">
  <br>
  一旦跳转到 <code>boxes</code> 字段，会出现一个下拉菜单，允许根据需要选择另一个属性。
</p>

### 关键字参数

所有各种 Ultralytics [任务](../tasks/index.md)和[模式](../modes/index.md)有超过 💯 个关键字参数！这是很多需要记住的内容，很容易忘记参数是 `save_frame` 还是 `save_frames`（顺便说一下，肯定是 `save_frames`）。这就是 `ultra.kwargs` 代码片段可以帮助的地方！

!!! example

    要插入包含所有[推理参数](../modes/predict.md#inference-arguments)的 [predict](../modes/predict.md) 方法，请使用 `ultra.kwargs-predict`，它将插入以下代码（包括注释）。

    ```python
    model.predict(
        source=src,  # (str, 可选) 图像或视频的源目录
        imgsz=640,  # (int | list) 预测的输入图像尺寸，整数或列表 [w,h]
        conf=0.25,  # (float) 最小置信度阈值
        iou=0.7,  # (float) NMS 的交并比 (IoU) 阈值
        vid_stride=1,  # (int) 视频帧率步长
        stream_buffer=False,  # (bool) 在队列中缓冲传入帧 (True) 或仅保留最新帧 (False)
        visualize=False,  # (bool) 可视化模型特征
        augment=False,  # (bool) 对预测源应用图像增强
        agnostic_nms=False,  # (bool) 类别无关的 NMS
        classes=None,  # (int | list[int], 可选) 按类别过滤结果，即 classes=0 或 classes=[0,2,3]
        retina_masks=False,  # (bool) 使用高分辨率分割掩码
        embed=None,  # (list[int], 可选) 从给定层返回特征向量/嵌入
        show=False,  # (bool) 如果环境允许则显示预测的图像和视频
        save=True,  # (bool) 保存预测结果
        save_frames=False,  # (bool) 保存预测的单个视频帧
        save_txt=False,  # (bool) 将结果保存为 .txt 文件
        save_conf=False,  # (bool) 保存带有置信度分数的结果
        save_crop=False,  # (bool) 保存带有结果的裁剪图像
        stream=False,  # (bool) 通过返回生成器来处理长视频或大量图像，减少内存使用
        verbose=True,  # (bool) 在终端中启用/禁用详细推理日志
    )
    ```

    此代码片段具有所有关键字参数的字段，以及 `model` 和 `src` 的字段，以防您在代码中使用了不同的变量。在包含关键字参数的每一行上，都包含简要描述以供参考。

### 所有代码片段

了解可用代码片段的最佳方式是下载并安装扩展并尝试使用！如果您好奇并想事先查看列表，可以访问[存储库](https://github.com/Burhan-Q/ultralytics-snippets)或 [VS Code 市场上的扩展页面](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets)查看所有可用代码片段的表格。

## 结论

Ultralytics-Snippets VS Code 扩展旨在帮助数据科学家和机器学习工程师更高效地使用 Ultralytics YOLO 构建[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用程序。通过提供预构建的代码片段和有用的示例，我们帮助您专注于最重要的事情：创建创新解决方案。请通过访问 [VS Code 市场上的扩展页面](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets)并留下评论来分享您的反馈。⭐

## 常见问题

### 如何请求新的代码片段？

可以使用 Ultralytics-Snippets [存储库](https://github.com/Burhan-Q/ultralytics-snippets)上的 Issues 请求新的代码片段。

### Ultralytics 扩展的费用是多少？

它是 100% 免费的！

### 为什么我看不到代码片段预览？

VS Code 使用组合键 <kbd>Ctrl</kbd>+<kbd>Space</kbd> 在预览窗口中显示更多/更少信息。如果您在输入代码片段前缀时没有看到代码片段预览，使用此组合键应该可以恢复预览。

### 如何禁用 Ultralytics 中的扩展推荐？

如果您使用 VS Code 并开始看到提示您安装 Ultralytics-snippets 扩展的消息，并且不想再看到该消息，有两种方法可以禁用此消息。

1. 安装 Ultralytics-snippets，消息将不再显示 😆！

2. 您可以使用 `yolo settings vscode_msg False` 禁用消息显示，而无需安装扩展。如果您不熟悉，可以在[快速入门](../quickstart.md)页面上了解更多关于 [Ultralytics 设置](../quickstart.md#ultralytics-settings)的信息。

### 我有一个新的 Ultralytics 代码片段想法，如何添加？

访问 Ultralytics-snippets [存储库](https://github.com/Burhan-Q/ultralytics-snippets)并打开 Issue 或 Pull Request！

### 如何卸载 Ultralytics-Snippets 扩展？

与任何其他 VS Code 扩展一样，您可以通过导航到 VS Code 中的扩展菜单来卸载它。在菜单中找到 Ultralytics-snippets 扩展，点击齿轮图标 (⚙)，然后点击"Uninstall"以删除扩展。

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vscode-extension-menu.avif" alt="VS Code 扩展菜单">
  <br>
</p>
