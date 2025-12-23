---
comments: true
description: 学习如何使用 Kaggle 训练 Ultralytics YOLO11 模型，享受免费的 GPU/TPU 资源。了解 Kaggle 的功能、优势和高效模型开发的最佳实践。
keywords: Kaggle, YOLO11, Ultralytics, 机器学习, 模型训练, GPU, TPU, 云计算, 数据科学, 计算机视觉
---

# 使用 Kaggle 训练 YOLO11 模型指南

如果你正在学习 AI 并从事[小型项目](../solutions/index.md)，你可能还没有访问强大计算资源的权限，而高端硬件可能相当昂贵。幸运的是，Kaggle 是 Google 旗下的平台，提供了一个很好的解决方案。Kaggle 提供免费的基于云的环境，你可以在其中访问 GPU 资源、处理大型数据集，并与多元化的数据科学家和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)爱好者社区协作。

Kaggle 是[训练](../guides/model-training-tips.md)和实验 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) 的绝佳选择。Kaggle Notebooks 使在项目中使用流行的机器学习库和框架变得容易。让我们探索 Kaggle 的主要功能，并学习如何在这个平台上训练 YOLO11 模型！

## 什么是 Kaggle？

Kaggle 是一个将来自世界各地的数据科学家聚集在一起的平台，用于协作、学习和竞争解决现实世界的数据科学问题。Kaggle 由 Anthony Goldbloom 和 Jeremy Howard 于 2010 年创立，并于 2017 年被 Google 收购，它使用户能够连接、发现和共享数据集、使用 GPU 驱动的 notebooks，并参与数据科学竞赛。该平台旨在通过提供强大的工具和资源，帮助经验丰富的专业人士和热心的学习者实现他们的目标。

截至 2022 年，Kaggle 拥有超过 [1000 万用户](https://www.kaggle.com/discussions/general/332147)，为开发和实验机器学习模型提供了丰富的环境。你不需要担心本地机器的规格或设置；只需一个 Kaggle 账户和一个 Web 浏览器就可以直接开始。

## 使用 Kaggle 训练 YOLO11

由于平台可以访问强大的 GPU，在 Kaggle 上训练 YOLO11 模型既简单又高效。

要开始，请访问 [Kaggle YOLO11 Notebook](https://www.kaggle.com/code/glennjocherultralytics/ultralytics-yolo11-notebook)。Kaggle 的环境预装了 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 等库，使设置过程变得轻松。

![什么是 Kaggle 与 YOLO11 的集成？](https://github.com/ultralytics/docs/releases/download/0/kaggle-integration-yolov8.avif)

登录 Kaggle 账户后，你可以点击复制和编辑代码的选项，在加速器设置下选择 GPU，然后运行 notebook 的单元格开始训练模型。有关模型训练过程和最佳实践的详细理解，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

![使用 Kaggle 进行带 GPU 的机器学习模型训练](https://github.com/ultralytics/docs/releases/download/0/using-kaggle-for-machine-learning-model-training-with-a-gpu.avif)

在[官方 YOLO11 Kaggle notebook 页面](https://www.kaggle.com/code/glennjocherultralytics/ultralytics-yolo11-notebook)上，如果你点击右上角的三个点，你会注意到更多选项会弹出。

![官方 YOLO11 Kaggle Notebook 页面选项概览](https://github.com/ultralytics/docs/releases/download/0/overview-options-yolov8-kaggle-notebook.avif)

这些选项包括：

- **查看版本**：浏览 notebook 的不同版本以查看随时间的变化，并在需要时恢复到以前的版本。
- **复制 API 命令**：获取 API 命令以编程方式与 notebook 交互，这对于自动化和集成到工作流程中很有用。
- **在 Google Notebooks 中打开**：在 Google 托管的 notebook 环境中打开 notebook。
- **在 Colab 中打开**：在 [Google Colab](./google-colab.md) 中启动 notebook 进行进一步编辑和执行。
- **关注评论**：订阅评论部分以获取更新并与社区互动。
- **下载代码**：将整个 notebook 下载为 Jupyter (.ipynb) 文件，用于离线使用或在本地环境中进行版本控制。
- **添加到收藏**：将 notebook 保存到 Kaggle 账户中的收藏中，以便于访问和组织。
- **书签**：为 notebook 添加书签以便将来快速访问。
- **嵌入 Notebook**：获取嵌入链接以将 notebook 包含在博客、网站或文档中。

### 使用 Kaggle 时的常见问题

在使用 Kaggle 时，你可能会遇到一些常见问题。以下是一些帮助你顺利使用平台的要点：

- **访问 GPU**：在 Kaggle notebooks 中，你可以随时激活 GPU，每周允许使用最多 30 小时。Kaggle 提供具有 16GB 内存的 NVIDIA Tesla P100 GPU，还提供使用 NVIDIA GPU T4 x2 的选项。强大的硬件加速你的机器学习任务，使模型训练和推理更快。
- **Kaggle 内核**：Kaggle 内核是免费的 Jupyter notebook 服务器，可以集成 GPU，允许你在云计算机上执行机器学习操作。你不必依赖自己计算机的 CPU，避免过载并释放本地资源。
- **Kaggle 数据集**：Kaggle 数据集可以免费下载。但是，检查每个数据集的许可证以了解任何使用限制很重要。某些数据集可能对学术出版物或商业用途有限制。你可以直接将数据集下载到 Kaggle notebook 或通过 [Kaggle API](https://www.kaggle.com/docs/api) 下载到其他任何地方。
- **保存和提交 Notebooks**：要在 Kaggle 上保存和提交 notebook，点击"保存版本"。这会保存 notebook 的当前状态。后台内核完成生成输出文件后，你可以从主 notebook 页面的输出标签访问它们。
- **协作**：Kaggle 支持协作，但多个用户不能同时编辑 notebook。Kaggle 上的协作是异步的，意味着用户可以在不同时间共享和处理同一个 notebook。
- **恢复到以前的版本**：如果你需要恢复到 notebook 的以前版本，打开 notebook 并点击右上角的三个垂直点以选择"查看版本"。找到你想恢复的版本，点击旁边的"..."菜单，然后选择"恢复到版本"。notebook 恢复后，点击"保存版本"以提交更改。

## Kaggle 的关键功能

接下来，让我们了解 Kaggle 提供的功能，这些功能使其成为数据科学和机器学习爱好者的优秀平台。以下是一些关键亮点：

- **数据集**：Kaggle 托管了大量关于各种主题的[数据集](https://docs.ultralytics.com/datasets/)集合。你可以轻松搜索并在项目中使用这些数据集，这对于训练和测试 YOLO11 模型特别方便。
- **竞赛**：Kaggle 以其令人兴奋的竞赛而闻名，允许数据科学家和机器学习爱好者解决现实世界的问题。竞争有助于提高你的技能、学习新技术并在社区中获得认可。
- **免费访问 TPU**：Kaggle 提供免费访问强大的 [TPU](https://www.ultralytics.com/glossary/tpu-tensor-processing-unit)，这对于训练复杂的机器学习模型至关重要。这意味着你可以加速处理并提升 YOLO11 项目的性能，而无需产生额外费用。
- **与 GitHub 集成**：Kaggle 允许你轻松连接 GitHub 仓库以上传 notebooks 并保存你的工作。这种集成使管理和访问文件变得方便。
- **社区和讨论**：Kaggle 拥有强大的数据科学家和机器学习从业者社区。讨论论坛和共享的 notebooks 是学习和故障排除的绝佳资源。你可以轻松找到帮助、分享知识并与他人协作。

## 为什么应该在 YOLO11 项目中使用 Kaggle？

有多个平台可用于训练和评估机器学习模型，那么是什么让 Kaggle 脱颖而出呢？让我们深入了解使用 Kaggle 进行机器学习项目的好处：

- **公开 Notebooks**：你可以将 Kaggle notebooks 设为公开，允许其他用户查看、投票、fork 和讨论你的工作。Kaggle 促进协作、反馈和想法分享，帮助你改进 YOLO11 模型。
- **全面的 Notebook 提交历史**：Kaggle 创建 notebook 提交的详细历史记录。这允许你随时间审查和跟踪更改，使理解项目演变和在需要时恢复到以前版本变得更容易。
- **控制台访问**：Kaggle 提供控制台，让你对环境有更多控制。此功能允许你直接从命令行执行各种任务，增强工作流程和生产力。
- **资源可用性**：Kaggle 上的每个 notebook 编辑会话都提供大量资源：CPU 和 GPU 会话 12 小时执行时间，TPU 会话 9 小时执行时间，以及 20 GB 自动保存的磁盘空间。
- **Notebook 调度**：Kaggle 允许你安排 notebooks 在特定时间运行。你可以自动化重复任务而无需手动干预，例如定期训练模型。

## 继续学习 Kaggle

如果你想了解更多关于 Kaggle 的信息，这里有一些很好的资源可以指导你：

- [**Kaggle Learn**](https://www.kaggle.com/learn)：在 Kaggle Learn 上发现各种免费的交互式教程。这些课程涵盖基本的数据科学主题，并提供实践经验帮助你掌握新技能。
- [**Kaggle 入门**](https://www.kaggle.com/code/alexisbcook/getting-started-with-kaggle)：这个全面的指南带你了解使用 Kaggle 的基础知识，从参加竞赛到创建你的第一个 notebook。对于新手来说是一个很好的起点。
- [**Kaggle Medium 页面**](https://medium.com/@kaggleteam)：在 Kaggle 的 Medium 页面上探索教程、更新和社区贡献。这是了解最新趋势和深入了解数据科学的绝佳来源。
- [**使用 Kaggle 集成训练 Ultralytics YOLO 模型**](https://www.ultralytics.com/blog/train-ultralytics-yolo-models-using-the-kaggle-integration)：这篇博客文章提供了关于如何专门为 Ultralytics YOLO 模型利用 Kaggle 的额外见解。

## 总结

我们已经看到 Kaggle 如何通过提供免费访问强大的 GPU 来提升你的 YOLO11 项目，使模型训练和评估高效。Kaggle 的平台用户友好，预装了库以便快速设置。Ultralytics YOLO11 和 Kaggle 之间的集成创建了一个无缝的环境，用于开发、训练和部署最先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型，而无需昂贵的硬件。

有关更多详细信息，请访问 [Kaggle 文档](https://www.kaggle.com/docs)。

对更多 YOLO11 集成感兴趣？查看 [Ultralytics 集成指南](https://docs.ultralytics.com/integrations/)，探索用于机器学习项目的其他工具和功能。

## 常见问题

### 如何在 Kaggle 上训练 YOLO11 模型？

在 Kaggle 上训练 YOLO11 模型非常简单。首先，访问 [Kaggle YOLO11 Notebook](https://www.kaggle.com/code/glennjocherultralytics/ultralytics-yolo11-notebook)。登录 Kaggle 账户，复制并编辑 notebook，在加速器设置下选择 GPU。运行 notebook 单元格开始训练。有关更详细的步骤，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

### 使用 Kaggle 进行 YOLO11 模型训练有什么好处？

Kaggle 为训练 YOLO11 模型提供了几个优势：

- **免费 GPU 访问**：每周最多使用 30 小时的强大 GPU，如 NVIDIA Tesla P100 或 T4 x2。
- **预装库**：TensorFlow 和 PyTorch 等库已预装，简化设置。
- **社区协作**：与庞大的数据科学家和机器学习爱好者社区互动。
- **版本控制**：轻松管理 notebooks 的不同版本，并在需要时恢复到以前的版本。

有关更多详细信息，请访问我们的 [Ultralytics 集成指南](https://docs.ultralytics.com/integrations/)。

### 使用 Kaggle 进行 YOLO11 时可能遇到哪些常见问题，如何解决？

常见问题包括：

- **访问 GPU**：确保在 notebook 设置中激活 GPU。Kaggle 每周允许最多 30 小时的 GPU 使用。
- **数据集许可证**：检查每个数据集的许可证以了解使用限制。
- **保存和提交 Notebooks**：点击"保存版本"以保存 notebook 的状态，并从输出标签访问输出文件。
- **协作**：Kaggle 支持异步协作；多个用户不能同时编辑 notebook。

有关更多故障排除提示，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)。

### 为什么应该选择 Kaggle 而不是 Google Colab 等其他平台来训练 YOLO11 模型？

Kaggle 提供了使其成为绝佳选择的独特功能：

- **公开 Notebooks**：与社区分享你的工作以获得反馈和协作。
- **免费访问 TPU**：无需额外费用即可使用强大的 TPU 加速训练。
- **全面的历史记录**：通过 notebook 提交的详细历史记录跟踪随时间的变化。
- **资源可用性**：每个 notebook 会话提供大量资源，包括 CPU 和 GPU 会话 12 小时执行时间。

有关与 Google Colab 的比较，请参阅我们的 [Google Colab 指南](./google-colab.md)。

### 如何恢复到 Kaggle notebook 的以前版本？

要恢复到以前的版本：

1. 打开 notebook 并点击右上角的三个垂直点。
2. 选择"查看版本"。
3. 找到你想恢复的版本，点击旁边的"..."菜单，然后选择"恢复到版本"。
4. 点击"保存版本"以提交更改。
