---
comments: true
description: 使用 Paperspace Gradient 的一体化 MLOps 平台简化 YOLO11 训练。访问 GPU、自动化工作流程并轻松部署。
keywords: YOLO11, Paperspace Gradient, MLOps, 机器学习, 训练, GPU, Jupyter notebooks, 模型部署, AI, 云平台
---

# 使用 Paperspace Gradient 简化 YOLO11 模型训练

训练像 [YOLO11](../models/yolo11.md) 这样的计算机视觉模型可能很复杂。它涉及管理大型数据集、使用不同类型的计算机硬件（如 GPU、TPU 和 CPU），以及确保训练过程中数据流畅通。通常，开发人员最终会花费大量时间管理他们的计算机系统和环境。当你只想专注于构建最佳模型时，这可能会令人沮丧。

这就是像 Paperspace Gradient 这样的平台可以简化事情的地方。Paperspace Gradient 是一个 MLOps 平台，让你可以在一个地方构建、训练和部署[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型。使用 Gradient，开发人员可以专注于训练他们的 YOLO11 模型，而无需管理基础设施和环境的麻烦。

## Paperspace

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/paperspace-overview.avif" alt="Paperspace 概览">
</p>

[Paperspace](https://www.paperspace.com/) 由密歇根大学毕业生于 2014 年创立，并于 2023 年被 DigitalOcean 收购，是一个专为机器学习设计的云平台。它为用户提供强大的 GPU、协作式 Jupyter notebooks、用于部署的容器服务、机器学习任务的自动化工作流程以及高性能虚拟机。这些功能旨在简化从编码到部署的整个机器学习开发过程。

## Paperspace Gradient

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/paperspace-gradient-overview.avif" alt="Paperspace Gradient 概览">
</p>

Paperspace Gradient 是一套工具，旨在使在云中使用 AI 和机器学习变得更快更容易。Gradient 涵盖了整个[机器学习生命周期](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations)，从构建和训练模型到部署它们。

在其工具包中，它包括通过作业运行器支持 Google 的 TPU、对 Jupyter notebooks 和容器的全面支持，以及新的编程语言集成。它对语言集成的关注特别突出，允许用户轻松地将现有的 Python 项目适配为使用最先进的 GPU 基础设施。

## 使用 Paperspace Gradient 训练 YOLO11

Paperspace Gradient 使只需点击几下即可训练 YOLO11 模型成为可能。得益于集成，你可以访问 [Paperspace 控制台](https://console.paperspace.com/github/ultralytics/ultralytics)并立即开始训练模型。有关模型训练过程和最佳实践的详细理解，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

登录后点击下图所示的"Start Machine"按钮。几秒钟后，托管的 GPU 环境将启动，然后你可以运行 notebook 的单元格。

![使用 Paperspace Gradient 训练 YOLO11](https://github.com/ultralytics/docs/releases/download/0/start-machine-button.avif)

在 Glenn Jocher（Ultralytics 创始人）和 Paperspace 的 James Skelton 的讨论中探索更多 YOLO11 和 Paperspace Gradient 的功能。观看下面的讨论。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3HbbQHitN7g?si=DjuwrzMkW1WEoH5Y"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics 直播第 7 期：一切都与环境有关：使用 Gradient 优化 YOLO11 训练
</p>

## Paperspace Gradient 的关键功能

当你探索 [Paperspace 控制台](https://console.paperspace.com/github/ultralytics/ultralytics)时，你会看到机器学习工作流程的每个步骤是如何得到支持和增强的。以下是一些需要注意的事项：

- **一键 Notebooks**：Gradient 提供专门为 YOLO11 定制的预配置 [Jupyter Notebooks](../integrations/jupyterlab.md)，无需环境设置和依赖管理。只需选择所需的 notebook 并立即开始实验。

- **硬件灵活性**：从具有不同 CPU、GPU 和 TPU 配置的各种机器类型中选择，以满足你的训练需求和预算。Gradient 处理所有后端设置，让你专注于模型开发。

- **实验跟踪**：Gradient 自动跟踪你的实验，包括超参数、指标和代码更改。这允许你轻松比较不同的训练运行、识别最佳配置并重现成功的结果。

- **数据集管理**：直接在 Gradient 中高效管理你的数据集。轻松上传、版本控制和预处理数据，简化项目的数据准备阶段。

- **模型服务**：只需点击几下即可将训练好的 YOLO11 模型部署为 REST API。Gradient 处理基础设施，让你轻松地将[目标检测](https://www.ultralytics.com/glossary/object-detection)模型集成到你的应用中。

- **实时监控**：通过 Gradient 的直观仪表板监控已部署模型的性能和健康状况。深入了解推理速度、资源利用率和潜在错误。

## 为什么应该在 YOLO11 项目中使用 Gradient？

虽然有许多选项可用于训练、部署和评估 YOLO11 模型，但与 Paperspace Gradient 的集成提供了一组独特的优势，使其与其他解决方案区分开来。让我们探索是什么使这种集成独特：

- **增强的协作**：共享工作区和版本控制促进无缝团队合作并确保可重复性，允许你的团队有效地协同工作并维护项目的清晰历史记录。

- **低成本 GPU**：Gradient 以比主要云提供商或本地解决方案低得多的成本提供对高性能 GPU 的访问。通过按秒计费，你只需为实际使用的资源付费，优化你的预算。

- **可预测的成本**：Gradient 的按需定价确保成本透明和可预测。你可以根据需要扩展或缩减资源，只需为使用的时间付费，避免不必要的开支。

- **无承诺**：你可以随时调整实例类型以适应不断变化的项目需求并优化成本性能平衡。没有锁定期或承诺，提供最大的灵活性。

## 总结

本指南探索了用于训练 YOLO11 模型的 [Paperspace Gradient 集成](https://www.ultralytics.com/blog/ultralytics-x-paperspace-advancing-object-detection-capabilities-through-partnership)。Gradient 提供了工具和基础设施来加速你的 AI 开发之旅，从轻松的模型训练和评估到简化的部署选项。

有关进一步探索，请访问 [Paperspace 官方文档](https://docs.digitalocean.com/products/paperspace/)。

此外，请访问 [Ultralytics 集成指南页面](index.md)了解更多关于不同 YOLO11 集成的信息。它充满了见解和提示，可以将你的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目提升到新的水平。

## 常见问题

### 如何使用 Paperspace Gradient 训练 YOLO11 模型？

使用 Paperspace Gradient 训练 YOLO11 模型既简单又高效。首先，登录 [Paperspace 控制台](https://console.paperspace.com/github/ultralytics/ultralytics)。接下来，点击"Start Machine"按钮启动托管的 GPU 环境。环境准备好后，你可以运行 notebook 的单元格开始训练 YOLO11 模型。有关详细说明，请参阅我们的 [YOLO11 模型训练指南](../modes/train.md)。

### 使用 Paperspace Gradient 进行 YOLO11 项目有什么优势？

Paperspace Gradient 为训练和部署 YOLO11 模型提供了几个独特的优势：

- **硬件灵活性**：从各种 CPU、GPU 和 TPU 配置中选择。
- **一键 Notebooks**：使用为 YOLO11 预配置的 Jupyter Notebooks，无需担心环境设置。
- **实验跟踪**：自动跟踪超参数、指标和代码更改。
- **数据集管理**：直接在 Gradient 中高效管理你的数据集。
- **模型服务**：轻松将模型部署为 REST API。
- **实时监控**：通过仪表板监控模型性能和资源利用率。

### 为什么应该选择 Ultralytics YOLO11 而不是其他目标检测模型？

Ultralytics YOLO11 以其实时目标检测能力和高[准确率](https://www.ultralytics.com/glossary/accuracy)而脱颖而出。它与 Paperspace Gradient 等平台的无缝集成通过简化训练和部署过程来提高生产力。YOLO11 支持各种用例，从安全系统到零售库存管理。在我们的 [YOLO11 概览](https://www.ultralytics.com/yolo)中了解 YOLO11 的全部功能和优势。

### 我可以使用 Paperspace Gradient 在边缘设备上部署 YOLO11 模型吗？

是的，你可以使用 Paperspace Gradient 在边缘设备上部署 YOLO11 模型。该平台支持各种部署格式，如 [TFLite](../integrations/tflite.md) 和 [Edge TPU](../integrations/edge-tpu.md)，这些格式针对边缘设备进行了优化。在 Gradient 上训练模型后，请参阅我们的[导出指南](../modes/export.md)了解将模型转换为所需格式的说明。

### Paperspace Gradient 中的实验跟踪如何帮助改进 YOLO11 训练？

Paperspace Gradient 中的实验跟踪通过自动记录超参数、指标和代码更改来简化模型开发过程。这允许你轻松比较不同的训练运行、识别最佳配置并重现成功的实验。类似的功能可以在与 Ultralytics YOLO11 集成的其他[实验跟踪工具](../integrations/clearml.md)中找到。
