---
comments: true
description: 学习训练计算机视觉模型的最佳实践，包括批量大小优化、混合精度训练、早停和优化器选择，以提高效率和准确性。
keywords: 模型训练机器学习, AI 模型训练, 轮次数量, 如何在机器学习中训练模型, 机器学习最佳实践, 什么是模型训练
---

# 机器学习最佳实践和模型训练技巧

## 简介

在进行[计算机视觉项目](./steps-of-a-cv-project.md)时，最重要的步骤之一是模型训练。在达到这一步之前，您需要[定义目标](./defining-project-goals.md)并[收集和标注数据](./data-collection-and-annotation.md)。在[预处理数据](./preprocessing_annotated_data.md)以确保其干净和一致之后，您可以继续训练模型。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GIrFEoR5PoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 模型训练技巧 | 如何处理大型数据集 | 批量大小、GPU 利用率和<a href="https://www.ultralytics.com/glossary/mixed-precision">混合精度</a>
</p>

那么，什么是[模型训练](../modes/train.md)？模型训练是教导模型识别视觉模式并根据数据进行预测的过程。它直接影响应用的性能和准确性。在本指南中，我们将介绍最佳实践、优化技术和故障排除技巧，帮助您有效地训练计算机视觉模型。

## 如何训练机器学习模型

计算机视觉模型通过调整其内部参数来最小化误差进行训练。最初，模型被输入大量标记的图像。它对这些图像中的内容进行预测，并将预测与实际标签或内容进行比较以计算误差。这些误差显示了模型的预测与真实值的偏差程度。

在训练期间，模型通过称为[反向传播](https://www.ultralytics.com/glossary/backpropagation)的过程迭代地进行预测、计算误差并更新其参数。在这个过程中，模型调整其内部参数（权重和偏置）以减少误差。通过多次重复这个循环，模型逐渐提高其准确性。随着时间的推移，它学会识别复杂的模式，如形状、颜色和纹理。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/backpropagation-diagram.avif" alt="什么是反向传播？">
</p>

这个学习过程使[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型能够执行各种[任务](../tasks/index.md)，包括[目标检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)和[图像分类](../tasks/classify.md)。最终目标是创建一个能够将其学习泛化到新的、未见过的图像的模型，以便它能够在真实世界应用中准确理解视觉数据。

现在我们知道了训练模型时幕后发生的事情，让我们看看训练模型时需要考虑的要点。

## 在大型数据集上训练

当您计划使用大型数据集训练模型时，有几个不同的方面需要考虑。例如，您可以调整批量大小、控制 GPU 利用率、选择使用多尺度训练等。让我们详细了解每个选项。

### 批量大小和 GPU 利用率

在大型数据集上训练模型时，有效利用 GPU 是关键。批量大小是一个重要因素。它是机器学习模型在单次训练迭代中处理的数据样本数量。使用 GPU 支持的最大批量大小，您可以充分利用其功能并减少模型训练所需的时间。但是，您要避免耗尽 GPU 内存。如果遇到内存错误，请逐步减少批量大小，直到模型顺利训练。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxl6Bbpcxs0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 Ultralytics YOLO11 中使用批量推理 | 在 Python 中加速目标检测 🎉
</p>

对于 YOLO11，您可以在[训练配置](../modes/train.md)中设置 `batch_size` 参数以匹配您的 GPU 容量。此外，在训练脚本中设置 `batch=-1` 将根据设备的功能自动确定可以高效处理的[批量大小](https://www.ultralytics.com/glossary/batch-size)。通过微调批量大小，您可以充分利用 GPU 资源并改善整体训练过程。

### 子集训练

子集训练是一种智能策略，涉及在代表较大数据集的较小数据集上训练模型。它可以节省时间和资源，特别是在初始模型开发和测试期间。如果您时间紧迫或正在尝试不同的模型配置，子集训练是一个好选择。

对于 YOLO11，您可以使用 `fraction` 参数轻松实现子集训练。此参数允许您指定用于训练的数据集比例。例如，设置 `fraction=0.1` 将使用 10% 的数据训练模型。您可以使用此技术进行快速迭代和调整模型，然后再承诺使用完整数据集训练模型。子集训练帮助您快速取得进展并及早发现潜在问题。

### 多尺度训练

多尺度训练是一种通过在不同大小的图像上训练来提高模型泛化能力的技术。您的模型可以学习在不同尺度和距离检测对象，变得更加鲁棒。

例如，当您训练 YOLO11 时，可以通过设置 `scale` 参数来启用多尺度训练。此参数按指定因子调整训练图像的大小，模拟不同距离的对象。例如，设置 `scale=0.5` 在训练期间随机将训练图像缩放 0.5 到 1.5 倍。配置此参数允许您的模型体验各种图像尺度，并提高其在不同对象大小和场景中的检测能力。

### 缓存

缓存是提高训练机器学习模型效率的重要技术。通过将预处理的图像存储在内存中，缓存减少了 GPU 等待从磁盘加载数据的时间。模型可以持续接收数据，而不会因磁盘 I/O 操作而延迟。

在训练 YOLO11 时，可以使用 `cache` 参数控制缓存：

- _`cache=True`_：将数据集图像存储在 RAM 中，提供最快的访问速度，但代价是增加内存使用。
- _`cache='disk'`_：将图像存储在磁盘上，比 RAM 慢但比每次加载新数据快。
- _`cache=False`_：禁用缓存，完全依赖磁盘 I/O，这是最慢的选项。

### 混合精度训练

混合精度训练同时使用 16 位（FP16）和 32 位（FP32）浮点类型。通过使用 FP16 进行更快的计算和 FP32 在需要时保持精度，利用了 FP16 和 FP32 的优势。大多数[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)的操作都在 FP16 中完成，以受益于更快的计算和更低的内存使用。但是，模型权重的主副本保存在 FP32 中，以确保权重更新步骤期间的准确性。您可以在相同的硬件限制内处理更大的模型或更大的批量大小。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/mixed-precision-training-overview.avif" alt="混合精度训练概述">
</p>

要实现混合精度训练，您需要修改训练脚本并确保您的硬件（如 GPU）支持它。许多现代[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架，如 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)，提供对混合精度的内置支持。

使用 YOLO11 时，混合精度训练非常简单。您可以在训练配置中使用 `amp` 标志。设置 `amp=True` 启用自动混合精度（AMP）训练。混合精度训练是优化模型训练过程的简单而有效的方法。

### 预训练权重

使用预训练权重是加速模型训练过程的明智方法。预训练权重来自已经在大型数据集上训练过的模型，为您的模型提供了一个良好的起点。[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)将预训练模型适应于新的相关任务。微调预训练模型涉及从这些权重开始，然后在您的特定数据集上继续训练。这种训练方法可以缩短训练时间，并且通常会获得更好的性能，因为模型从对基本特征的扎实理解开始。

`pretrained` 参数使 YOLO11 的迁移学习变得简单。设置 `pretrained=True` 将使用默认的预训练权重，或者您可以指定自定义预训练模型的路径。有效使用预训练权重和迁移学习可以提升模型的能力并降低训练成本。

### 其他处理大型数据集时需要考虑的技术

处理大型数据集时还有几种其他技术需要考虑：

- **[学习率](https://www.ultralytics.com/glossary/learning-rate)调度器**：实施学习率调度器可以在训练期间动态调整学习率。调整良好的学习率可以防止模型越过最小值并提高稳定性。训练 YOLO11 时，`lrf` 参数通过将最终学习率设置为初始学习率的一部分来帮助管理学习率调度。
- **分布式训练**：对于处理大型数据集，分布式训练可以改变游戏规则。您可以通过将训练工作负载分散到多个 GPU 或机器上来减少训练时间。这种方法对于具有大量计算资源的企业级项目特别有价值。

## 训练的轮次数量

训练模型时，[轮次](https://www.ultralytics.com/glossary/epoch)是指对整个训练数据集的一次完整遍历。在一个轮次期间，模型处理训练集中的每个示例一次，并根据学习算法更新其参数。通常需要多个轮次才能让模型随时间学习和优化其参数。

一个常见的问题是如何确定训练模型的轮次数量。一个好的起点是 300 个轮次。如果模型过早过拟合，您可以减少轮次数量。如果 300 个轮次后没有发生[过拟合](https://www.ultralytics.com/glossary/overfitting)，您可以将训练扩展到 600、1200 或更多轮次。

然而，理想的轮次数量可能因数据集大小和项目目标而异。较大的数据集可能需要更多轮次才能让模型有效学习，而较小的数据集可能需要较少的轮次以避免过拟合。对于 YOLO11，您可以在训练脚本中设置 `epochs` 参数。

## 早停

早停是优化模型训练的宝贵技术。通过监控验证性能，您可以在模型停止改进时停止训练。您可以节省计算资源并防止过拟合。

该过程涉及设置一个耐心参数，该参数确定在停止训练之前等待验证指标改进的轮次数。如果模型的性能在这些轮次内没有改进，则停止训练以避免浪费时间和资源。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/early-stopping-overview.avif" alt="早停概述">
</p>

对于 YOLO11，您可以通过在训练配置中设置 patience 参数来启用早停。例如，`patience=5` 意味着如果验证指标连续 5 个轮次没有改进，训练将停止。使用此方法可确保训练过程保持高效并实现最佳性能，而不会进行过多的计算。

## 在云端和本地训练之间选择

训练模型有两种选择：云端训练和本地训练。

云端训练提供可扩展性和强大的硬件，非常适合处理大型数据集和复杂模型。[Google Cloud](https://cloud.google.com/)、[AWS](https://aws.amazon.com/) 和 [Azure](https://azure.microsoft.com/) 等平台提供按需访问高性能 GPU 和 TPU，加快训练时间并支持更大模型的实验。然而，云端训练可能很昂贵，特别是长期使用，数据传输可能会增加成本和延迟。

本地训练提供更大的控制和定制，让您可以根据特定需求定制环境并避免持续的云端成本。对于长期项目，它可能更经济，而且由于您的数据保留在本地，它更安全。然而，本地硬件可能有资源限制并需要维护，这可能导致大型模型的训练时间更长。

## 选择优化器

优化器是一种算法，它调整神经网络的权重以最小化[损失函数](https://www.ultralytics.com/glossary/loss-function)，损失函数衡量模型的表现。简单来说，优化器通过调整参数来帮助模型学习以减少误差。选择正确的优化器直接影响模型学习的速度和准确性。

您还可以微调优化器参数以提高模型性能。调整学习率设置更新参数时的步长大小。为了稳定性，您可能从适中的学习率开始，并随时间逐渐降低以改善长期学习。此外，设置动量决定了过去更新对当前更新的影响程度。动量的常见值约为 0.9。它通常提供良好的平衡。

### 常见优化器

不同的优化器有各种优缺点。让我们看看几个常见的优化器。

- **SGD（随机梯度下降）**：
    - 使用损失函数相对于参数的梯度更新模型参数。
    - 简单高效，但收敛可能较慢，可能陷入局部最小值。

- **[Adam](https://www.ultralytics.com/glossary/adam-optimizer)（自适应矩估计）**：
    - 结合了带动量的 SGD 和 RMSProp 的优点。
    - 根据梯度的一阶和二阶矩估计调整每个参数的学习率。
    - 非常适合噪声数据和稀疏梯度。
    - 高效且通常需要较少的调整，是 YOLO11 的推荐优化器。

- **RMSProp（均方根传播）**：
    - 通过将梯度除以最近梯度幅度的运行平均值来调整每个参数的学习率。
    - 有助于处理梯度消失问题，对[循环神经网络](https://www.ultralytics.com/glossary/recurrent-neural-network-rnn)有效。

对于 YOLO11，`optimizer` 参数允许您从各种优化器中选择，包括 SGD、Adam、AdamW、NAdam、RAdam 和 RMSProp，或者您可以将其设置为 `auto` 以根据模型配置自动选择。

## 与社区联系

成为计算机视觉爱好者社区的一部分可以帮助您解决问题并更快地学习。以下是一些连接、获得帮助和分享想法的方式。

### 社区资源

- **GitHub Issues**：访问 [YOLO11 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)并使用 Issues 标签提问、报告错误和建议新功能。社区和维护者非常活跃，随时准备提供帮助。
- **Ultralytics Discord 服务器**：加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者聊天，获得支持并分享您的经验。

### 官方文档

- **Ultralytics YOLO11 文档**：查看[官方 YOLO11 文档](./index.md)获取各种计算机视觉项目的详细指南和有用技巧。

使用这些资源将帮助您解决挑战并了解计算机视觉社区的最新趋势和实践。

## 关键要点

训练计算机视觉模型涉及遵循良好实践、优化策略和解决出现的问题。调整批量大小、混合[精度](https://www.ultralytics.com/glossary/precision)训练和从预训练权重开始等技术可以使您的模型工作得更好、训练更快。子集训练和早停等方法帮助您节省时间和资源。与社区保持联系并跟上新趋势将帮助您不断提高模型训练技能。

## 常见问题

### 使用 Ultralytics YOLO 训练大型数据集时如何提高 GPU 利用率？

要提高 GPU 利用率，请将训练配置中的 `batch_size` 参数设置为 GPU 支持的最大大小。这确保您充分利用 GPU 的功能，减少训练时间。如果遇到内存错误，请逐步减少批量大小，直到训练顺利运行。对于 YOLO11，在训练脚本中设置 `batch=-1` 将自动确定高效处理的最佳批量大小。有关更多信息，请参阅[训练配置](../modes/train.md)。

### 什么是混合精度训练，如何在 YOLO11 中启用它？

混合精度训练同时使用 16 位（FP16）和 32 位（FP32）浮点类型来平衡计算速度和精度。这种方法加速训练并减少内存使用，而不会牺牲模型[准确性](https://www.ultralytics.com/glossary/accuracy)。要在 YOLO11 中启用混合精度训练，请在训练配置中将 `amp` 参数设置为 `True`。这将激活自动混合精度（AMP）训练。有关此优化技术的更多详情，请参阅[训练配置](../modes/train.md)。

### 多尺度训练如何增强 YOLO11 模型性能？

多尺度训练通过在不同大小的图像上训练来增强模型性能，使模型能够更好地泛化到不同的尺度和距离。在 YOLO11 中，您可以通过在训练配置中设置 `scale` 参数来启用多尺度训练。例如，`scale=0.5` 将图像大小减半，而 `scale=2.0` 将其加倍。这种技术模拟不同距离的对象，使模型在各种场景中更加鲁棒。有关设置和更多详情，请查看[训练配置](../modes/train.md)。

### 如何使用预训练权重加速 YOLO11 中的训练？

使用预训练权重可以通过利用已经熟悉基本视觉特征的模型来大大加速训练并提高模型准确性。在 YOLO11 中，只需在训练配置中将 `pretrained` 参数设置为 `True` 或提供自定义预训练权重的路径。这种称为迁移学习的方法允许在大型数据集上训练的模型有效地适应您的特定应用。在[训练配置指南](../modes/train.md)中了解更多关于如何使用预训练权重及其好处的信息。

### 训练模型的推荐轮次数量是多少，如何在 YOLO11 中设置？

轮次数量是指模型训练期间对训练数据集的完整遍历次数。典型的起点是 300 个轮次。如果您的模型过早过拟合，您可以减少数量。或者，如果没有观察到过拟合，您可以将训练扩展到 600、1200 或更多轮次。要在 YOLO11 中设置此参数，请在训练脚本中使用 `epochs` 参数。有关确定理想轮次数量的更多建议，请参阅[轮次数量](#训练的轮次数量)部分。
