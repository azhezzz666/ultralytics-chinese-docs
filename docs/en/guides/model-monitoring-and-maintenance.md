---
comments: true
description: 了解监控、维护和记录计算机视觉模型的关键实践，以保证准确性、发现异常并减轻数据漂移。
keywords: 计算机视觉模型, AI 模型监控, 数据漂移检测, AI 中的异常检测, 模型维护
---

# 部署后维护您的计算机视觉模型

## 简介

如果您在这里，我们可以假设您已经完成了[计算机视觉项目中的许多步骤](./steps-of-a-cv-project.md)：从[收集需求](./defining-project-goals.md)、[标注数据](./data-collection-and-annotation.md)和[训练模型](./model-training-tips.md)到最终[部署](./model-deployment-practices.md)它。您的应用现在正在生产中运行，但您的项目并没有在这里结束。计算机视觉项目最重要的部分是确保您的模型随着时间的推移继续满足您的[项目目标](./defining-project-goals.md)，这就是监控、维护和记录计算机视觉模型发挥作用的地方。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zCupPHqSLTI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 部署后如何维护计算机视觉模型 | 数据漂移检测
</p>

在本指南中，我们将仔细研究如何在部署后维护您的计算机视觉模型。我们将探讨模型监控如何帮助您及早发现问题，如何保持模型准确和最新，以及为什么文档对于故障排除很重要。

## 模型监控是关键

密切关注已部署的计算机视觉模型至关重要。如果没有适当的监控，模型可能会失去准确性。一个常见问题是数据分布偏移或[数据漂移](https://www.ultralytics.com/glossary/data-drift)，即模型遇到的数据与其训练数据发生变化。当模型必须对它不认识的数据进行预测时，可能会导致误解和性能不佳。异常值或不寻常的数据点也可能影响模型的准确性。

定期模型监控帮助开发者跟踪[模型的性能](./model-evaluation-insights.md)，发现异常，并快速解决数据漂移等问题。它还通过指示何时需要更新来帮助管理资源，避免昂贵的大修，并保持模型的相关性。

### 模型监控的最佳实践

以下是在生产中监控计算机视觉模型时需要记住的一些最佳实践：

- **定期跟踪性能**：持续监控模型的性能以检测随时间的变化。
- **仔细检查数据质量**：检查数据中的缺失值或异常。
- **使用多样化的数据源**：监控来自各种来源的数据，以全面了解模型的性能。
- **结合监控技术**：使用漂移检测算法和基于规则的方法的组合来识别各种问题。
- **监控输入和输出**：密切关注模型处理的数据和它产生的结果，以确保一切正常运行。
- **设置警报**：为异常行为（如性能下降）实施警报，以便能够快速采取纠正措施。

### AI 模型监控工具

您可以使用自动化监控工具来更轻松地在部署后监控模型。许多工具提供实时洞察和警报功能。以下是一些可以协同工作的开源模型监控工具示例：

- **[Prometheus](https://prometheus.io/)**：Prometheus 是一个开源监控工具，用于收集和存储指标以进行详细的性能跟踪。它可以轻松与 Kubernetes 和 Docker 集成，按设定的间隔收集数据并将其存储在时间序列数据库中。Prometheus 还可以抓取 HTTP 端点以收集实时指标。可以使用 PromQL 语言查询收集的数据。
- **[Grafana](https://grafana.com/)**：Grafana 是一个开源[数据可视化](https://www.ultralytics.com/glossary/data-visualization)和监控工具，允许您查询、可视化、警报和理解您的指标，无论它们存储在哪里。它与 Prometheus 配合良好，并提供高级数据可视化功能。您可以创建自定义仪表板来显示计算机视觉模型的重要指标，如推理延迟、错误率和资源使用情况。Grafana 将收集的数据转换为易于阅读的仪表板，包括折线图、热图和直方图。它还支持警报，可以通过 Slack 等渠道发送，以快速通知团队任何问题。
- **[Evidently AI](https://www.evidentlyai.com/)**：Evidently AI 是一个开源工具，专为监控和调试生产中的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型而设计。它从 pandas DataFrame 生成交互式报告，帮助分析机器学习模型。Evidently AI 可以检测数据漂移、模型性能下降以及已部署模型可能出现的其他问题。

上面介绍的三个工具，Evidently AI、Prometheus 和 Grafana，可以无缝协作，作为一个完全开源的 ML 监控解决方案，可用于生产。Evidently AI 用于收集和计算指标，Prometheus 存储这些指标，Grafana 显示它们并设置警报。虽然还有许多其他工具可用，但这种设置是一个令人兴奋的开源选项，为[模型监控](https://www.ultralytics.com/glossary/model-monitoring)和维护您的模型提供了强大的功能。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/evidently-prometheus-grafana-monitoring-tools.avif" alt="开源模型监控工具概述">
</p>

### 异常检测和警报系统

异常是指与预期有很大偏差的任何数据点或模式。对于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型，异常可能是与模型训练图像非常不同的图像。这些意外的图像可能是数据分布变化、异常值或可能降低模型性能的行为的迹象。设置警报系统来检测这些异常是模型监控的重要组成部分。

通过为关键指标设置标准性能水平和限制，您可以及早发现问题。当性能超出这些限制时，会触发警报，促使快速修复。定期使用新数据更新和重新训练模型可以使它们随着数据变化保持相关性和准确性。

#### 配置阈值和警报时需要注意的事项

在设置警报系统时，请记住以下最佳实践：

- **标准化警报**：对所有警报使用一致的工具和格式，如电子邮件或 Slack 等消息应用。标准化使您更容易快速理解和响应警报。
- **包含预期行为**：警报消息应清楚说明出了什么问题、预期是什么以及评估的时间范围。它帮助您评估警报的紧迫性和上下文。
- **可配置警报**：使警报易于配置以适应变化的条件。允许自己编辑阈值、暂停、禁用或确认警报。

### 数据漂移检测

数据漂移检测是一个概念，有助于识别输入数据的统计属性随时间变化的情况，这可能会降低模型性能。在您决定重新训练或调整模型之前，这种技术有助于发现存在问题。数据漂移处理的是数据整体格局随时间的变化，而[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)侧重于识别可能需要立即关注的罕见或意外数据点。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/data-drift-detection-overview.avif" alt="数据漂移检测概述">
</p>

以下是几种检测数据漂移的方法：

**持续监控**：定期监控模型的输入数据和输出以发现漂移迹象。跟踪关键指标并将其与历史数据进行比较以识别重大变化。

**统计技术**：使用 Kolmogorov-Smirnov 检验或群体稳定性指数（PSI）等方法来检测数据分布的变化。这些测试将新数据的分布与[训练数据](https://www.ultralytics.com/glossary/training-data)进行比较，以识别显著差异。

**特征漂移**：监控单个特征的漂移。有时，整体数据分布可能保持稳定，但单个特征可能会漂移。识别哪些特征正在漂移有助于微调重新训练过程。

## 模型维护

模型维护对于保持计算机视觉模型随时间准确和相关至关重要。模型维护涉及定期更新和重新训练模型、解决数据漂移，并确保模型随着数据和环境的变化保持相关性。您可能想知道模型维护与模型监控有何不同。监控是关于实时观察模型的性能以及早发现问题。另一方面，维护是关于修复这些问题。

### 定期更新和重新训练

一旦模型部署，在监控过程中，您可能会注意到数据模式或性能的变化，表明模型漂移。定期更新和重新训练成为模型维护的重要组成部分，以确保模型能够处理新的模式和场景。根据数据变化的方式，您可以使用几种技术。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/computer-vision-model-drift-overview.avif" alt="计算机视觉模型漂移概述">
</p>

例如，如果数据随时间逐渐变化，增量学习是一个好方法。增量学习涉及使用新数据更新模型，而无需从头开始完全重新训练，从而节省计算资源和时间。然而，如果数据发生了剧烈变化，定期完全重新训练可能是更好的选择，以确保模型不会在新数据上[过拟合](https://www.ultralytics.com/glossary/overfitting)，同时失去对旧模式的跟踪。

无论使用哪种方法，更新后的验证和测试都是必须的。在单独的[测试数据集](./model-testing.md)上验证模型以检查性能改进或下降非常重要。

### 决定何时重新训练模型

重新训练计算机视觉模型的频率取决于数据变化和模型性能。每当您观察到显著的性能下降或检测到数据漂移时，就重新训练模型。定期评估可以通过针对新数据测试模型来帮助确定正确的重新训练计划。监控性能指标和数据模式可以让您决定模型是否需要更频繁的更新以保持[准确性](https://www.ultralytics.com/glossary/accuracy)。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/when-to-retrain-overview.avif" alt="何时重新训练概述">
</p>

## 文档

记录计算机视觉项目使其更容易理解、复现和协作。良好的文档涵盖模型架构、[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)、数据集、评估指标等。它提供透明度，帮助团队成员和利益相关者了解已完成的工作及其原因。文档还通过提供过去决策和方法的清晰参考来帮助故障排除、维护和未来增强。

### 需要记录的关键要素

以下是项目文档中应包含的一些关键要素：

- **[项目概述](./steps-of-a-cv-project.md)**：提供项目的高级摘要，包括问题陈述、解决方案方法、预期结果和项目范围。解释计算机视觉在解决问题中的作用，并概述阶段和可交付成果。
- **模型架构**：详细说明模型的结构和设计，包括其组件、层和连接。解释所选的超参数及其背后的理由。
- **[数据准备](./data-collection-and-annotation.md)**：描述数据源、类型、格式、大小和预处理步骤。讨论数据质量、可靠性以及在训练模型之前应用的任何转换。
- **[训练过程](./model-training-tips.md)**：记录训练过程，包括使用的数据集、训练参数和[损失函数](https://www.ultralytics.com/glossary/loss-function)。解释模型是如何训练的以及训练期间遇到的任何挑战。
- **[评估指标](./model-evaluation-insights.md)**：指定用于评估模型性能的指标，如准确率、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [F1 分数](https://www.ultralytics.com/glossary/f1-score)。包括性能结果和对这些指标的分析。
- **[部署步骤](./model-deployment-options.md)**：概述部署模型所采取的步骤，包括使用的工具和平台、部署配置以及任何特定的挑战或考虑因素。
- **监控和维护程序**：提供部署后监控模型性能的详细计划。包括检测和解决数据和模型漂移的方法，并描述定期更新和重新训练的过程。

### 文档工具

在记录 AI 项目时有很多选择，开源工具特别受欢迎。其中两个是 [Jupyter Notebooks](https://docs.ultralytics.com/integrations/jupyterlab/) 和 MkDocs。Jupyter Notebooks 允许您创建带有嵌入代码、可视化和文本的交互式文档，非常适合分享实验和分析。MkDocs 是一个静态站点生成器，易于设置和部署，非常适合在线创建和托管项目文档。

## 与社区联系

加入计算机视觉爱好者社区可以帮助您解决问题并更快地学习。以下是一些连接、获得支持和分享想法的方式。

### 社区资源

- **GitHub Issues：**查看 [YOLO11 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)并使用 Issues 标签提问、报告错误和建议新功能。社区和维护者非常活跃和支持。
- **Ultralytics Discord 服务器：**加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者聊天，获得支持并分享您的经验。

### 官方文档

- **Ultralytics YOLO11 文档：**访问[官方 YOLO11 文档](./index.md)获取各种计算机视觉项目的详细指南和有用技巧。

使用这些资源将帮助您解决挑战并了解计算机视觉社区的最新趋势和实践。

## 关键要点

我们介绍了监控、维护和记录计算机视觉模型的关键技巧。定期更新和重新训练帮助模型适应新的数据模式。检测和修复数据漂移有助于保持模型准确。持续监控可以及早发现问题，良好的文档使协作和未来更新更容易。遵循这些步骤将帮助您的计算机视觉项目随时间保持成功和有效。

## 常见问题

### 如何监控已部署的计算机视觉模型的性能？

监控已部署的计算机视觉模型的性能对于确保其随时间的准确性和可靠性至关重要。您可以使用 [Prometheus](https://prometheus.io/)、[Grafana](https://grafana.com/) 和 [Evidently AI](https://www.evidentlyai.com/) 等工具来跟踪关键指标、检测异常和识别数据漂移。定期监控输入和输出，为异常行为设置警报，并使用多样化的数据源来全面了解模型的性能。有关更多详情，请查看我们关于[模型监控](#模型监控是关键)的部分。

### 部署后维护计算机视觉模型的最佳实践是什么？

维护计算机视觉模型涉及定期更新、重新训练和监控，以确保持续的准确性和相关性。最佳实践包括：

- **持续监控**：定期跟踪性能指标和数据质量。
- **数据漂移检测**：使用统计技术识别数据分布的变化。
- **定期更新和重新训练**：根据数据变化实施增量学习或定期完全重新训练。
- **文档**：维护模型架构、训练过程和评估指标的详细文档。有关更多见解，请访问我们的[模型维护](#模型维护)部分。

### 为什么数据漂移检测对 AI 模型很重要？

数据漂移检测很重要，因为它有助于识别输入数据的统计属性随时间变化的情况，这可能会降低模型性能。持续监控、统计测试（如 Kolmogorov-Smirnov 检验）和特征漂移分析等技术可以帮助及早发现问题。解决数据漂移可确保您的模型在变化的环境中保持准确和相关。在我们的[数据漂移检测](#数据漂移检测)部分了解更多关于数据漂移检测的信息。

### 我可以使用哪些工具进行计算机视觉模型中的异常检测？

对于计算机视觉模型中的异常检测，[Prometheus](https://prometheus.io/)、[Grafana](https://grafana.com/) 和 [Evidently AI](https://www.evidentlyai.com/) 等工具非常有效。这些工具可以帮助您设置警报系统来检测偏离预期行为的异常数据点或模式。可配置的警报和标准化的消息可以帮助您快速响应潜在问题。在我们的[异常检测和警报系统](#异常检测和警报系统)部分探索更多。

### 如何有效地记录我的计算机视觉项目？

计算机视觉项目的有效文档应包括：

- **项目概述**：高级摘要、问题陈述和解决方案方法。
- **模型架构**：模型结构、组件和超参数的详细信息。
- **数据准备**：关于数据源、预处理步骤和转换的信息。
- **训练过程**：训练过程、使用的数据集和遇到的挑战的描述。
- **评估指标**：用于性能评估和分析的指标。
- **部署步骤**：[模型部署](https://www.ultralytics.com/glossary/model-deployment)所采取的步骤和任何特定挑战。
- **监控和维护程序**：持续监控和维护的计划。有关更全面的指南，请参阅我们的[文档](#文档)部分。
