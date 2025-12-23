---
comments: true
description: 探索 Ultralytics Explorer，用于语义搜索、SQL 查询、向量相似性和自然语言数据集探索。
keywords: Ultralytics Explorer, 计算机视觉数据集, 语义搜索, SQL 查询, 向量相似性, 数据集可视化, Python API, 机器学习, 计算机视觉
---

# Ultralytics Explorer

!!! warning "社区提示 ⚠️"

    从 **`ultralytics>=8.3.10`** 开始，Ultralytics Explorer 支持已弃用。类似（且扩展的）数据集探索功能可在 [Ultralytics HUB](https://hub.ultralytics.com/) 中使用。

<p>
    <img width="1709" alt="Ultralytics Explorer 截图 1" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-1.avif">
</p>

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>

Ultralytics Explorer 是一个使用语义搜索、SQL 查询、向量相似性搜索和自然语言提示来探索计算机视觉数据集的工具。它还提供 Python API 来访问相同的功能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer API | 语义搜索、SQL 查询和 Ask AI 功能
</p>

## 安装可选依赖

Explorer 的某些功能依赖于外部库。当您使用 Explorer 时，这些库会自动安装。要手动安装这些依赖，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

!!! tip

    Explorer 基于嵌入/语义搜索和 SQL 查询工作，由 [LanceDB](https://lancedb.com/) 无服务器向量数据库提供支持。与传统的内存数据库不同，它持久化到磁盘而不牺牲性能，因此您可以在本地扩展到像 COCO 这样的大型数据集而不会耗尽内存。

## Explorer API

这是一个用于探索数据集的 Python API。它也为 GUI Explorer 提供支持。您可以使用它创建自己的探索性笔记本或脚本，以获取数据集的洞察。

在 [Explorer API 文档](api.md)中探索完整的功能和使用示例。

## GUI Explorer 使用

GUI 演示在浏览器中运行，允许您为数据集创建[嵌入](https://www.ultralytics.com/glossary/embeddings)并搜索相似图像、运行 SQL 查询和执行语义搜索。可以使用以下命令运行：

```bash
yolo explorer
```

!!! note

    Ask AI 功能使用 OpenAI，因此首次运行 GUI 时会提示您设置 OpenAI 的 API 密钥。
    您可以这样设置 - `yolo settings openai_api_key="..."`

<p>
    <img width="1709" alt="Ultralytics Explorer OpenAI 集成" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-explorer-openai-integration.avif">
</p>

## 常见问题

### 什么是 Ultralytics Explorer，它如何帮助处理计算机视觉数据集？

Ultralytics Explorer 是一个强大的工具，旨在通过语义搜索、SQL 查询、向量相似性搜索甚至自然语言来探索[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)（CV）数据集。这个多功能工具提供 GUI 和 Python API，允许用户与数据集无缝交互。通过利用 [LanceDB](https://lancedb.com/) 等技术，Ultralytics Explorer 确保高效、可扩展地访问大型数据集，而不会过度使用内存。无论您是进行详细的数据集分析还是探索数据模式，Ultralytics Explorer 都能简化整个过程。

了解更多关于 [Explorer API](api.md) 的信息。

### 如何安装 Ultralytics Explorer 的依赖？

要手动安装 Ultralytics Explorer 所需的可选依赖，可以使用以下 `pip` 命令：

```bash
pip install ultralytics[explorer]
```

这些依赖对于语义搜索和 SQL 查询的完整功能至关重要。通过包含由 [LanceDB](https://lancedb.com/) 提供支持的库，安装确保数据库操作保持高效和可扩展，即使对于像 [COCO](../detect/coco.md) 这样的大型数据集也是如此。

### 如何使用 Ultralytics Explorer 的 GUI 版本？

使用 Ultralytics Explorer 的 GUI 版本非常简单。安装必要的依赖后，您可以使用以下命令启动 GUI：

```bash
yolo explorer
```

GUI 提供了一个用户友好的界面，用于创建数据集嵌入、搜索相似图像、运行 SQL 查询和进行语义搜索。此外，与 OpenAI 的 Ask AI 功能集成允许您使用自然语言查询数据集，增强了灵活性和易用性。

有关存储和可扩展性信息，请查看我们的[安装说明](#安装可选依赖)。

### Ultralytics Explorer 中的 Ask AI 功能是什么？

Ultralytics Explorer 中的 Ask AI 功能允许用户使用自然语言查询与数据集交互。由 [OpenAI](https://www.ultralytics.com/blog/openai-gpt-4o-showcases-ai-potential) 提供支持，此功能使您能够提出复杂问题并获得有洞察力的答案，而无需编写 SQL 查询或类似命令。要使用此功能，您需要在首次运行 GUI 时设置 OpenAI API 密钥：

```bash
yolo settings openai_api_key="YOUR_API_KEY"
```

有关此功能及其集成方式的更多信息，请参阅我们的 [GUI Explorer 使用](#gui-explorer-使用)部分。

### 我可以在 Google Colab 中运行 Ultralytics Explorer 吗？

是的，Ultralytics Explorer 可以在 Google Colab 中运行，为数据集探索提供便捷而强大的环境。您可以通过打开提供的 Colab 笔记本开始，该笔记本已预配置所有必要的设置：

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>

此设置允许您充分探索数据集，利用 Google 的云资源。在我们的 [Google Colab 指南](../../integrations/google-colab.md)中了解更多信息。
