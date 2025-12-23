---
comments: true
description: 使用 Ultralytics Explorer GUI 解锁高级数据探索。利用语义搜索、运行 SQL 查询，并使用 AI 进行自然语言数据洞察。
keywords: Ultralytics Explorer GUI, 语义搜索, 向量相似性, SQL 查询, AI, 自然语言搜索, 数据探索, 机器学习, OpenAI, 大语言模型
---

# Explorer GUI

!!! warning "社区提示 ⚠️"

    从 **`ultralytics>=8.3.10`** 开始，Ultralytics Explorer 支持已弃用。类似（且扩展的）数据集探索功能可在 [Ultralytics HUB](https://hub.ultralytics.com/) 中使用。

Explorer GUI 基于 [Ultralytics Explorer API](api.md) 构建。它允许您运行语义/向量相似性搜索、SQL 查询，以及使用由大语言模型提供支持的 Ask AI 功能进行自然语言查询。

<p>
    <img width="1709" alt="Explorer Dashboard 截图 1" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-1.avif">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=306"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer Dashboard 概述
</p>

### 安装

```bash
pip install ultralytics[explorer]
```

!!! note

    Ask AI 功能使用 OpenAI，因此首次运行 GUI 时会提示您设置 OpenAI API 密钥。
    使用 `yolo settings openai_api_key="..."` 进行设置。

## 向量语义相似性搜索

[语义搜索](https://www.ultralytics.com/glossary/semantic-search)是一种基于给定图像查找相似图像的技术。它基于相似图像将具有相似[嵌入](https://www.ultralytics.com/glossary/embeddings)的理念。在 UI 中，您可以选择一个或多个图像并搜索与它们相似的图像。当您想要查找与给定图像相似的图像或一组表现不如预期的图像时，这非常有用。

例如，在这个 VOC 探索仪表板中，用户选择了几张飞机图像：

<p>
<img width="1710" alt="Explorer Dashboard 截图 2" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-2.avif">
</p>

运行相似性搜索后，您应该会看到类似的结果：

<p>
<img width="1710" alt="Explorer Dashboard 截图 3" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-3.avif">
</p>

## Ask AI

此功能允许您使用自然语言过滤数据集，无需编写 SQL。AI 驱动的查询生成器将您的提示转换为查询并返回匹配结果。例如，您可以问："显示 100 张恰好有一个人和 2 只狗的图像。也可以有其他物体"，它将生成查询并显示这些结果。以下是询问"显示恰好有 5 个人的 10 张图像"时的示例输出：

<p>
<img width="1709" alt="Explorer Dashboard 截图 4" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-4.avif">
</p>

注意：此功能使用[大语言模型](https://www.ultralytics.com/glossary/large-language-model-llm)，因此结果是概率性的，可能不准确。

## 在计算机视觉数据集上运行 SQL 查询

您可以在数据集上运行 SQL 查询来过滤它。如果您只提供 WHERE 子句，它也可以工作。例如，以下 WHERE 子句返回至少包含一个人和一只狗的图像：

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

<p>
<img width="1707" alt="Explorer Dashboard 截图 5" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-5.avif">
</p>

此演示使用 Explorer API 构建，您可以使用它创建自己的探索性笔记本或脚本，以获取数据集的洞察。要开始使用，请查看 [Explorer API 文档](api.md)。

## 常见问题

### 什么是 Ultralytics Explorer GUI，如何安装？

Ultralytics Explorer GUI 是一个强大的界面，使用 [Ultralytics Explorer API](api.md) 解锁高级数据探索功能。它允许您运行语义/向量相似性搜索、SQL 查询，以及使用由[大语言模型](https://www.ultralytics.com/glossary/large-language-model-llm)（LLMs）提供支持的 Ask AI 功能进行自然语言查询。

要安装 Explorer GUI，您可以使用 pip：

```bash
pip install ultralytics[explorer]
```

注意：要使用 Ask AI 功能，您需要设置 OpenAI API 密钥：`yolo settings openai_api_key="..."`。

### Ultralytics Explorer GUI 中的语义搜索功能如何工作？

Ultralytics Explorer GUI 中的语义搜索功能允许您根据嵌入查找与给定图像相似的图像。此技术对于识别和探索具有视觉相似性的图像非常有用。要使用此功能，请在 UI 中选择一个或多个图像并执行相似图像搜索。结果将显示与所选图像非常相似的图像，便于高效的数据集探索和[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)。

通过访问[功能概述](#向量语义相似性搜索)部分了解更多关于语义搜索和其他功能的信息。

### 我可以在 Ultralytics Explorer GUI 中使用自然语言过滤数据集吗？

是的，通过由大语言模型（LLMs）提供支持的 Ask AI 功能，您可以使用自然语言查询过滤数据集。您不需要精通 SQL。例如，您可以问"显示 100 张恰好有一个人和 2 只狗的图像。也可以有其他物体"，AI 将在后台生成适当的查询以提供所需的结果。

### 如何使用 Ultralytics Explorer GUI 在数据集上运行 SQL 查询？

Ultralytics Explorer GUI 允许您直接在数据集上运行 SQL 查询，以高效地过滤和管理数据。要运行查询，请导航到 GUI 中的 SQL 查询部分并编写您的查询。例如，要显示至少有一个人和一只狗的图像，您可以使用：

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

您也可以只提供 WHERE 子句，使查询过程更加灵活。

有关更多详细信息，请参阅 [SQL 查询部分](#在计算机视觉数据集上运行-sql-查询)。

### 使用 Ultralytics Explorer GUI 进行数据探索有什么好处？

Ultralytics Explorer GUI 通过语义搜索、SQL 查询和通过 Ask AI 功能进行自然语言交互等功能增强数据探索。这些功能允许用户：

- 高效查找视觉相似的图像。
- 使用复杂的 SQL 查询过滤数据集。
- 利用 AI 执行自然语言搜索，无需高级 SQL 专业知识。

这些功能使其成为希望深入了解数据集的开发人员、研究人员和数据科学家的多功能工具。

在 [Explorer GUI 文档](#explorer-gui)中探索更多关于这些功能的信息。
