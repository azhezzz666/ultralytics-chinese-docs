---
comments: true
description: 使用 OpenAI CLIP、Meta FAISS 和 Flask 构建语义图像搜索 Web 应用。学习如何嵌入图像并使用自然语言检索它们。
keywords: CLIP, FAISS, Flask, 语义搜索, 图像检索, OpenAI, Ultralytics, 教程, 计算机视觉, Web 应用
---

# 使用 OpenAI CLIP 和 Meta FAISS 进行语义图像搜索

## 简介

本指南将引导您使用 [OpenAI CLIP](https://openai.com/blog/clip)、[Meta FAISS](https://github.com/facebookresearch/faiss) 和 [Flask](https://flask.palletsprojects.com/en/stable/) 构建**语义图像搜索**引擎。通过将 CLIP 强大的视觉-语言嵌入与 FAISS 高效的最近邻搜索相结合，您可以创建一个功能齐全的 Web 界面，使用自然语言查询检索相关图像。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zplKRlX3sLg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>相似性搜索如何工作 | 使用 OpenAI CLIP、META FAISS 和 Ultralytics 包进行视觉搜索 🎉
</p>

## 语义图像搜索视觉预览

![Flask 网页语义搜索结果概述](https://github.com/ultralytics/docs/releases/download/0/flask-ui.avif)

## 工作原理

- **CLIP** 使用视觉编码器（例如 ResNet 或 ViT）处理图像，使用文本编码器（基于 Transformer）处理语言，将两者投影到相同的多模态嵌入空间。这允许使用[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)直接比较文本和图像。
- **FAISS**（Facebook AI 相似性搜索）构建图像嵌入的索引，并能够快速、可扩展地检索与给定查询最接近的向量。
- **Flask** 提供一个简单的 Web 界面来提交自然语言查询并显示索引中语义匹配的图像。

此架构支持零样本搜索，这意味着您不需要标签或类别，只需要图像数据和一个好的提示。

!!! example "使用 Ultralytics Python 包进行语义图像搜索"

    ??? note "图像路径警告"

         如果您使用自己的图像，请确保提供图像目录的绝对路径。否则，由于 Flask 的文件服务限制，图像可能不会出现在网页上。

    === "Python"

        ```python
        from ultralytics import solutions

        app = solutions.SearchApp(
            # data = "path/to/img/directory" # 可选，使用您自己的图像构建搜索引擎
            device="cpu"  # 配置处理设备，例如 "cpu" 或 "cuda"
        )

        app.run(debug=False)  # 您也可以使用 `debug=True` 参数进行测试
        ```

## `VisualAISearch` 类

此类执行所有后端操作：

- 从本地图像加载或构建 FAISS 索引。
- 使用 CLIP 提取图像和文本[嵌入](https://platform.openai.com/docs/guides/embeddings)。
- 使用余弦相似度执行相似性搜索。

!!! example "相似图像搜索"

    ??? note "图像路径警告"

         如果您使用自己的图像，请确保提供图像目录的绝对路径。否则，由于 Flask 的文件服务限制，图像可能不会出现在网页上。

    === "Python"

        ```python
        from ultralytics import solutions

        searcher = solutions.VisualAISearch(
            # data = "path/to/img/directory" # 可选，使用您自己的图像构建搜索引擎
            device="cuda"  # 配置处理设备，例如 "cpu" 或 "cuda"
        )

        results = searcher("a dog sitting on a bench")

        # 排名结果：
        #     - 000000546829.jpg | 相似度: 0.3269
        #     - 000000549220.jpg | 相似度: 0.2899
        #     - 000000517069.jpg | 相似度: 0.2761
        #     - 000000029393.jpg | 相似度: 0.2742
        #     - 000000534270.jpg | 相似度: 0.2680
        ```

## `VisualAISearch` 参数

下表列出了 `VisualAISearch` 的可用参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["data"]) }}
{% from "macros/track-args.md" import param_table %}
{{ param_table(["device"]) }}

## 使用 CLIP 和 FAISS 进行语义图像搜索的优势

使用 CLIP 和 FAISS 构建自己的语义图像搜索系统具有几个显著优势：

1. **零样本能力**：您不需要在特定数据集上训练模型。CLIP 的零样本学习让您可以使用自由形式的自然语言对任何图像数据集执行搜索查询，节省时间和资源。

2. **类人理解**：与基于关键词的搜索引擎不同，CLIP 理解语义上下文。它可以根据抽象、情感或关系查询检索图像，如"大自然中快乐的孩子"或"夜晚的未来城市天际线"。

    ![OpenAI Clip 图像检索工作流程](https://github.com/ultralytics/docs/releases/download/0/clip-image-retrieval.avif)

3. **无需标签或元数据**：传统图像搜索系统需要仔细标注的数据。这种方法只需要原始图像。CLIP 无需任何手动标注即可生成嵌入。

4. **灵活且可扩展的搜索**：FAISS 即使在大规模数据集上也能实现快速的最近邻搜索。它针对速度和内存进行了优化，即使有数千（或数百万）个嵌入也能实现实时响应。

    ![Meta FAISS 嵌入向量构建工作流程](https://github.com/ultralytics/docs/releases/download/0/faiss-indexing-workflow.avif)

5. **跨领域应用**：无论您是构建个人照片档案、创意灵感工具、产品搜索引擎，还是艺术推荐系统，这个技术栈都能以最少的调整适应各种领域。

## 常见问题

### CLIP 如何理解图像和文本？

[CLIP](https://github.com/openai/CLIP)（对比语言图像预训练）是由 [OpenAI](https://openai.com/) 开发的模型，它学习连接视觉和语言信息。它在大量图像与自然语言描述配对的数据集上进行训练。这种训练使它能够将图像和文本映射到共享的嵌入空间，因此您可以使用向量相似度直接比较它们。

### 为什么 CLIP 被认为对 AI 任务如此强大？

CLIP 的突出之处在于其泛化能力。它不是仅针对特定标签或任务进行训练，而是从自然语言本身学习。这使它能够处理灵活的查询，如"一个人骑水上摩托"或"超现实的梦境"，使其适用于从分类到创意语义搜索的各种任务，而无需重新训练。

### FAISS 在这个项目（语义搜索）中究竟做什么？

[FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)（Facebook AI 相似性搜索）是一个工具包，可帮助您非常高效地搜索高维向量。一旦 CLIP 将您的图像转换为嵌入，FAISS 就可以快速轻松地找到与文本查询最接近的匹配项，非常适合实时图像检索。

### 如果 CLIP 和 FAISS 来自 OpenAI 和 Meta，为什么要使用 [Ultralytics](https://www.ultralytics.com/) [Python 包](https://github.com/ultralytics/ultralytics/)？

虽然 CLIP 和 FAISS 分别由 OpenAI 和 Meta 开发，但 [Ultralytics Python 包](https://pypi.org/project/ultralytics/)简化了它们集成到完整语义图像搜索管道中的过程，只需 2 行代码即可工作：

!!! example "相似图像搜索"

    === "Python"

        ```python
        from ultralytics import solutions

        searcher = solutions.VisualAISearch(
            # data = "path/to/img/directory" # 可选，使用您自己的图像构建搜索引擎
            device="cuda"  # 配置处理设备，例如 "cpu" 或 "cuda"
        )

        results = searcher("a dog sitting on a bench")

        # 排名结果：
        #     - 000000546829.jpg | 相似度: 0.3269
        #     - 000000549220.jpg | 相似度: 0.2899
        #     - 000000517069.jpg | 相似度: 0.2761
        #     - 000000029393.jpg | 相似度: 0.2742
        #     - 000000534270.jpg | 相似度: 0.2680
        ```

这个高级实现处理：

- 基于 CLIP 的图像和文本嵌入生成。
- FAISS 索引创建和管理。
- 使用余弦相似度的高效语义搜索。
- 基于目录的图像加载和[可视化](https://www.ultralytics.com/glossary/data-visualization)。

### 我可以自定义这个应用的前端吗？

是的。当前设置使用 Flask 和基本的 HTML 前端，但您可以用自己的 HTML 替换它，或使用 React、Vue 或其他前端框架构建更动态的 UI。Flask 可以作为您自定义界面的后端 API。

### 是否可以搜索视频而不是静态图像？

不能直接搜索。一个简单的解决方法是从视频中提取单独的帧（例如每秒一帧），将它们视为独立图像，然后将这些图像输入系统。这样，搜索引擎可以语义索引视频中的视觉时刻。
