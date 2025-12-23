---
comments: true
description: 探索 Ultralytics Explorer API，用于通过 SQL 查询、向量相似性搜索和语义搜索进行数据集探索。了解安装和使用技巧。
keywords: Ultralytics, Explorer API, 数据集探索, SQL 查询, 相似性搜索, 语义搜索, Python API, 嵌入, 数据分析
---

# Ultralytics Explorer API

!!! warning "社区提示 ⚠️"

    从 **`ultralytics>=8.3.10`** 开始，Ultralytics Explorer 支持已弃用。类似（且扩展的）数据集探索功能可在 [Ultralytics HUB](https://hub.ultralytics.com/) 中使用。

## 简介

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>
Explorer API 是一个用于探索数据集的 Python API。它支持使用 SQL 查询、向量相似性搜索和语义搜索来过滤和搜索数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=279"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer API 概述
</p>

## 安装

Explorer 的某些功能依赖于外部库。当您使用 Explorer 时，这些库会自动安装。要手动安装这些依赖，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

## 使用方法

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo11n.pt")

# 为数据集创建嵌入表
explorer.create_embeddings_table()

# 搜索与给定图像相似的图像
df = explorer.get_similar(img="path/to/image.jpg")

# 或搜索与给定索引相似的图像
df = explorer.get_similar(idx=0)
```

!!! note

    给定数据集和模型对的[嵌入](https://www.ultralytics.com/glossary/embeddings)表只创建一次并自动重用。这些在底层使用 [LanceDB](https://lancedb.github.io/lancedb/)，它可以在磁盘上扩展，因此您可以为像 COCO 这样的大型数据集创建和重用嵌入而不会耗尽内存。

如果您想强制更新嵌入表，可以将 `force=True` 传递给 `create_embeddings_table` 方法。

您可以直接访问 LanceDB 表对象以执行高级分析。在[使用嵌入表部分](#4-使用嵌入表)了解更多信息。

## 1. 相似性搜索

相似性搜索是一种查找与给定图像相似的图像的技术。它基于相似图像将具有相似嵌入的理念。一旦构建了嵌入表，您可以通过以下任何方式运行语义搜索：

- 在数据集中的给定索引或索引列表上：`exp.get_similar(idx=[1,10], limit=10)`
- 在不在数据集中的任何图像或图像列表上：`exp.get_similar(img=["path/to/img1", "path/to/img2"], limit=10)`

在多个输入的情况下，使用它们嵌入的聚合。

您将获得一个 pandas DataFrame，其中包含与输入最相似的 `limit` 个数据点，以及它们在嵌入空间中的距离。您可以使用此数据集执行进一步的过滤。

!!! example "语义搜索"

    === "使用图像"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
        print(similar.head())

        # 使用多个索引搜索
        similar = exp.get_similar(
            img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"],
            limit=10,
        )
        print(similar.head())
        ```

    === "使用数据集索引"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(idx=1, limit=10)
        print(similar.head())

        # 使用多个索引搜索
        similar = exp.get_similar(idx=[1, 10], limit=10)
        print(similar.head())
        ```

### 绘制相似图像

您还可以使用 `plot_similar` 方法绘制相似图像。此方法接受与 `get_similar` 相同的参数，并在网格中绘制相似图像。

!!! example "绘制相似图像"

    === "使用图像"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
        exp.create_embeddings_table()

        plt = exp.plot_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
        plt.show()
        ```

    === "使用数据集索引"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
        exp.create_embeddings_table()

        plt = exp.plot_similar(idx=1, limit=10)
        plt.show()
        ```

## 2. Ask AI（自然语言查询）

此功能允许您使用自然语言过滤数据集，无需编写 SQL。AI 驱动的查询生成器将您的提示转换为查询并返回匹配结果。例如，您可以问："显示 100 张恰好有一个人和 2 只狗的图像。也可以有其他物体"，它将生成查询并显示这些结果。
注意：此功能使用大语言模型，因此结果是概率性的，可能不准确。

!!! example "Ask AI"

    ```python
    from ultralytics.data.explorer import plot_query_result

    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
    exp.create_embeddings_table()

    df = exp.ask_ai("显示 100 张恰好有一个人和 2 只狗的图像。也可以有其他物体")
    print(df.head())

    # 绘制结果
    plt = plot_query_result(df)
    plt.show()
    ```

## 3. SQL 查询

您可以使用 `sql_query` 方法在数据集上运行 SQL 查询。此方法接受 SQL 查询作为输入，并返回包含结果的 pandas DataFrame。

!!! example "SQL 查询"

    ```python
    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
    exp.create_embeddings_table()

    df = exp.sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%'")
    print(df.head())
    ```

### 绘制 SQL 查询结果

您还可以使用 `plot_sql_query` 方法绘制 SQL 查询的结果。此方法接受与 `sql_query` 相同的参数，并在网格中绘制结果。

!!! example "绘制 SQL 查询结果"

    ```python
    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo11n.pt")
    exp.create_embeddings_table()

    # 绘制 SQL 查询
    exp.plot_sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%' LIMIT 10")
    ```

## 4. 使用嵌入表

您还可以直接使用嵌入表。一旦创建了嵌入表，您可以使用 `Explorer.table` 访问它。

!!! tip

    Explorer 在内部使用 [LanceDB](https://lancedb.github.io/lancedb/) 表。您可以使用 `Explorer.table` 对象直接访问此表，并运行原始查询、推送前置和后置过滤器等。

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()
    table = exp.table
    ```

以下是您可以使用表执行的一些示例：

### 获取原始嵌入

!!! example

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()
    table = exp.table

    embeddings = table.to_pandas()["vector"]
    print(embeddings)
    ```

### 使用前置和后置过滤器进行高级查询

!!! example

    ```python
    from ultralytics import Explorer

    exp = Explorer(model="yolo11n.pt")
    exp.create_embeddings_table()
    table = exp.table

    # 虚拟嵌入
    embedding = [i for i in range(256)]
    rs = table.search(embedding).metric("cosine").where("").limit(10)
    ```

### 创建向量索引

使用大型数据集时，您还可以创建专用的向量索引以加快查询速度。这是通过在 LanceDB 表上使用 `create_index` 方法完成的。

```python
table.create_index(num_partitions=..., num_sub_vectors=...)
```

## 5. 嵌入应用

您可以使用嵌入表执行各种探索性分析。以下是一些示例：

### 相似性索引

Explorer 带有 `similarity_index` 操作：

- 它尝试估计每个数据点与数据集其余部分的相似程度。
- 它通过计算在生成的嵌入空间中有多少图像嵌入比 `max_dist` 更接近当前图像来实现，同时考虑 `top_k` 个相似图像。

它返回一个包含以下列的 pandas DataFrame：

- `idx`：数据集中图像的索引
- `im_file`：图像文件的路径
- `count`：数据集中比 `max_dist` 更接近当前图像的图像数量
- `sim_im_files`：`count` 个相似图像的路径列表

!!! tip

    对于给定的数据集、模型、`max_dist` 和 `top_k`，相似性索引一旦生成将被重用。如果您的数据集已更改，或者您只是需要重新生成相似性索引，可以传递 `force=True`。

!!! example "相似性索引"

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()

    sim_idx = exp.similarity_index()
    ```

您可以使用相似性索引构建自定义条件来过滤数据集。例如，您可以使用以下代码过滤掉与数据集中任何其他图像不相似的图像：

```python
import numpy as np

sim_count = np.array(sim_idx["count"])
sim_idx["im_file"][sim_count > 30]
```

### 可视化嵌入空间

您还可以使用您选择的绘图工具可视化嵌入空间。例如，这是一个使用 matplotlib 的简单示例：

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用 PCA 将维度降低到 3 个分量以进行 3D 可视化
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(embeddings)

# 使用 Matplotlib Axes3D 创建 3D 散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# 散点图
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.5)
ax.set_title("降维后 256 维数据的 3D 散点图 (PCA)")
ax.set_xlabel("分量 1")
ax.set_ylabel("分量 2")
ax.set_zlabel("分量 3")

plt.show()
```

使用 Explorer API 开始创建您自己的计算机视觉数据集探索报告。如需灵感，请查看 [VOC 探索示例](explorer.md)。

## 使用 Ultralytics Explorer 构建的应用

尝试我们基于 Explorer API 的 [GUI 演示](dashboard.md)

## 即将推出

- [ ] 从数据集合并特定标签。示例 - 从 COCO 导入所有 `person` 标签，从 Cityscapes 导入 `car` 标签
- [ ] 删除相似性索引高于给定阈值的图像
- [ ] 合并/删除条目后自动持久化新数据集
- [ ] 高级数据集可视化

## 常见问题

### Ultralytics Explorer API 用于什么？

Ultralytics Explorer API 专为全面的数据集探索而设计。它允许用户使用 SQL 查询、向量相似性搜索和语义搜索来过滤和搜索数据集。这个强大的 Python API 可以处理大型数据集，非常适合使用 Ultralytics 模型的各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务。

### 如何安装 Ultralytics Explorer API？

要安装 Ultralytics Explorer API 及其依赖，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

这将自动安装 Explorer API 功能所需的所有外部库。有关其他设置详细信息，请参阅我们文档的[安装部分](#安装)。

### 如何使用 Ultralytics Explorer API 进行相似性搜索？

您可以通过创建嵌入表并查询相似图像来使用 Ultralytics Explorer API 执行相似性搜索。以下是一个基本示例：

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo11n.pt")
explorer.create_embeddings_table()

# 搜索与给定图像相似的图像
similar_images_df = explorer.get_similar(img="path/to/image.jpg")
print(similar_images_df.head())
```

有关更多详细信息，请访问[相似性搜索部分](#1-相似性搜索)。

### 将 LanceDB 与 Ultralytics Explorer 一起使用有什么好处？

LanceDB 在 Ultralytics Explorer 底层使用，提供可扩展的磁盘嵌入表。这确保您可以为像 COCO 这样的大型数据集创建和重用嵌入而不会耗尽内存。这些表只创建一次并可以重用，提高了数据处理的效率。

### Ultralytics Explorer API 中的 Ask AI 功能如何工作？

Ask AI 功能允许用户使用自然语言查询过滤数据集。此功能利用大语言模型在后台将这些查询转换为 SQL 查询。以下是一个示例：

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo11n.pt")
explorer.create_embeddings_table()

# 使用自然语言查询
query_result = explorer.ask_ai("显示 100 张恰好有一个人和 2 只狗的图像。也可以有其他物体")
print(query_result.head())
```

有关更多示例，请查看 [Ask AI 部分](#2-ask-ai自然语言查询)。
