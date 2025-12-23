---
comments: true
description: 学习如何使用 SAHI 实现 YOLO11 的切片推理。优化内存使用并提高大规模应用的检测准确性。
keywords: YOLO11, SAHI, 切片推理, 目标检测, Ultralytics, 高分辨率图像, 计算效率, 集成指南
---

# Ultralytics 文档：使用 YOLO11 与 SAHI 进行切片推理

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 SAHI 切片推理"></a>

欢迎阅读 Ultralytics 关于如何将 YOLO11 与 [SAHI](https://github.com/obss/sahi)（切片辅助超推理）一起使用的文档。本综合指南旨在为您提供实现 SAHI 与 YOLO11 所需的所有基本知识。我们将深入探讨 SAHI 是什么、为什么切片推理对大规模应用至关重要，以及如何将这些功能与 YOLO11 集成以增强[目标检测](https://www.ultralytics.com/glossary/object-detection)性能。

<p align="center">
  <img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/sahi-sliced-inference-overview.avif" alt="SAHI 切片推理概述">
</p>

## SAHI 简介

SAHI（切片辅助超推理）是一个创新库，旨在优化大规模和高分辨率图像的目标检测算法。其核心功能在于将图像分割成可管理的切片，在每个切片上运行目标检测，然后将结果拼接在一起。SAHI 与包括 YOLO 系列在内的多种目标检测模型兼容，从而在确保优化使用计算资源的同时提供灵活性。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ILqMBah5ZvI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行 SAHI（切片辅助超推理）推理
</p>

### SAHI 的主要特性

- **无缝集成**：SAHI 与 YOLO 模型轻松集成，这意味着您可以开始切片和检测而无需大量代码修改。
- **资源效率**：通过将大图像分解成较小的部分，SAHI 优化了内存使用，允许您在资源有限的硬件上运行高质量检测。
- **高[准确性](https://www.ultralytics.com/glossary/accuracy)**：SAHI 通过在拼接过程中使用智能算法合并重叠的检测框来保持检测准确性。

## 什么是切片推理？

切片推理是指将大型或高分辨率图像细分为较小的片段（切片），在这些切片上进行目标检测，然后重新编译切片以在原始图像上重建对象位置的做法。这种技术在计算资源有限或处理可能导致内存问题的超高分辨率图像时非常有价值。

### 切片推理的优势

- **减少计算负担**：较小的图像切片处理更快，消耗更少的内存，使其能够在低端硬件上更流畅地运行。

- **保持检测质量**：由于每个切片都是独立处理的，因此只要切片足够大以捕获感兴趣的对象，目标检测的质量就不会降低。

- **增强可扩展性**：该技术允许目标检测更容易地扩展到不同大小和分辨率的图像，使其适用于从卫星图像到医学诊断的广泛应用。

<table border="0">
  <tr>
    <th>不使用 SAHI 的 YOLO11</th>
    <th>使用 SAHI 的 YOLO11</th>
  </tr>
  <tr>
    <td><img src="https://github.com/ultralytics/docs/releases/download/0/yolov8-without-sahi.avif" alt="不使用 SAHI 的 YOLO11" width="640"></td>
    <td><img src="https://github.com/ultralytics/docs/releases/download/0/yolov8-with-sahi.avif" alt="使用 SAHI 的 YOLO11" width="640"></td>
  </tr>
</table>

## 安装和准备

### 安装

要开始使用，请安装最新版本的 SAHI 和 Ultralytics：

```bash
pip install -U ultralytics sahi
```

### 导入模块和下载资源

以下是如何导入必要的模块并下载 YOLO11 模型和一些测试图像：

```python
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import download_yolo11n_model

# 下载 YOLO11 模型
model_path = "models/yolo11n.pt"
download_yolo11n_model(model_path)

# 下载测试图像
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)
```

## 使用 YOLO11 进行标准推理

### 实例化模型

您可以像这样实例化用于目标检测的 YOLO11 模型：

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cpu",  # 或 'cuda:0'
)
```

### 执行标准预测

使用图像路径或 numpy 图像执行标准推理。

```python
from sahi.predict import get_prediction
from sahi.utils.cv import read_image

# 使用图像路径
result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

# 使用 numpy 图像
result_with_np_image = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
```

### 可视化结果

导出并可视化预测的边界框和掩码：

```python
from IPython.display import Image

result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
```

## 使用 YOLO11 进行切片推理

通过指定切片尺寸和重叠比率来执行切片推理：

```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

## 处理预测结果

SAHI 提供了一个 `PredictionResult` 对象，可以转换为各种标注格式：

```python
# 访问对象预测列表
object_prediction_list = result.object_prediction_list

# 转换为 COCO 标注、COCO 预测、imantics 和 fiftyone 格式
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
```

## 批量预测

对图像目录进行批量预测：

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo11n.pt",
    model_device="cpu",  # 或 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

您现在可以使用 YOLO11 与 SAHI 进行标准推理和切片推理了。

## 引用和致谢

如果您在研究或开发工作中使用 SAHI，请引用原始 SAHI 论文并致谢作者：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{akyon2022sahi,
          title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
          author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
          journal={2022 IEEE International Conference on Image Processing (ICIP)},
          doi={10.1109/ICIP46576.2022.9897990},
          pages={966-970},
          year={2022}
        }
        ```

我们感谢 SAHI 研究团队为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 SAHI 及其创建者的更多信息，请访问 [SAHI GitHub 仓库](https://github.com/obss/sahi)。

## 常见问题

### 如何将 YOLO11 与 SAHI 集成以进行目标检测中的切片推理？

将 Ultralytics YOLO11 与 SAHI（切片辅助超推理）集成进行切片推理，通过将高分辨率图像分割成可管理的切片来优化目标检测任务。这种方法改善了内存使用并确保了高检测准确性。要开始使用，您需要安装 ultralytics 和 sahi 库：

```bash
pip install -U ultralytics sahi
```

然后，下载 YOLO11 模型和测试图像：

```python
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import download_yolo11n_model

# 下载 YOLO11 模型
model_path = "models/yolo11n.pt"
download_yolo11n_model(model_path)

# 下载测试图像
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
```

有关更详细的说明，请参阅我们的[切片推理指南](#使用-yolo11-进行切片推理)。

### 为什么应该在大图像上使用 SAHI 与 YOLO11 进行目标检测？

在大图像上使用 SAHI 与 Ultralytics YOLO11 进行目标检测有几个好处：

- **减少计算负担**：较小的切片处理更快，消耗更少的内存，使其可以在资源有限的硬件上运行高质量检测。
- **保持检测准确性**：SAHI 使用智能算法合并重叠的框，保持检测质量。
- **增强可扩展性**：通过在不同图像大小和分辨率上扩展目标检测任务，SAHI 成为卫星图像分析和医学诊断等各种应用的理想选择。

在我们的文档中了解更多关于[切片推理的优势](#切片推理的优势)。

### 使用 YOLO11 与 SAHI 时可以可视化预测结果吗？

是的，使用 YOLO11 与 SAHI 时可以可视化预测结果。以下是如何导出和可视化结果：

```python
from IPython.display import Image

result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
```

此命令将可视化的预测保存到指定目录，然后您可以加载图像以在笔记本或应用程序中查看。有关详细指南，请查看[标准推理部分](#可视化结果)。

### SAHI 为改进 YOLO11 目标检测提供了哪些功能？

SAHI（切片辅助超推理）提供了几个补充 Ultralytics YOLO11 目标检测的功能：

- **无缝集成**：SAHI 轻松与 YOLO 模型集成，只需最少的代码调整。
- **资源效率**：它将大图像分割成较小的切片，优化内存使用和速度。
- **高准确性**：通过在拼接过程中有效合并重叠的检测框，SAHI 保持高检测准确性。

要深入了解，请阅读 SAHI 的[主要特性](#sahi-的主要特性)。

### 如何使用 YOLO11 和 SAHI 处理大规模推理项目？

要使用 YOLO11 和 SAHI 处理大规模推理项目，请遵循以下最佳实践：

1. **安装所需库**：确保您拥有最新版本的 ultralytics 和 sahi。
2. **配置切片推理**：确定适合您特定项目的最佳切片尺寸和重叠比率。
3. **运行批量预测**：使用 SAHI 的功能对图像目录执行批量预测，从而提高效率。

批量预测示例：

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo11n.pt",
    model_device="cpu",  # 或 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

有关更详细的步骤，请访问我们的[批量预测](#批量预测)部分。
