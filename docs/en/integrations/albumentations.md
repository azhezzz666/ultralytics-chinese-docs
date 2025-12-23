---
comments: true
description: 学习如何使用 Albumentations 与 YOLO11 增强数据增强，提高模型性能，简化您的计算机视觉项目。
keywords: Albumentations, YOLO11, 数据增强, Ultralytics, 计算机视觉, 目标检测, 模型训练, 图像变换, 机器学习
---

# 使用 Albumentations 增强数据集以训练 YOLO11

当您构建[计算机视觉模型](../models/index.md)时，[训练数据](../datasets/index.md)的质量和多样性对模型性能起着重要作用。Albumentations 提供了一种快速、灵活、高效的方式来应用广泛的图像变换，可以提高模型适应真实场景的能力。它可以轻松与 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 集成，帮助您为[目标检测](../tasks/detect.md)、[分割](../tasks/segment.md)和[分类](../tasks/classify.md)任务创建强大的数据集。

通过使用 Albumentations，您可以使用几何变换和颜色调整等技术增强 YOLO11 训练数据。在本文中，我们将了解 Albumentations 如何改进您的[数据增强](../guides/preprocessing_annotated_data.md)过程，使您的 [YOLO11 项目](../solutions/index.md)更具影响力。让我们开始吧！

## 用于图像增强的 Albumentations

[Albumentations](https://albumentations.ai/) 是一个开源图像增强库，创建于 [2018 年 6 月](https://arxiv.org/pdf/1809.06839)。它旨在简化和加速[计算机视觉](https://www.ultralytics.com/blog/exploring-image-processing-computer-vision-and-machine-vision)中的图像增强过程。Albumentations 以[性能](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations)和灵活性为设计理念，支持多种增强技术，从简单的旋转和翻转变换到更复杂的亮度和对比度调整。Albumentations 帮助开发人员为[图像分类](https://www.youtube.com/watch?v=5BO0Il_YYAg)、[目标检测](https://www.youtube.com/watch?v=5ku7npMrW40&t=1s)和[分割](https://www.youtube.com/watch?v=o4Zd-IeMlSY)等任务生成丰富、多样的数据集。

您可以使用 Albumentations 轻松地将增强应用于图像、[分割掩码](https://www.ultralytics.com/glossary/image-segmentation)、[边界框](https://www.ultralytics.com/glossary/bounding-box)和[关键点](../datasets/pose/index.md)，并确保数据集的所有元素一起变换。它与 [PyTorch](../integrations/torchscript.md) 和 [TensorFlow](../integrations/tensorboard.md) 等流行的深度学习框架无缝配合，使其适用于广泛的项目。

此外，无论您处理的是小型数据集还是大规模[计算机视觉任务](../tasks/index.md)，Albumentations 都是增强的绝佳选择。它确保快速高效的处理，减少数据准备所花费的时间。同时，它有助于提高[模型性能](../guides/yolo-performance-metrics.md)，使您的模型在实际应用中更加有效。

## Albumentations 的主要功能

Albumentations 提供许多有用的功能，简化了广泛[计算机视觉应用](https://www.ultralytics.com/blog/exploring-how-the-applications-of-computer-vision-work)的复杂图像增强。以下是一些主要功能：

- **广泛的变换范围**：Albumentations 提供超过 [70 种不同的变换](https://github.com/albumentations-team/albumentations?tab=readme-ov-file#list-of-augmentations)，包括几何变化（如旋转、翻转）、颜色调整（如亮度、对比度）和噪声添加（如高斯噪声）。拥有多种选项可以创建高度多样化和强大的训练数据集。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-augmentation.avif" alt="图像增强示例">
</p>

- **高性能优化**：基于 OpenCV 和 NumPy 构建，Albumentations 使用 SIMD（单指令多数据）等高级优化技术，同时处理多个数据点以加速处理。它能快速处理大型数据集，是图像增强最快的选项之一。

- **三个级别的增强**：Albumentations 支持三个级别的增强：像素级变换、空间级变换和混合级变换。像素级变换仅影响输入图像，不改变掩码、边界框或关键点。同时，空间级变换会变换图像及其元素，如掩码和边界框。此外，混合级变换是一种独特的数据增强方式，将多个图像组合成一个。

![不同级别增强概述](https://github.com/ultralytics/docs/releases/download/0/levels-of-augmentation.avif)

- **[基准测试结果](https://albumentations.ai/docs/benchmarks/image-benchmarks/)**：在基准测试方面，Albumentations 始终优于其他库，尤其是在大型数据集上。

## 为什么应该在视觉 AI 项目中使用 Albumentations？

在图像增强方面，Albumentations 是计算机视觉任务的可靠工具。以下是您应该考虑在视觉 AI 项目中使用它的几个关键原因：

- **易于使用的 API**：Albumentations 提供单一、简单的 API，用于将广泛的增强应用于图像、掩码、边界框和关键点。它设计为易于适应不同的数据集，使[数据准备](../guides/data-collection-and-annotation.md)更简单、更高效。

- **严格的错误测试**：增强流程中的错误可能会悄悄地破坏输入数据，通常不被注意但最终会降低模型性能。Albumentations 通过全面的测试套件解决这个问题，帮助在开发早期发现错误。

- **可扩展性**：Albumentations 可以轻松添加新的增强，并通过单一接口与内置变换一起在计算机视觉流程中使用。

## 如何使用 Albumentations 增强 YOLO11 训练数据

现在我们已经介绍了 Albumentations 是什么以及它能做什么，让我们看看如何使用它来增强 YOLO11 模型训练的数据。设置很简单，因为如果您安装了 Albumentations 包，它会直接集成到 [Ultralytics 的训练模式](../modes/train.md)中并自动应用。

### 安装

要将 Albumentations 与 YOLO11 一起使用，首先确保安装了必要的包。如果未安装 Albumentations，训练期间将不会应用增强。设置完成后，您就可以创建增强数据集进行训练，Albumentations 会自动集成以增强您的模型。

!!! tip "安装"

    === "命令行"

        ```bash
        # 安装所需的包
        pip install albumentations ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 使用

安装必要的包后，您就可以开始将 Albumentations 与 YOLO11 一起使用了。当您训练 YOLO11 时，通过与 Albumentations 的集成，一组增强会自动应用，使增强模型性能变得简单。

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 使用默认增强训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "自定义变换（仅 Python API）"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 定义自定义 Albumentations 变换
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]

        # 使用自定义 Albumentations 变换训练模型
        results = model.train(
            data="coco8.yaml",
            epochs=100,
            imgsz=640,
            augmentations=custom_transforms,  # 传递自定义变换
        )
        ```

接下来，让我们仔细看看训练期间应用的具体增强。

### 模糊

Albumentations 中的 Blur 变换通过在小方形区域或核内平均像素值来对图像应用简单的模糊效果。这是使用 OpenCV `cv2.blur` 函数完成的，有助于减少图像中的噪声，尽管它也会略微降低图像细节。

以下是此集成中使用的参数和值：

- **blur_limit**：控制模糊效果的大小范围。默认范围是 (3, 7)，意味着模糊的核大小可以在 3 到 7 像素之间变化，只允许奇数以保持模糊居中。

- **p**：应用模糊的概率。在集成中，p=0.01，因此每张图像有 1% 的机会应用此模糊。低概率允许偶尔的模糊效果，引入一些变化以帮助模型泛化，而不会过度模糊图像。

<img width="776" alt="模糊增强示例" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-blur.avif">

### 中值模糊

Albumentations 中的 MedianBlur 变换对图像应用中值模糊效果，这对于在保留边缘的同时减少噪声特别有用。与典型的模糊方法不同，MedianBlur 使用中值滤波器，这对于去除椒盐噪声特别有效，同时保持边缘周围的清晰度。

以下是此集成中使用的参数和值：

- **blur_limit**：此参数控制模糊核的最大大小。在此集成中，默认范围为 (3, 7)，意味着模糊的核大小在 3 到 7 像素之间随机选择，只允许奇数值以确保正确对齐。

- **p**：设置应用中值模糊的概率。这里，p=0.01，因此变换有 1% 的机会应用于每张图像。这个低概率确保中值模糊被谨慎使用，通过偶尔看到减少噪声和保留边缘的图像来帮助模型泛化。

下图显示了此增强应用于图像的示例。

<img width="764" alt="中值模糊增强示例" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-median-blur.avif">

### 灰度

Albumentations 中的 ToGray 变换将图像转换为灰度，将其减少为单通道格式，并可选择复制此通道以匹配指定数量的输出通道。可以使用不同的方法来调整灰度亮度的计算方式，从简单的平均到更高级的技术，以实现对比度和亮度的真实感知。

以下是此集成中使用的参数和值：

- **num_output_channels**：设置输出图像中的通道数。如果此值大于 1，单个灰度通道将被复制以创建多通道灰度图像。默认设置为 3，给出具有三个相同通道的灰度图像。

- **method**：定义灰度转换方法。默认方法 "weighted_average" 应用公式 (0.299R + 0.587G + 0.114B)，与人类感知紧密对齐，提供自然外观的灰度效果。其他选项，如 "from_lab"、"desaturation"、"average"、"max" 和 "pca"，根据速度、亮度强调或细节保留的各种需求提供创建灰度图像的替代方法。

- **p**：控制灰度变换应用的频率。p=0.01 时，每张图像有 1% 的机会转换为灰度，使彩色和灰度图像的混合成为可能，以帮助模型更好地泛化。

下图显示了应用此灰度变换的示例。

<img width="759" alt="灰度增强示例" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-grayscale.avif">

### 对比度受限自适应直方图均衡化 (CLAHE)

Albumentations 中的 CLAHE 变换应用对比度受限自适应直方图均衡化 (CLAHE)，这是一种通过在局部区域（瓦片）而不是整个图像上均衡直方图来增强图像对比度的技术。CLAHE 产生平衡的增强效果，避免了标准直方图均衡化可能导致的过度放大对比度，特别是在初始对比度较低的区域。

以下是此集成中使用的参数和值：

- **clip_limit**：控制对比度增强范围。设置为默认范围 (1, 4)，它确定每个瓦片中允许的最大对比度。较高的值用于更多对比度，但也可能引入噪声。

- **tile_grid_size**：定义瓦片网格的大小，通常为 (行, 列)。默认值为 (8, 8)，意味着图像被分成 8x8 网格。较小的瓦片大小提供更局部的调整，而较大的瓦片大小创建更接近全局均衡的效果。

- **p**：应用 CLAHE 的概率。这里，p=0.01 仅在 1% 的时间引入增强效果，确保对比度调整被谨慎应用，以在训练图像中偶尔变化。

下图显示了应用 CLAHE 变换的示例。

<img width="760" alt="CLAHE 增强示例" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-CLAHE.avif">

## 使用自定义 Albumentations 变换

虽然默认的 Albumentations 集成提供了一组可靠的增强，但您可能希望为特定用例自定义变换。使用 Ultralytics YOLO11，您可以通过 Python API 使用 `augmentations` 参数轻松传递自定义 Albumentations 变换。

### 如何定义自定义变换

您可以定义自己的 Albumentations 变换列表并将其传递给训练函数。这将替换默认的 Albumentations 变换，同时保持所有其他 YOLO 增强（如 `hsv_h`、`degrees`、`mosaic` 等）处于活动状态。

以下是使用更高级变换的示例：

```python
import albumentations as A

from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 使用各种增强技术定义自定义变换
custom_transforms = [
    # 模糊变体
    A.OneOf(
        [
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ],
        p=0.3,
    ),
    # 噪声变体
    A.OneOf(
        [
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ],
        p=0.2,
    ),
    # 颜色和对比度调整
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    # 模拟遮挡
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2
    ),
]

# 使用自定义变换训练
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    augmentations=custom_transforms,
)
```

### 重要注意事项

使用自定义 Albumentations 变换时，请记住以下几点：

- **仅限 Python API**：自定义变换只能通过 Python API 传递，不能通过 CLI 或 YAML 配置文件。
- **替换默认值**：您的自定义变换将完全替换默认的 Albumentations 变换。其他 YOLO 增强保持活动状态。
- **边界框处理**：Ultralytics 自动处理大多数变换的边界框调整，但复杂的空间变换可能需要额外测试。
- **性能**：某些变换计算成本较高。监控训练速度并相应调整。
- **任务兼容性**：自定义 Albumentations 变换适用于检测和分割任务，但不适用于分类（使用不同的增强流程）。

### 自定义变换的用例

不同的应用受益于不同的增强策略：

- **医学成像**：使用弹性变形、网格扭曲和专门的噪声模式
- **航空/卫星图像**：应用模拟不同高度、天气条件和光照角度的变换
- **低光场景**：强调噪声添加和亮度调整，以训练适应挑战性光照的强大模型
- **工业检测**：为质量控制应用添加纹理变化和模拟缺陷

有关可用变换及其参数的完整列表，请访问 [Albumentations 文档](https://albumentations.ai/docs/)。

有关使用自定义 Albumentations 变换与 YOLO11 的更详细示例和最佳实践，请参阅 [YOLO 数据增强指南](../guides/yolo-data-augmentation.md#custom-albumentations-transforms-augmentations)。

## 继续学习 Albumentations

如果您有兴趣了解更多关于 Albumentations 的信息，请查看以下资源获取更深入的说明和示例：

- **[Albumentations 文档](https://albumentations.ai/docs/)**：官方文档提供了支持的变换和高级使用技术的完整范围。

- **[Ultralytics Albumentations 指南](https://docs.ultralytics.com/reference/data/augment/?h=albumentation#ultralytics.data.augment.Albumentations)**：更详细地了解促进此集成的函数细节。

- **[Albumentations GitHub 仓库](https://github.com/albumentations-team/albumentations/)**：仓库包含示例、基准测试和讨论，帮助您开始自定义增强。

## 关键要点

在本指南中，我们探讨了 Albumentations 的关键方面，这是一个用于图像增强的优秀 Python 库。我们讨论了它广泛的变换范围、优化的性能，以及如何在下一个 YOLO11 项目中使用它。

此外，如果您想了解更多关于其他 Ultralytics YOLO11 集成的信息，请访问我们的[集成指南页面](../integrations/index.md)。您将在那里找到宝贵的资源和见解。

## 常见问题

### 如何将 Albumentations 与 YOLO11 集成以改进数据增强？

Albumentations 与 YOLO11 无缝集成，如果您安装了 Albumentations 包，它会在训练期间自动应用。以下是入门方法：

```python
# 安装所需的包
# !pip install albumentations ultralytics
from ultralytics import YOLO

# 加载并使用自动增强训练模型
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100)
```

该集成包括优化的增强，如模糊、中值模糊、灰度转换和 CLAHE，具有精心调整的概率以增强模型性能。

### 与其他增强库相比，使用 Albumentations 有哪些主要优势？

Albumentations 因以下几个原因而脱颖而出：

1. 性能：基于 OpenCV 和 NumPy 构建，具有 SIMD 优化，速度卓越
2. 灵活性：支持 70 多种变换，涵盖像素级、空间级和混合级增强
3. 兼容性：与 [PyTorch](../integrations/torchscript.md) 和 [TensorFlow](../integrations/tensorboard.md) 等流行框架无缝配合
4. 可靠性：广泛的测试套件防止静默数据损坏
5. 易用性：所有增强类型的单一统一 API

### 哪些类型的计算机视觉任务可以从 Albumentations 增强中受益？

Albumentations 增强各种[计算机视觉任务](../tasks/index.md)，包括：

- [目标检测](../tasks/detect.md)：提高模型对光照、尺度和方向变化的鲁棒性
- [实例分割](../tasks/segment.md)：通过多样化变换增强掩码预测准确性
- [分类](../tasks/classify.md)：通过颜色和几何增强提高模型泛化能力
- [姿态估计](../tasks/pose.md)：帮助模型适应不同的视角和光照条件

该库多样化的增强选项使其对任何需要强大模型性能的视觉任务都很有价值。
