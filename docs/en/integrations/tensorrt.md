---
comments: true
description: 学习如何将 YOLO11 模型转换为 TensorRT 以实现 NVIDIA GPU 上的高速推理。通过我们的分步指南提升效率并部署优化模型。
keywords: YOLOv8, YOLO11, TensorRT, NVIDIA, GPU, 深度学习, 模型优化, 高速推理, 模型导出
---

# YOLO11 模型的 TensorRT 导出

在高性能环境中部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型可能需要一种能够最大化速度和效率的格式。当您在 NVIDIA GPU 上部署模型时尤其如此。

通过使用 TensorRT 导出格式，您可以增强 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以在 NVIDIA 硬件上实现快速高效的推理。本指南将为您提供易于遵循的转换步骤，并帮助您在[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)项目中充分利用 NVIDIA 的先进技术。

## TensorRT

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tensorrt-overview.avif" alt="TensorRT 概览">
</p>

[TensorRT](https://developer.nvidia.com/tensorrt) 由 NVIDIA 开发，是一个专为高速深度学习推理设计的高级软件开发工具包（SDK）。它非常适合[目标检测](https://www.ultralytics.com/glossary/object-detection)等实时应用。

该工具包针对 NVIDIA GPU 优化深度学习模型，从而实现更快、更高效的操作。TensorRT 模型经过 TensorRT 优化，包括层融合、精度校准（INT8 和 FP16）、动态张量内存管理和内核自动调优等技术。将深度学习模型转换为 TensorRT 格式使开发人员能够充分发挥 NVIDIA GPU 的潜力。

TensorRT 以其与各种模型格式的兼容性而闻名，包括 TensorFlow、[PyTorch](https://www.ultralytics.com/glossary/pytorch) 和 ONNX，为开发人员提供了一个灵活的解决方案，用于集成和优化来自不同框架的模型。这种多功能性使得能够在不同的硬件和软件环境中高效[部署模型](https://www.ultralytics.com/glossary/model-deployment)。

## TensorRT 模型的主要特性

TensorRT 模型提供了一系列关键特性，有助于在高速深度学习推理中实现效率和有效性：

- **精度校准**：TensorRT 支持精度校准，允许针对特定精度要求微调模型。这包括支持 INT8 和 FP16 等降低精度的格式，可以进一步提高推理速度，同时保持可接受的精度水平。

- **层融合**：TensorRT 优化过程包括层融合，其中[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)的多个层被组合成单个操作。这通过最小化内存访问和计算来减少计算开销并提高推理速度。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tensorrt-layer-fusion.avif" alt="TensorRT 层融合">
</p>

- **动态张量内存管理**：TensorRT 在推理期间高效管理张量内存使用，减少内存开销并优化内存分配。这导致更高效的 GPU 内存利用。

- **自动内核调优**：TensorRT 应用自动内核调优，为模型的每一层选择最优化的 GPU 内核。这种自适应方法确保模型充分利用 GPU 的计算能力。

## TensorRT 中的部署选项

在我们查看将 YOLO11 模型导出为 TensorRT 格式的代码之前，让我们了解 TensorRT 模型通常在哪里使用。

TensorRT 提供多种部署选项，每种选项在集成便利性、性能优化和灵活性之间有不同的平衡：

- **在 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 中部署**：此方法将 TensorRT 集成到 TensorFlow 中，允许优化的模型在熟悉的 TensorFlow 环境中运行。它对于具有支持和不支持层混合的模型很有用，因为 TF-TRT 可以高效处理这些情况。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tf-trt-workflow.avif" alt="TensorRT 概览">
</p>

- **独立 TensorRT 运行时 API**：提供精细控制，非常适合性能关键型应用。它更复杂，但允许自定义实现不支持的运算符。

- **NVIDIA Triton 推理服务器**：支持来自各种框架的模型的选项。特别适合云端或边缘推理，它提供并发模型执行和模型分析等功能。

## 将 YOLO11 模型导出为 TensorRT

您可以通过将 YOLO11 模型转换为 TensorRT 格式来提高执行效率和优化性能。

### 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [YOLO11 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 使用方法

在深入了解使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型系列](../models/index.md)。这将帮助您选择最适合项目需求的模型。

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TensorRT 格式
        model.export(format="engine")  # 创建 'yolo11n.engine'

        # 加载导出的 TensorRT 模型
        tensorrt_model = YOLO("yolo11n.engine")

        # 运行推理
        results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TensorRT 格式
        yolo export model=yolo11n.pt format=engine # 创建 'yolo11n.engine'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数        | 类型              | 默认值         | 描述                                                                                                                                                                                                                                                      |
| ----------- | ----------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`             | `'engine'`     | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                                                                                               |
| `imgsz`     | `int` 或 `tuple`  | `640`          | 模型输入所需的图像尺寸。可以是正方形图像的整数，也可以是特定尺寸的元组 `(height, width)`。                                                                                                                                |
| `half`      | `bool`            | `False`        | 启用 FP16（半精度）量化，减少模型大小并可能在支持的硬件上加速推理。                                                                                                                                     |
| `int8`      | `bool`            | `False`        | 激活 INT8 量化，进一步压缩模型并加速推理，同时[精度](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。                                                                    |
| `dynamic`   | `bool`            | `False`        | 允许动态输入尺寸，增强处理不同图像尺寸的灵活性。                                                                                                                                                                          |
| `simplify`  | `bool`            | `True`         | 使用 `onnxslim` 简化模型图，可能提高性能和兼容性。                                                                                                                                                                 |
| `workspace` | `float` 或 `None` | `None`         | 设置 TensorRT 优化的最大工作空间大小（GiB），平衡内存使用和性能；使用 `None` 让 TensorRT 自动分配到设备最大值。                                                                                      |
| `nms`       | `bool`            | `False`        | 添加非极大值抑制（NMS），对于准确高效的检测后处理至关重要。                                                                                                                                                              |
| `batch`     | `int`             | `1`            | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。                                                                                                                          |
| `data`      | `str`             | `'coco8.yaml'` | [数据集](https://docs.ultralytics.com/datasets/)配置文件的路径（默认：`coco8.yaml`），对量化至关重要。                                                                                                            |
| `fraction`  | `float`           | `1.0`          | 指定用于 INT8 量化校准的数据集比例。允许在完整数据集的子集上进行校准，适用于实验或资源有限的情况。如果启用 INT8 但未指定，将使用完整数据集。 |
| `device`    | `str`             | `None`         | 指定导出设备：GPU（`device=0`）、NVIDIA Jetson 的 DLA（`device=dla:0` 或 `device=dla:1`）。                                                                                                                                                  |

!!! tip

    导出到 TensorRT 时，请确保使用支持 CUDA 的 GPU。

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

### 使用 INT8 量化导出 TensorRT

使用 TensorRT 以 INT8 [精度](https://www.ultralytics.com/glossary/precision)导出 Ultralytics YOLO 模型会执行训练后量化（PTQ）。TensorRT 使用校准进行 PTQ，它测量 YOLO 模型在代表性输入数据上处理推理时每个激活张量内的激活分布，然后使用该分布来估计每个张量的缩放值。每个作为量化候选的激活张量都有一个通过校准过程推导出的相关缩放值。

在处理隐式量化网络时，TensorRT 会机会性地使用 INT8 来优化层执行时间。如果一个层在 INT8 下运行更快，并且其数据输入和输出上分配了量化缩放值，则该层将被分配 INT8 精度的内核，否则 TensorRT 会根据哪种精度能为该层带来更快的执行时间来选择 FP32 或 FP16 精度的内核。

!!! tip

    **关键**是确保用于部署 TensorRT 模型权重的设备与用于 INT8 精度导出的设备相同，因为校准结果可能因设备而异。

#### 配置 INT8 导出

使用 [export](../modes/export.md) 导出 Ultralytics YOLO 模型时提供的参数将**极大地**影响导出模型的性能。它们还需要根据可用的设备资源进行选择，但默认参数_应该_适用于大多数 [Ampere（或更新）NVIDIA 独立 GPU](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。使用的校准算法是 `"MINMAX_CALIBRATION"`，您可以在 [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/MinMaxCalibrator.html)中阅读有关可用选项的更多详细信息。Ultralytics 测试发现 `"MINMAX_CALIBRATION"` 是最佳选择，导出固定使用此算法。

- `workspace`：控制转换模型权重时设备内存分配的大小（GiB）。
    - 根据您的校准需求和资源可用性调整 `workspace` 值。虽然较大的 `workspace` 可能会增加校准时间，但它允许 TensorRT 探索更广泛的优化策略，可能增强模型性能和[精度](https://www.ultralytics.com/glossary/accuracy)。相反，较小的 `workspace` 可以减少校准时间，但可能限制优化策略，影响量化模型的质量。

    - 默认值为 `workspace=None`，这将允许 TensorRT 自动分配内存，手动配置时，如果校准崩溃（无警告退出），可能需要增加此值。

    - 如果 `workspace` 的值大于设备可用内存，TensorRT 将在导出期间报告 `UNSUPPORTED_STATE`，这意味着应降低 `workspace` 的值或将其设置为 `None`。

    - 如果 `workspace` 设置为最大值且校准失败/崩溃，请考虑使用 `None` 进行自动分配，或通过减少 `imgsz` 和 `batch` 的值来降低内存需求。

    - <u><b>请记住</b> INT8 校准是特定于每个设备的</u>，借用"高端"GPU 进行校准，可能会导致在另一台设备上运行推理时性能不佳。

- `batch`：将用于推理的最大批量大小。在推理期间可以使用较小的批量，但推理不会接受大于指定值的批量。

!!! note

    在校准期间，将使用提供的 `batch` 大小的两倍。使用小批量可能导致校准期间的缩放不准确。这是因为该过程根据它看到的数据进行调整。小批量可能无法捕获完整的值范围，导致最终校准出现问题，因此 `batch` 大小会自动加倍。如果未指定[批量大小](https://www.ultralytics.com/glossary/batch-size) `batch=1`，校准将以 `batch=1 * 2` 运行以减少校准缩放误差。

NVIDIA 的实验使他们建议使用至少 500 张代表您模型数据的校准图像进行 INT8 量化校准。这是一个指导原则，而不是_硬性_要求，<u>**您需要实验以确定数据集需要什么才能表现良好**。</u>由于 TensorRT 的 INT8 校准需要校准数据，因此在 TensorRT 的 `int8=True` 时务必使用 `data` 参数，并使用 `data="my_dataset.yaml"`，这将使用[验证](../modes/val.md)中的图像进行校准。当导出到带有 INT8 量化的 TensorRT 时未传递 `data` 值，默认将使用基于模型任务的["小型"示例数据集](../datasets/index.md)之一，而不是抛出错误。

!!! example

    === "Python"

        ```{ .py .annotate }
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(
            format="engine",
            dynamic=True,  # (1)!
            batch=8,  # (2)!
            workspace=4,  # (3)!
            int8=True,
            data="coco.yaml",  # (4)!
        )

        # 加载导出的 TensorRT INT8 模型
        model = YOLO("yolov8n.engine", task="detect")

        # 运行推理
        result = model.predict("https://ultralytics.com/images/bus.jpg")
        ```

        1. 使用动态轴导出，当使用 `int8=True` 导出时，即使未显式设置，这也将默认启用。有关更多信息，请参阅[导出参数](../modes/export.md#arguments)。
        2. 设置导出模型的最大批量大小为 8，校准时使用 `batch = 2 * 8` 以避免校准期间的缩放误差。
        3. 分配 4 GiB 内存，而不是为转换过程分配整个设备内存。
        4. 使用 [COCO 数据集](../datasets/detect/coco.md)进行校准，特别是用于[验证](../modes/val.md)的图像（共 5,000 张）。


    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为带有 INT8 量化的 TensorRT 格式
        yolo export model=yolo11n.pt format=engine batch=8 workspace=4 int8=True data=coco.yaml # 创建 'yolov8n.engine'

        # 使用导出的 TensorRT 量化模型运行推理
        yolo predict model=yolov8n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

???+ warning "校准缓存"

    TensorRT 将生成一个校准 `.cache`，可以重复使用以加速使用相同数据导出未来模型权重，但当数据差异很大或 `batch` 值发生剧烈变化时，这可能导致校准不佳。在这些情况下，应将现有的 `.cache` 重命名并移动到不同的目录或完全删除。

#### 使用 YOLO 与 TensorRT INT8 的优势

- **减少模型大小：** 从 FP32 到 INT8 的量化可以将模型大小减少 4 倍（在磁盘或内存中），从而加快下载时间、降低存储需求，并在部署模型时减少内存占用。

- **更低的功耗：** INT8 导出的 YOLO 模型的降低精度操作与 FP32 模型相比可以消耗更少的功率，特别是对于电池供电的设备。

- **提高推理速度：** TensorRT 针对目标硬件优化模型，可能在 GPU、嵌入式设备和加速器上实现更快的推理速度。

??? note "关于推理速度的说明"

    使用导出到 TensorRT INT8 的模型进行的前几次推理调用可能会有比平时更长的预处理、推理和/或后处理时间。当在推理期间更改 `imgsz` 时也可能发生这种情况，特别是当 `imgsz` 与导出期间指定的不同时（导出 `imgsz` 设置为 TensorRT "最佳"配置文件）。

#### 使用 YOLO 与 TensorRT INT8 的缺点

- **评估指标下降：** 使用较低精度意味着 `mAP`、`Precision`、`Recall` 或任何[其他用于评估模型性能的指标](../guides/yolo-performance-metrics.md)可能会有所下降。请参阅[性能结果部分](#ultralytics-yolo-tensorrt-导出性能)，比较在各种设备的小样本上使用 INT8 导出时 `mAP50` 和 `mAP50-95` 的差异。

- **增加开发时间：** 为数据集和设备找到 INT8 校准的"最佳"设置可能需要大量测试。

- **硬件依赖性：** 校准和性能提升可能高度依赖硬件，模型权重的可移植性较差。

## Ultralytics YOLO TensorRT 导出性能

### NVIDIA A100

!!! tip "性能"

    使用 Ubuntu 22.04.3 LTS、`python 3.10.12`、`ultralytics==8.2.4`、`tensorrt==8.6.1.post1` 测试

    === "检测（COCO）"

        有关在 [COCO](../datasets/detect/coco.md) 上训练的这些模型的使用示例，请参阅[检测文档](../tasks/detect.md)，其中包含 80 个预训练类别。

        !!! note

            使用预训练权重 `yolov8n.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.52         | 0.51 \| 0.56       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 0.52         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.34         | 0.34 \| 0.41       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 0.33         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.28         | 0.27 \| 0.31       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 0.29         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "分割（COCO）"

        有关在 [COCO](../datasets/segment/coco.md) 上训练的这些模型的使用示例，请参阅[分割文档](../tasks/segment.md)，其中包含 80 个预训练类别。

        !!! note

            使用预训练权重 `yolov8n-seg.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | mAP<sup>val</sup><br>50(M) | mAP<sup>val</sup><br>50-95(M) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.62         | 0.61 \| 0.68       |                      |                         |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 0.63         |                    | 0.52                 | 0.36                    | 0.49                 | 0.31                    | 1       | 640                   |
        | FP16      | Predict      | 0.40         | 0.39 \| 0.44       |                      |                         |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 0.43         |                    | 0.52                 | 0.36                    | 0.49                 | 0.30                    | 1       | 640                   |
        | INT8      | Predict      | 0.34         | 0.33 \| 0.37       |                      |                         |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 0.36         |                    | 0.46                 | 0.32                    | 0.43                 | 0.27                    | 1       | 640                   |

    === "分类（ImageNet）"

        有关在 [ImageNet](../datasets/classify/imagenet.md) 上训练的这些模型的使用示例，请参阅[分类文档](../tasks/classify.md)，其中包含 1000 个预训练类别。

        !!! note

            使用预训练权重 `yolov8n-cls.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试        | mean<br>(ms) | min \| max<br>(ms) | top-1 | top-5 | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|------------------|--------------|--------------------|-------|-------|---------|-----------------------|
        | FP32      | Predict          | 0.26         | 0.25 \| 0.28       |       |       | 8       | 640                   |
        | FP32      | ImageNet<sup>val</sup> | 0.26         |                    | 0.35  | 0.61  | 1       | 640                   |
        | FP16      | Predict          | 0.18         | 0.17 \| 0.19       |       |       | 8       | 640                   |
        | FP16      | ImageNet<sup>val</sup> | 0.18         |                    | 0.35  | 0.61  | 1       | 640                   |
        | INT8      | Predict          | 0.16         | 0.15 \| 0.57       |       |       | 8       | 640                   |
        | INT8      | ImageNet<sup>val</sup> | 0.15         |                    | 0.32  | 0.59  | 1       | 640                   |

    === "姿态（COCO）"

        有关在 [COCO](../datasets/pose/coco.md) 上训练的这些模型的使用示例，请参阅[姿态估计文档](../tasks/pose.md)，其中包含 1 个预训练类别"person"。

        !!! note

            使用预训练权重 `yolov8n-pose.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | mAP<sup>val</sup><br>50(P) | mAP<sup>val</sup><br>50-95(P) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 0.54         | 0.53 \| 0.58       |                      |                         |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 0.55         |                    | 0.91                 | 0.69                    | 0.80                 | 0.51                    | 1       | 640                   |
        | FP16      | Predict      | 0.37         | 0.35 \| 0.41       |                      |                         |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 0.36         |                    | 0.91                 | 0.69                    | 0.80                 | 0.51                    | 1       | 640                   |
        | INT8      | Predict      | 0.29         | 0.28 \| 0.33       |                      |                         |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 0.30         |                    | 0.90                 | 0.68                    | 0.78                 | 0.47                    | 1       | 640                   |

    === "OBB（DOTAv1）"

        有关在 [DOTAv1](../datasets/obb/dota-v2.md#dota-v10) 上训练的这些模型的使用示例，请参阅[旋转检测文档](../tasks/obb.md)，其中包含 15 个预训练类别。

        !!! note

            使用预训练权重 `yolov8n-obb.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试      | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|----------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict        | 0.52         | 0.51 \| 0.59       |                      |                         | 8       | 640                   |
        | FP32      | DOTAv1<sup>val</sup> | 0.76         |                    | 0.50                 | 0.36                    | 1       | 640                   |
        | FP16      | Predict        | 0.34         | 0.33 \| 0.42       |                      |                         | 8       | 640                   |
        | FP16      | DOTAv1<sup>val</sup> | 0.59         |                    | 0.50                 | 0.36                    | 1       | 640                   |
        | INT8      | Predict        | 0.29         | 0.28 \| 0.33       |                      |                         | 8       | 640                   |
        | INT8      | DOTAv1<sup>val</sup> | 0.32         |                    | 0.45                 | 0.32                    | 1       | 640                   |

### 消费级 GPU

!!! tip "检测性能（COCO）"

    === "RTX 3080 12 GB"

        使用 Windows 10.0.19045、`python 3.10.9`、`ultralytics==8.2.4`、`tensorrt==10.0.0b6` 测试

        !!! note

            使用预训练权重 `yolov8n.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 1.06         | 0.75 \| 1.88       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 1.37         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.62         | 0.75 \| 1.13       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 0.85         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.52         | 0.38 \| 1.00       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 0.74         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "RTX 3060 12 GB"

        使用 Windows 10.0.22631、`python 3.11.9`、`ultralytics==8.2.4`、`tensorrt==10.0.1` 测试

        !!! note

            使用预训练权重 `yolov8n.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 1.76         | 1.69 \| 1.87       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 1.94         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 0.86         | 0.75 \| 1.00       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 1.43         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.80         | 0.75 \| 1.00       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 1.35         |                    | 0.47                 | 0.33                    | 1       | 640                   |

    === "RTX 2060 6 GB"

        使用 Pop!_OS 22.04 LTS、`python 3.10.12`、`ultralytics==8.2.4`、`tensorrt==8.6.1.post1` 测试

        !!! note

            使用预训练权重 `yolov8n.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 2.84         | 2.84 \| 2.85       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 2.94         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 1.09         | 1.09 \| 1.10       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 1.20         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 0.75         | 0.74 \| 0.75       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 0.76         |                    | 0.47                 | 0.33                    | 1       | 640                   |

### 嵌入式设备

!!! tip "检测性能（COCO）"

    === "Jetson Orin NX 16GB"

        使用 JetPack 6.0 (L4T 36.3) Ubuntu 22.04.4 LTS、`python 3.10.12`、`ultralytics==8.2.16`、`tensorrt==10.0.1` 测试

        !!! note

            使用预训练权重 `yolov8n.engine` 显示每次测试的 `mean`、`min`（最快）和 `max`（最慢）推理时间

        | 精度 | 评估测试    | mean<br>(ms) | min \| max<br>(ms) | mAP<sup>val</sup><br>50(B) | mAP<sup>val</sup><br>50-95(B) | `batch` | 尺寸<br><sup>（像素）</sup> |
        |-----------|--------------|--------------|--------------------|----------------------|-------------------------|---------|-----------------------|
        | FP32      | Predict      | 6.11         | 6.10 \| 6.29       |                      |                         | 8       | 640                   |
        | FP32      | COCO<sup>val</sup> | 6.17         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | FP16      | Predict      | 3.18         | 3.18 \| 3.20       |                      |                         | 8       | 640                   |
        | FP16      | COCO<sup>val</sup> | 3.19         |                    | 0.52                 | 0.37                    | 1       | 640                   |
        | INT8      | Predict      | 2.30         | 2.29 \| 2.35       |                      |                         | 8       | 640                   |
        | INT8      | COCO<sup>val</sup> | 2.32         |                    | 0.46                 | 0.32                    | 1       | 640                   |

!!! info

    请参阅我们的 [NVIDIA Jetson 与 Ultralytics YOLO 快速入门指南](../guides/nvidia-jetson.md)，了解更多关于设置和配置的信息。

#### 评估方法

展开以下部分以获取有关这些模型如何导出和测试的信息。

??? example "导出配置"

    有关导出配置参数的详细信息，请参阅[导出模式](../modes/export.md)。

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    # TensorRT FP32
    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2)

    # TensorRT FP16
    out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

    # TensorRT INT8 带校准 `data`（即 COCO、ImageNet 或 DOTAv1，根据适当的模型任务）
    out = model.export(
        format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, workspace=2, int8=True, data="coco8.yaml"
    )
    ```

??? example "预测循环"

    有关更多信息，请参阅[预测模式](../modes/predict.md)。

    ```python
    import cv2

    from ultralytics import YOLO

    model = YOLO("yolov8n.engine")
    img = cv2.imread("path/to/image.jpg")

    for _ in range(100):
        result = model.predict(
            [img] * 8,  # 同一图像的 batch=8
            verbose=False,
            device="cuda",
        )
    ```

??? example "验证配置"

    有关验证配置参数的更多信息，请参阅 [`val` 模式](../modes/val.md)。

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.engine")
    results = model.val(
        data="data.yaml",  # COCO、ImageNet 或 DOTAv1，根据适当的模型任务
        batch=1,
        imgsz=640,
        verbose=False,
        device="cuda",
    )
    ```

## 部署导出的 YOLO11 TensorRT 模型

成功将 Ultralytics YOLO11 模型导出为 TensorRT 格式后，您现在可以部署它们了。有关在各种设置中部署 TensorRT 模型的深入说明，请查看以下资源：

- **[使用 Triton 服务器部署 Ultralytics](../guides/triton-inference-server.md)**：我们关于如何使用 NVIDIA 的 Triton 推理（以前称为 TensorRT 推理）服务器专门与 Ultralytics YOLO 模型配合使用的指南。

- **[使用 NVIDIA TensorRT 部署深度神经网络](https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/)**：本文解释了如何使用 NVIDIA TensorRT 在基于 GPU 的部署平台上高效部署深度神经网络。

- **[NVIDIA PC 的端到端 AI：NVIDIA TensorRT 部署](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-nvidia-tensorrt-deployment/)**：这篇博文解释了在 NVIDIA PC 上使用 NVIDIA TensorRT 优化和部署 AI 模型。

- **[NVIDIA TensorRT 的 GitHub 仓库：](https://github.com/NVIDIA/TensorRT)**：这是包含 NVIDIA TensorRT 源代码和文档的官方 GitHub 仓库。

## 总结

在本指南中，我们重点介绍了将 Ultralytics YOLO11 模型转换为 NVIDIA 的 TensorRT 模型格式。这一转换步骤对于提高 YOLO11 模型的效率和速度至关重要，使它们更有效且适合各种部署环境。

有关使用详情的更多信息，请查看 [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/)。

如果您对其他 Ultralytics YOLO11 集成感兴趣，我们的[集成指南页面](../integrations/index.md)提供了大量信息丰富的资源和见解。

## 常见问题

### 如何将 YOLO11 模型转换为 TensorRT 格式？

要将 Ultralytics YOLO11 模型转换为 TensorRT 格式以优化 NVIDIA GPU 推理，请按照以下步骤操作：

1. **安装所需的包**：

    ```bash
    pip install ultralytics
    ```

2. **导出您的 YOLO11 模型**：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.export(format="engine")  # 创建 'yolo11n.engine'

    # 运行推理
    model = YOLO("yolo11n.engine")
    results = model("https://ultralytics.com/images/bus.jpg")
    ```

有关更多详细信息，请访问 [YOLO11 安装指南](../quickstart.md)和[导出文档](../modes/export.md)。

### 使用 TensorRT 优化 YOLO11 模型有哪些好处？

使用 TensorRT 优化 YOLO11 模型提供多项好处：

- **更快的推理速度**：TensorRT 优化模型层并使用精度校准（INT8 和 FP16）来加速推理，而不会显著牺牲精度。
- **内存效率**：TensorRT 动态管理张量内存，减少开销并提高 GPU 内存利用率。
- **层融合**：将多个层组合成单个操作，减少计算复杂度。
- **内核自动调优**：自动为每个模型层选择优化的 GPU 内核，确保最大性能。

要了解更多信息，请探索 [NVIDIA 的官方 TensorRT 文档](https://developer.nvidia.com/tensorrt)和我们的[深入 TensorRT 概述](#tensorrt)。

### 我可以对 YOLO11 模型使用 TensorRT 的 INT8 量化吗？

是的，您可以使用 TensorRT 以 INT8 量化导出 YOLO11 模型。此过程涉及训练后量化（PTQ）和校准：

1. **使用 INT8 导出**：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="engine", batch=8, workspace=4, int8=True, data="coco.yaml")
    ```

2. **运行推理**：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.engine", task="detect")
    result = model.predict("https://ultralytics.com/images/bus.jpg")
    ```

有关更多详细信息，请参阅[使用 INT8 量化导出 TensorRT 部分](#使用-int8-量化导出-tensorrt)。

### 如何在 NVIDIA Triton 推理服务器上部署 YOLO11 TensorRT 模型？

可以使用以下资源在 NVIDIA Triton 推理服务器上部署 YOLO11 TensorRT 模型：

- **[使用 Triton 服务器部署 Ultralytics YOLOv8](../guides/triton-inference-server.md)**：设置和使用 Triton 推理服务器的分步指南。
- **[NVIDIA Triton 推理服务器文档](https://developer.nvidia.com/blog/deploying-deep-learning-nvidia-tensorrt/)**：详细部署选项和配置的官方 NVIDIA 文档。

这些指南将帮助您在各种部署环境中高效集成 YOLOv8 模型。

### 导出到 TensorRT 的 YOLOv8 模型观察到的性能改进是什么？

TensorRT 的性能改进可能因使用的硬件而异。以下是一些典型的基准测试：

- **NVIDIA A100**：
    - **FP32** 推理：~0.52 毫秒/图像
    - **FP16** 推理：~0.34 毫秒/图像
    - **INT8** 推理：~0.28 毫秒/图像
    - INT8 精度下 mAP 略有下降，但速度显著提高。

- **消费级 GPU（如 RTX 3080）**：
    - **FP32** 推理：~1.06 毫秒/图像
    - **FP16** 推理：~0.62 毫秒/图像
    - **INT8** 推理：~0.52 毫秒/图像

不同硬件配置的详细性能基准可在[性能部分](#ultralytics-yolo-tensorrt-导出性能)中找到。

有关 TensorRT 性能的更全面见解，请参阅 [Ultralytics 文档](../modes/export.md)和我们的性能分析报告。
