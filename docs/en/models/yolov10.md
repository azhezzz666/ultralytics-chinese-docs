---
comments: true
description: 探索 YOLOv10 实时目标检测，消除 NMS 并提高效率。以低计算成本实现顶级性能。
keywords: YOLOv10, 实时目标检测, 无NMS, 深度学习, 清华大学, Ultralytics, 机器学习, 神经网络, 性能优化
---

# YOLOv10：实时端到端目标检测

YOLOv10 由[清华大学](https://www.tsinghua.edu.cn/en/)的研究人员基于 [Ultralytics](https://www.ultralytics.com/) [Python 包](https://pypi.org/project/ultralytics/)构建，引入了一种新的实时目标检测方法，解决了之前 YOLO 版本中存在的后处理和模型架构缺陷。通过消除非极大值抑制 (NMS) 并优化各种模型组件，YOLOv10 以显著降低的计算开销实现了最先进的性能。大量实验证明了其在多个模型规模上的卓越精度-延迟权衡。

![YOLOv10 无 NMS 训练的一致双重分配](https://github.com/ultralytics/docs/releases/download/0/yolov10-consistent-dual-assignment.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics 在 SKU-110k 数据集上训练 YOLOv10 | 零售数据集
</p>

## 概述

实时目标检测旨在以低延迟准确预测图像中的目标类别和位置。由于 YOLO 系列在性能和效率之间取得了平衡，它一直处于这一研究的前沿。然而，对 NMS 的依赖和架构效率低下阻碍了最佳性能。YOLOv10 通过引入用于无 NMS 训练的[一致双重分配](https://arxiv.org/abs/2405.14458)和整体效率-精度驱动的模型设计策略来解决这些问题。

### 架构

YOLOv10 的架构建立在之前 YOLO 模型的优势之上，同时引入了几项关键创新。模型架构由以下组件组成：

1. **[主干网络](https://www.ultralytics.com/glossary/backbone)**：负责[特征提取](https://www.ultralytics.com/glossary/feature-extraction)，YOLOv10 中的主干网络使用增强版的 CSPNet（跨阶段部分网络）来改善梯度流并减少计算冗余。
2. **颈部**：颈部设计用于聚合来自不同尺度的特征并将其传递给头部。它包含 PAN（路径聚合网络）层以实现有效的多尺度特征融合。
3. **一对多头**：在训练期间为每个目标生成多个预测，以提供丰富的监督信号并提高学习精度。
4. **一对一头**：在推理期间为每个目标生成单个最佳预测，以消除对 NMS 的需求，从而减少延迟并提高效率。

## 关键特性

1. **无 NMS 训练**：利用一致双重分配消除对 NMS 的需求，减少[推理延迟](https://www.ultralytics.com/glossary/inference-latency)。
2. **整体模型设计**：从效率和精度两个角度对各种组件进行全面优化，包括轻量级分类头、空间-通道解耦下采样和基于秩的块设计。
3. **增强的模型能力**：融入大核[卷积](https://www.ultralytics.com/glossary/convolution)和部分自注意力模块，在不显著增加计算成本的情况下提高性能。

## 模型变体

YOLOv10 提供各种模型规模以满足不同的应用需求：

- **YOLOv10n**：适用于极端资源受限环境的纳米版本。
- **YOLOv10s**：平衡速度和精度的小型版本。
- **YOLOv10m**：通用中型版本。
- **YOLOv10b**：增加宽度以获得更高精度的平衡版本。
- **YOLOv10l**：以增加计算资源为代价获得更高精度的大型版本。
- **YOLOv10x**：追求最大精度和性能的超大型版本。

## 性能

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv10"]'></canvas>

YOLOv10 在精度和效率方面优于之前的 YOLO 版本和其他最先进的模型。例如，YOLOv10s 在 COCO 数据集上具有相似 AP 的情况下比 [RT-DETR-R18](../models/rtdetr.md) 快 1.8 倍，YOLOv10b 在相同性能下比 [YOLOv9-C](../models/yolov9.md) 延迟低 46%，参数少 25%。

!!! tip "性能"

    === "检测 (COCO)"

    延迟使用 TensorRT FP16 在 T4 GPU 上测量。

    | 模型          | 输入尺寸 | AP<sup>val</sup> | FLOPs (G) | 延迟 (ms) |
    | ------------- | -------- | ---------------- | --------- | --------- |
    | [YOLOv10n][1] | 640      | 38.5             | **6.7**   | **1.84**  |
    | [YOLOv10s][2] | 640      | 46.3             | 21.6      | 2.49      |
    | [YOLOv10m][3] | 640      | 51.1             | 59.1      | 4.74      |
    | [YOLOv10b][4] | 640      | 52.5             | 92.0      | 5.74      |
    | [YOLOv10l][5] | 640      | 53.2             | 120.3     | 7.28      |
    | [YOLOv10x][6] | 640      | **54.4**         | 160.4     | 10.70     |

## 方法论

### 无 NMS 训练的一致双重分配

YOLOv10 采用双重标签分配，在训练期间结合一对多和一对一策略，以确保丰富的监督和高效的端到端部署。一致匹配度量对齐了两种策略之间的监督，提高了[推理](../modes/predict.md)期间预测的质量。

### 整体效率-[精度](https://www.ultralytics.com/glossary/accuracy)驱动的模型设计

#### 效率增强

1. **轻量级分类头**：通过使用深度可分离卷积减少分类头的计算开销。
2. **空间-通道解耦下采样**：解耦空间缩减和通道调制，以最小化信息丢失和计算成本。
3. **基于秩的块设计**：根据固有阶段冗余调整块设计，确保最佳参数利用。

#### 精度增强

1. **大核卷积**：扩大[感受野](https://www.ultralytics.com/glossary/receptive-field)以增强特征提取能力。
2. **部分自注意力 (PSA)**：融入自注意力模块，以最小开销改善全局表示学习。

## 实验和结果

YOLOv10 已在 [COCO](../datasets/detect/coco.md) 等标准基准上进行了广泛测试，展示了卓越的性能和效率。该模型在不同变体上实现了最先进的结果，与之前版本和其他当代检测器相比，在延迟和精度方面展示了显著改进。

## 比较

![YOLOv10 与 SOTA 目标检测器的比较](https://github.com/ultralytics/docs/releases/download/0/yolov10-comparison-sota-detectors.avif)

与其他最先进检测器相比：

- YOLOv10s / x 在相似精度下比 RT-DETR-R18 / R101 快 1.8× / 1.3×
- YOLOv10b 在相同精度下比 YOLOv9-C 参数少 25%，延迟低 46%
- YOLOv10l / x 以 1.8× / 2.3× 更少的参数比 [YOLOv8l / x](../models/yolov8.md) 高出 0.3 AP / 0.5 AP

!!! tip "性能"

    === "检测 (COCO)"

        以下是 YOLOv10 变体与其他最先进模型的详细比较：

        | 模型              | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(G)</sup> | mAP<sup>val<br>50-95</sup> | 延迟<br><sup>(ms)</sup> | 前向延迟<br><sup>(ms)</sup> |
        | ----------------- | ------------------ | ----------------- | -------------------- | -------------------- | ---------------------------- |
        | YOLOv6-3.0-N      | 4.7                | 11.4              | 37.0                 | 2.69                 | **1.76**                     |
        | Gold-YOLO-N       | 5.6                | 12.1              | **39.6**             | 2.92                 | 1.82                         |
        | YOLOv8n           | 3.2                | 8.7               | 37.3                 | 6.16                 | 1.77                         |
        | **[YOLOv10n][1]** | **2.3**            | **6.7**           | 39.5                 | **1.84**             | 1.79                         |
        |                   |                    |                   |                      |                      |                              |
        | YOLOv6-3.0-S      | 18.5               | 45.3              | 44.3                 | 3.42                 | 2.35                         |
        | Gold-YOLO-S       | 21.5               | 46.0              | 45.4                 | 3.82                 | 2.73                         |
        | YOLOv8s           | 11.2               | 28.6              | 44.9                 | 7.07                 | **2.33**                     |
        | **[YOLOv10s][2]** | **7.2**            | **21.6**          | **46.8**             | **2.49**             | 2.39                         |
        |                   |                    |                   |                      |                      |                              |
        | RT-DETR-R18       | 20.0               | 60.0              | 46.5                 | **4.58**             | **4.49**                     |
        | YOLOv6-3.0-M      | 34.9               | 85.8              | 49.1                 | 5.63                 | 4.56                         |
        | Gold-YOLO-M       | 41.3               | 87.5              | 49.8                 | 6.38                 | 5.45                         |
        | YOLOv8m           | 25.9               | 78.9              | 50.6                 | 9.50                 | 5.09                         |
        | **[YOLOv10m][3]** | **15.4**           | **59.1**          | **51.3**             | 4.74                 | 4.63                         |
        |                   |                    |                   |                      |                      |                              |
        | YOLOv6-3.0-L      | 59.6               | 150.7             | 51.8                 | 9.02                 | 7.90                         |
        | Gold-YOLO-L       | 75.1               | 151.7             | 51.8                 | 10.65                | 9.78                         |
        | YOLOv8l           | 43.7               | 165.2             | 52.9                 | 12.39                | 8.06                         |
        | RT-DETR-R50       | 42.0               | 136.0             | 53.1                 | 9.20                 | 9.07                         |
        | **[YOLOv10l][5]** | **24.4**           | **120.3**         | **53.4**             | **7.28**             | **7.21**                     |
        |                   |                    |                   |                      |                      |                              |
        | YOLOv8x           | 68.2               | 257.8             | 53.9                 | 16.86                | 12.83                        |
        | RT-DETR-R101      | 76.0               | 259.0             | 54.3                 | 13.71                | 13.58                        |
        | **[YOLOv10x][6]** | **29.5**           | **160.4**         | **54.4**             | **10.70**            | **10.60**                    |

        [1]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt
        [2]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt
        [3]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt
        [4]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10b.pt
        [5]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10l.pt
        [6]: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt

## 使用示例

使用 YOLOv10 预测新图像：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv10n 模型
        model = YOLO("yolov10n.pt")

        # 对图像执行目标检测
        results = model("image.jpg")

        # 显示结果
        results[0].show()
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 YOLOv10n 模型并对 'bus.jpg' 图像运行推理
        yolo detect predict model=yolov10n.pt source=path/to/bus.jpg
        ```

在自定义数据集上训练 YOLOv10：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头开始加载 YOLOv10n 模型
        model = YOLO("yolov10n.yaml")

        # 训练模型
        model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从头开始构建 YOLOv10n 模型并在 COCO8 示例数据集上训练 100 个轮次
        yolo train model=yolov10n.yaml data=coco8.yaml epochs=100 imgsz=640

        # 从头开始构建 YOLOv10n 模型并对 'bus.jpg' 图像运行推理
        yolo predict model=yolov10n.yaml source=path/to/bus.jpg
        ```

## 支持的任务和模式

YOLOv10 模型系列提供一系列模型，每个模型都针对高性能[目标检测](../tasks/detect.md)进行了优化。这些模型满足不同的计算需求和精度要求，使其适用于广泛的应用。

| 模型    | 文件名                                                                | 任务                                   | 推理 | 验证 | 训练 | 导出 |
| ------- | --------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv10 | `yolov10n.pt` `yolov10s.pt` `yolov10m.pt` `yolov10l.pt` `yolov10x.pt` | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |

## 导出 YOLOv10

由于 YOLOv10 引入了新操作，并非 Ultralytics 提供的所有导出格式目前都受支持。下表列出了使用 Ultralytics 成功转换 YOLOv10 的格式。如果您能够[提供贡献更改](../help/contributing.md)以添加对 YOLOv10 其他格式的导出支持，欢迎提交拉取请求。

| 导出格式                                          | 导出支持 | 导出模型推理 | 备注                                                                           |
| ------------------------------------------------- | -------- | ------------ | ------------------------------------------------------------------------------ |
| [TorchScript](../integrations/torchscript.md)     | ✅       | ✅           | 标准 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型格式。        |
| [ONNX](../integrations/onnx.md)                   | ✅       | ✅           | 广泛支持的部署格式。                                                           |
| [OpenVINO](../integrations/openvino.md)           | ✅       | ✅           | 针对 Intel 硬件优化。                                                          |
| [TensorRT](../integrations/tensorrt.md)           | ✅       | ✅           | 针对 NVIDIA GPU 优化。                                                         |
| [CoreML](../integrations/coreml.md)               | ✅       | ✅           | 仅限 Apple 设备。                                                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | ✅       | ✅           | [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 的标准模型格式。 |
| [TF GraphDef](../integrations/tf-graphdef.md)     | ✅       | ✅           | 旧版 TensorFlow 格式。                                                         |
| [TF Lite](../integrations/tflite.md)              | ✅       | ✅           | 针对移动和嵌入式设备优化。                                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | ✅       | ✅           | 专用于 Google Edge TPU 设备。                                                  |
| [TF.js](../integrations/tfjs.md)                  | ✅       | ✅           | 用于浏览器的 JavaScript 环境。                                                 |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | ❌       | ❌           | 在中国流行；全球支持较少。                                                     |
| [NCNN](../integrations/ncnn.md)                   | ✅       | ❌           | 层 `torch.topk` 不存在或未注册                                                 |

## 结论

YOLOv10 通过解决之前 YOLO 版本的缺点并融入创新设计策略，为实时目标检测设立了新标准。其以低计算成本提供高精度的能力使其成为广泛[实际应用](https://www.ultralytics.com/solutions)的理想选择，包括[制造业](https://www.ultralytics.com/solutions/ai-in-manufacturing)、[零售业](https://www.ultralytics.com/blog/ai-in-fashion-retail)和[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)。

## 引用和致谢

我们要感谢来自[清华大学](https://www.tsinghua.edu.cn/en/)的 YOLOv10 作者对 [Ultralytics](https://www.ultralytics.com/) 框架的广泛研究和重大贡献：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{THU-MIGyolov10,
          title={YOLOv10: Real-Time End-to-End Object Detection},
          author={Ao Wang, Hui Chen, Lihao Liu, et al.},
          journal={arXiv preprint arXiv:2405.14458},
          year={2024},
          institution={Tsinghua University},
          license = {AGPL-3.0}
        }
        ```

有关详细实现、架构创新和实验结果，请参阅清华大学团队的 YOLOv10 [研究论文](https://arxiv.org/pdf/2405.14458)和 [GitHub 仓库](https://github.com/THU-MIG/yolov10)。

## 常见问题

### 什么是 YOLOv10，它与之前的 YOLO 版本有何不同？

YOLOv10 由[清华大学](https://www.tsinghua.edu.cn/en/)的研究人员开发，为实时目标检测引入了几项关键创新。它通过在训练期间采用一致双重分配消除了对非极大值抑制 (NMS) 的需求，并优化了模型组件以实现卓越性能和降低计算开销。有关其架构和关键特性的更多详细信息，请查看 [YOLOv10 概述](#概述)部分。

### 如何开始使用 YOLOv10 运行推理？

为了方便推理，您可以使用 Ultralytics YOLO Python 库或命令行界面 (CLI)。以下是使用 YOLOv10 预测新图像的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv10n 模型
        model = YOLO("yolov10n.pt")
        results = model("image.jpg")
        results[0].show()
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolov10n.pt source=path/to/image.jpg
        ```

有关更多使用示例，请访问我们的[使用示例](#使用示例)部分。

### YOLOv10 提供哪些模型变体及其用例？

YOLOv10 提供多种模型变体以满足不同用例：

- **YOLOv10n**：适用于极端资源受限环境
- **YOLOv10s**：平衡速度和精度
- **YOLOv10m**：通用用途
- **YOLOv10b**：增加宽度以获得更高精度
- **YOLOv10l**：以计算资源为代价获得高精度
- **YOLOv10x**：最大精度和性能

每个变体都针对不同的计算需求和精度要求设计，使其适用于各种应用。探索[模型变体](#模型变体)部分获取更多信息。

### YOLOv10 中的无 NMS 方法如何提高性能？

YOLOv10 通过采用一致双重分配进行训练，消除了推理期间对非极大值抑制 (NMS) 的需求。这种方法减少了推理延迟并提高了预测效率。该架构还包括用于推理的一对一头，确保每个目标获得单个最佳预测。有关详细解释，请参阅[无 NMS 训练的一致双重分配](#无-nms-训练的一致双重分配)部分。

### 在哪里可以找到 YOLOv10 模型的导出选项？

YOLOv10 支持多种导出格式，包括 TorchScript、ONNX、OpenVINO 和 TensorRT。但是，由于其新操作，并非 Ultralytics 提供的所有导出格式目前都支持 YOLOv10。有关支持格式和导出说明的详细信息，请访问[导出 YOLOv10](#导出-yolov10) 部分。

### YOLOv10 模型的性能基准是什么？

YOLOv10 在精度和效率方面优于之前的 YOLO 版本和其他最先进的模型。例如，YOLOv10s 在 COCO 数据集上具有相似 AP 的情况下比 RT-DETR-R18 快 1.8 倍。YOLOv10b 在相同性能下比 YOLOv9-C 延迟低 46%，参数少 25%。详细基准可在[比较](#比较)部分找到。
