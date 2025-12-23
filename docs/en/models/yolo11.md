---
comments: true
description: 探索 YOLO11，最新的先进目标检测技术，为各种计算机视觉任务提供无与伦比的精度和效率。
keywords: YOLO11, 先进目标检测, YOLO 系列, Ultralytics, 计算机视觉, AI, 机器学习, 深度学习
---

# Ultralytics YOLO11

## 概述

YOLO11 是 [Ultralytics](https://www.ultralytics.com/) YOLO 系列实时目标检测器的最新迭代版本，重新定义了尖端[精度](https://www.ultralytics.com/glossary/accuracy)、速度和效率的可能性。在之前 YOLO 版本令人印象深刻的进步基础上，YOLO11 在架构和训练方法上引入了重大改进，使其成为各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务的多功能选择。

![Ultralytics YOLO11 对比图](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)

<div style="text-align: center">
    <audio controls preload="none" style="width:100%; max-width:1920px;">
      <source src="https://github.com/ultralytics/docs/releases/download/0/Ultralytics-YOLO11-podcast-notebook.LM.mp3" type="audio/mpeg">
      您的浏览器不支持音频元素。
    </audio>
    <p>Ultralytics YOLO11 🚀 播客由 <a href="https://notebooklm.google/">NotebookLM</a> 生成</p>
</div>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/-JXwa-WlkU8"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO11 进行目标检测和跟踪 | 如何进行基准测试 | YOLO11 发布🚀
</p>

## 关键特性

- **增强的特征提取：** YOLO11 采用改进的[骨干网络](https://www.ultralytics.com/glossary/backbone)和颈部架构，增强了[特征提取](https://www.ultralytics.com/glossary/feature-extraction)能力，实现更精确的目标检测和复杂任务性能。
- **优化的效率和速度：** YOLO11 引入了精细化的架构设计和优化的训练流程，提供更快的处理速度，同时保持精度和性能之间的最佳平衡。
- **更少参数实现更高精度：** 通过模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP），同时使用的参数比 YOLOv8m 少 22%，使其在不影响精度的情况下具有计算效率。
- **跨环境适应性：** YOLO11 可以无缝部署在各种环境中，包括边缘设备、云平台和支持 NVIDIA GPU 的系统，确保最大的灵活性。
- **广泛的任务支持：** 无论是目标检测、实例分割、图像分类、姿态估计还是旋转目标检测（OBB），YOLO11 都能满足各种计算机视觉挑战。

## 支持的任务和模式

YOLO11 建立在早期 Ultralytics YOLO 版本建立的多功能模型范围之上，提供跨各种计算机视觉任务的增强支持：

| 模型        | 文件名                                                                                    | 任务                                   | 推理 | 验证 | 训练 | 导出 |
| ----------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLO11      | `yolo11n.pt` `yolo11s.pt` `yolo11m.pt` `yolo11l.pt` `yolo11x.pt`                          | [检测](../tasks/detect.md)             | ✅   | ✅   | ✅   | ✅   |
| YOLO11-seg  | `yolo11n-seg.pt` `yolo11s-seg.pt` `yolo11m-seg.pt` `yolo11l-seg.pt` `yolo11x-seg.pt`      | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLO11-pose | `yolo11n-pose.pt` `yolo11s-pose.pt` `yolo11m-pose.pt` `yolo11l-pose.pt` `yolo11x-pose.pt` | [姿态/关键点](../tasks/pose.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLO11-obb  | `yolo11n-obb.pt` `yolo11s-obb.pt` `yolo11m-obb.pt` `yolo11l-obb.pt` `yolo11x-obb.pt`      | [旋转检测](../tasks/obb.md)            | ✅   | ✅   | ✅   | ✅   |
| YOLO11-cls  | `yolo11n-cls.pt` `yolo11s-cls.pt` `yolo11m-cls.pt` `yolo11l-cls.pt` `yolo11x-cls.pt`      | [分类](../tasks/classify.md)           | ✅   | ✅   | ✅   | ✅   |

此表提供了 YOLO11 模型变体的概述，展示了它们在特定任务中的适用性以及与推理、验证、训练和导出等操作模式的兼容性。这种灵活性使 YOLO11 适用于计算机视觉中的广泛应用，从实时检测到复杂的分割任务。


## 性能指标

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO11"]'></canvas>

!!! tip "性能"

    === "检测 (COCO)"

        查看[检测文档](../tasks/detect.md)了解在 [COCO](../datasets/detect/coco.md) 上训练的这些模型的使用示例，其中包含 80 个预训练类别。

        --8<-- "docs/macros/yolo-det-perf.md"

    === "分割 (COCO)"

        查看[分割文档](../tasks/segment.md)了解在 [COCO](../datasets/segment/coco.md) 上训练的这些模型的使用示例，其中包含 80 个预训练类别。

        --8<-- "docs/macros/yolo-seg-perf.md"

    === "分类 (ImageNet)"

        查看[分类文档](../tasks/classify.md)了解在 [ImageNet](../datasets/classify/imagenet.md) 上训练的这些模型的使用示例，其中包含 1000 个预训练类别。

        --8<-- "docs/macros/yolo-cls-perf.md"

    === "姿态 (COCO)"

        查看[姿态估计文档](../tasks/pose.md)了解在 [COCO](../datasets/pose/coco.md) 上训练的这些模型的使用示例，其中包含 1 个预训练类别 'person'。

        --8<-- "docs/macros/yolo-pose-perf.md"

    === "OBB (DOTAv1)"

        查看[旋转检测文档](../tasks/obb.md)了解在 [DOTAv1](../datasets/obb/dota-v2.md#dota-v10) 上训练的这些模型的使用示例，其中包含 15 个预训练类别。

        --8<-- "docs/macros/yolo-obb-perf.md"

## 使用示例

本节提供简单的 YOLO11 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

请注意，以下示例是针对用于[目标检测](https://www.ultralytics.com/glossary/object-detection)的 YOLO11 [检测](../tasks/detect.md)模型。有关其他支持的任务，请参阅[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[OBB](../tasks/obb.md) 和[姿态](../tasks/pose.md)文档。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLO11n 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLO11n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLO11n 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolo11n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

!!! tip "Ultralytics YOLO11 出版物"

    由于模型快速发展的特性，Ultralytics 尚未发布 YOLO11 的正式研究论文。我们专注于推进技术并使其更易于使用，而不是制作静态文档。有关 YOLO 架构、特性和使用的最新信息，请参阅我们的 [GitHub 仓库](https://github.com/ultralytics/ultralytics)和[文档](https://docs.ultralytics.com/)。

如果您在工作中使用 YOLO11 或此仓库中的任何其他软件，请使用以下格式引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo11_ultralytics,
          author = {Glenn Jocher and Jing Qiu},
          title = {Ultralytics YOLO11},
          version = {11.0.0},
          year = {2024},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

请注意，DOI 正在申请中，一旦可用将添加到引用中。YOLO11 模型根据 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 和[企业](https://www.ultralytics.com/license)许可证提供。

## 常见问题

### 与之前版本相比，Ultralytics YOLO11 有哪些关键改进？

Ultralytics YOLO11 相比其前身引入了几项重大进步。关键改进包括：

- **增强的特征提取：** YOLO11 采用改进的骨干网络和颈部架构，增强了[特征提取](https://www.ultralytics.com/glossary/feature-extraction)能力，实现更精确的目标检测。
- **优化的效率和速度：** 精细化的架构设计和优化的训练流程提供更快的处理速度，同时保持精度和性能之间的平衡。
- **更少参数实现更高精度：** YOLO11m 在 COCO 数据集上实现了更高的平均[精度](https://www.ultralytics.com/glossary/precision)均值（mAP），同时使用的参数比 YOLOv8m 少 22%，使其在不影响精度的情况下具有计算效率。
- **跨环境适应性：** YOLO11 可以部署在各种环境中，包括边缘设备、云平台和支持 NVIDIA GPU 的系统。
- **广泛的任务支持：** YOLO11 支持多种计算机视觉任务，如目标检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、图像分类、姿态估计和旋转目标检测（OBB）。

### 如何训练 YOLO11 模型进行目标检测？

可以使用 Python 或 CLI 命令训练 YOLO11 模型进行目标检测。以下是两种方法的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 YOLO11n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

有关更详细的说明，请参阅[训练](../modes/train.md)文档。

### YOLO11 模型可以执行哪些任务？

YOLO11 模型功能多样，支持广泛的计算机视觉任务，包括：

- **目标检测：** 识别和定位图像中的对象。
- **实例分割：** 检测对象并描绘其边界。
- **[图像分类](https://www.ultralytics.com/glossary/image-classification)：** 将图像分类到预定义的类别中。
- **姿态估计：** 检测和跟踪人体上的关键点。
- **旋转目标检测（OBB）：** 检测带有旋转的对象以获得更高的精度。

有关每个任务的更多信息，请参阅[检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和[旋转检测](../tasks/obb.md)文档。

### YOLO11 如何用更少的参数实现更高的精度？

YOLO11 通过模型设计和优化技术的进步，用更少的参数实现了更高的精度。改进的架构允许高效的特征提取和处理，从而在 COCO 等数据集上实现更高的平均精度均值（mAP），同时使用的参数比 YOLOv8m 少 22%。这使得 YOLO11 在不影响精度的情况下具有计算效率，适合部署在资源受限的设备上。

### YOLO11 可以部署在边缘设备上吗？

是的，YOLO11 设计为可在各种环境中适应，包括边缘设备。其优化的架构和高效的处理能力使其适合部署在边缘设备、云平台和支持 NVIDIA GPU 的系统上。这种灵活性确保 YOLO11 可以用于各种应用，从移动设备上的实时检测到云环境中的复杂分割任务。有关部署选项的更多详细信息，请参阅[导出](../modes/export.md)文档。
