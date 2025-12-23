---
comments: true
description: 学习如何将 Ultralytics YOLO11 模型导出为 Sony IMX500 格式，以便在配备片上处理功能的树莓派 AI 摄像头上高效部署边缘 AI。
keywords: Sony, IMX500, IMX 500, Atrios, MCT, 模型导出, 量化, 剪枝, 深度学习优化, 树莓派 AI 摄像头, 边缘 AI, PyTorch, IMX
---

# Ultralytics YOLO11 的 Sony IMX500 导出

本指南介绍如何将 Ultralytics YOLO11 模型导出并部署到配备 Sony IMX500 传感器的树莓派 AI 摄像头。

在计算能力有限的设备（如[树莓派 AI 摄像头](https://www.raspberrypi.com/products/ai-camera/)）上部署计算机视觉模型可能比较棘手。使用针对更快性能优化的模型格式会产生巨大差异。

IMX500 模型格式旨在以最小功耗实现神经网络的快速性能。它允许您优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以实现高速、低功耗推理。在本指南中，我们将引导您完成模型导出和部署到 IMX500 格式的过程，使您的模型更容易在[树莓派 AI 摄像头](https://www.raspberrypi.com/products/ai-camera/)上表现良好。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/ai-camera.avif" alt="树莓派 AI 摄像头">
</p>

## 为什么要导出为 IMX500？

Sony 的 [IMX500 智能视觉传感器](https://www.aitrios.sony-semicon.com/edge-ai-devices/raspberry-pi-ai-camera)是边缘 AI 处理领域的革命性硬件。它是世界上第一款具有片上 AI 功能的智能视觉传感器。该传感器有助于克服边缘 AI 中的许多挑战，包括数据处理瓶颈、隐私问题和性能限制。
与其他仅传递图像和帧的传感器不同，IMX500 能讲述完整的故事。它直接在传感器上处理数据，使设备能够实时生成洞察。

## Sony 的 YOLO11 模型 IMX500 导出

IMX500 旨在改变设备直接在传感器上处理数据的方式，无需将数据发送到云端进行处理。

IMX500 使用量化模型。量化使模型更小、更快，同时不会损失太多[精度](https://www.ultralytics.com/glossary/accuracy)。它非常适合边缘计算的有限资源，通过减少延迟并允许在本地快速处理数据（无需依赖云端），使应用程序能够快速响应。本地处理还能保护用户数据的隐私和安全，因为数据不会发送到远程服务器。

**IMX500 主要特性：**

- **元数据输出：** IMX500 不仅可以传输图像，还可以输出图像和元数据（推理结果），并且可以仅输出元数据以最小化数据大小、减少带宽和降低成本。
- **解决隐私问题：** 通过在设备上处理数据，IMX500 解决了隐私问题，非常适合以人为中心的应用，如人数统计和占用率跟踪。
- **实时处理：** 快速的片上处理支持实时决策，非常适合自动驾驶系统等边缘 AI 应用。

**开始之前：** 为获得最佳结果，请按照我们的[模型训练指南](https://docs.ultralytics.com/modes/train/)、[数据准备指南](https://docs.ultralytics.com/datasets/)和[超参数调优指南](https://docs.ultralytics.com/guides/hyperparameter-tuning/)确保您的 YOLO11 模型已做好导出准备。

## 支持的任务

目前，您只能将包含以下任务的模型导出为 IMX500 格式：

- [目标检测](https://docs.ultralytics.com/tasks/detect/)
- [姿态估计](https://docs.ultralytics.com/tasks/pose/)
- [图像分类](https://docs.ultralytics.com/tasks/classify/)
- [实例分割](https://docs.ultralytics.com/tasks/segment/)

## 使用示例

将 Ultralytics YOLO11 模型导出为 IMX500 格式，并使用导出的模型运行推理。

!!! note

    这里我们执行推理只是为了确保模型按预期工作。但是，要在树莓派 AI 摄像头上部署和推理，请跳转到[在部署中使用 IMX500 导出](#在部署中使用-imx500-导出)部分。

!!! example "目标检测"

    === "Python"

         ```python
         from ultralytics import YOLO

         # 加载 YOLO11n PyTorch 模型
         model = YOLO("yolo11n.pt")

         # 导出模型
         model.export(format="imx", data="coco8.yaml")  # 默认使用 PTQ 量化导出

         # 加载导出的模型
         imx_model = YOLO("yolo11n_imx_model")

         # 运行推理
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # 将 YOLO11n PyTorch 模型导出为带有训练后量化（PTQ）的 imx 格式
         yolo export model=yolo11n.pt format=imx data=coco8.yaml

         # 使用导出的模型运行推理
         yolo predict model=yolo11n_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

!!! example "姿态估计"

    === "Python"

         ```python
         from ultralytics import YOLO

         # 加载 YOLO11n-pose PyTorch 模型
         model = YOLO("yolo11n-pose.pt")

         # 导出模型
         model.export(format="imx", data="coco8-pose.yaml")  # 默认使用 PTQ 量化导出

         # 加载导出的模型
         imx_model = YOLO("yolo11n-pose_imx_model")

         # 运行推理
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # 将 YOLO11n-pose PyTorch 模型导出为带有训练后量化（PTQ）的 imx 格式
         yolo export model=yolo11n-pose.pt format=imx data=coco8-pose.yaml

         # 使用导出的模型运行推理
         yolo predict model=yolo11n-pose_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

!!! example "图像分类"

    === "Python"

         ```python
         from ultralytics import YOLO

         # 加载 YOLO11n-cls PyTorch 模型
         model = YOLO("yolo11n-cls.pt")

         # 导出模型
         model.export(format="imx", data="imagenet10")  # 默认使用 PTQ 量化导出

         # 加载导出的模型
         imx_model = YOLO("yolo11n-cls_imx_model")

         # 运行推理
         results = imx_model("https://ultralytics.com/images/bus.jpg", imgsz=224)
         ```

    === "CLI"

         ```bash
         # 将 YOLO11n-cls PyTorch 模型导出为带有训练后量化（PTQ）的 imx 格式
         yolo export model=yolo11n-cls.pt format=imx data=imagenet10

         # 使用导出的模型运行推理
         yolo predict model=yolo11n-cls_imx_model source='https://ultralytics.com/images/bus.jpg' imgsz=224
         ```

!!! example "实例分割"

    === "Python"

         ```python
         from ultralytics import YOLO

         # 加载 YOLO11n-seg PyTorch 模型
         model = YOLO("yolo11n-seg.pt")

         # 导出模型
         model.export(format="imx", data="coco8-seg.yaml")  # 默认使用 PTQ 量化导出

         # 加载导出的模型
         imx_model = YOLO("yolo11n-seg_imx_model")

         # 运行推理
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # 将 YOLO11n-seg PyTorch 模型导出为带有训练后量化（PTQ）的 imx 格式
         yolo export model=yolo11n-seg.pt format=imx data=coco8-seg.yaml

         # 使用导出的模型运行推理
         yolo predict model=yolo11n-seg_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

!!! warning

    Ultralytics 包会在运行时安装额外的导出依赖项。首次运行导出命令时，您可能需要重启控制台以确保其正常工作。

## 导出参数

| 参数       | 类型             | 默认值         | 描述                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'imx'`        | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                                                                                               |
| `imgsz`    | `int` 或 `tuple` | `640`          | 模型输入所需的图像尺寸。可以是正方形图像的整数，也可以是特定尺寸的元组 `(height, width)`。                                                                                                                                |
| `int8`     | `bool`           | `True`         | 激活 INT8 量化，进一步压缩模型并加速推理，同时[精度](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。                                                                    |
| `data`     | `str`            | `'coco8.yaml'` | [数据集](https://docs.ultralytics.com/datasets/)配置文件的路径（默认：`coco8.yaml`），对量化至关重要。                                                                                                            |
| `fraction` | `float`          | `1.0`          | 指定用于 INT8 量化校准的数据集比例。允许在完整数据集的子集上进行校准，适用于实验或资源有限的情况。如果启用 INT8 但未指定，将使用完整数据集。 |
| `device`   | `str`            | `None`         | 指定导出设备：GPU（`device=0`）、CPU（`device=cpu`）。                                                                                                                                                                                        |

!!! tip

    如果您在支持 CUDA 的 GPU 上导出，请传递参数 `device=0` 以加快导出速度。

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

导出过程将创建一个用于量化验证的 ONNX 模型，以及一个名为 `<model-name>_imx_model` 的目录。该目录将包含 `packerOut.zip` 文件，这对于将模型打包部署到 IMX500 硬件至关重要。此外，`<model-name>_imx_model` 文件夹还将包含一个文本文件（`labels.txt`），列出与模型关联的所有标签。

!!! example "文件夹结构"

    === "目标检测"

        ```bash
        yolo11n_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n_imx.onnx
        ├── yolo11n_imx_MemoryReport.json
        └── yolo11n_imx.pbtxt
        ```

    === "姿态估计"

        ```bash
        yolo11n-pose_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n-pose_imx.onnx
        ├── yolo11n-pose_imx_MemoryReport.json
        └── yolo11n-pose_imx.pbtxt
        ```

    === "图像分类"

        ```bash
        yolo11n-cls_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n-cls_imx.onnx
        ├── yolo11n-cls_imx_MemoryReport.json
        └── yolo11n-cls_imx.pbtxt
        ```

    === "实例分割"

        ```bash
        yolo11n-seg_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n-seg_imx.onnx
        ├── yolo11n-seg_imx_MemoryReport.json
        └── yolo11n-seg_imx.pbtxt
        ```

## 在部署中使用 IMX500 导出

将 Ultralytics YOLO11n 模型导出为 IMX500 格式后，可以将其部署到树莓派 AI 摄像头进行推理。

### 硬件先决条件

确保您拥有以下硬件：

1. 树莓派 5 或树莓派 4 Model B
2. 树莓派 AI 摄像头

将树莓派 AI 摄像头连接到树莓派上的 15 针 MIPI CSI 连接器，然后为树莓派通电。

### 软件先决条件

!!! note

    本指南已在运行于树莓派 5 上的 Raspberry Pi OS Bookworm 上测试通过。

步骤 1：打开终端窗口并执行以下命令，将树莓派软件更新到最新版本。

```bash
sudo apt update && sudo apt full-upgrade
```

步骤 2：安装 IMX500 固件，这是操作 IMX500 传感器所必需的。

```bash
sudo apt install imx500-all
```

步骤 3：重启树莓派以使更改生效。

```bash
sudo reboot
```

步骤 4：安装 [Aitrios 树莓派应用模块库](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library)

```bash
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

步骤 5：使用以下脚本运行 YOLO11 目标检测、姿态估计、分类和分割，这些脚本可在 [aitrios-rpi-application-module-library 示例](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/tree/main/examples/aicam)中找到。

!!! note

    在运行这些脚本之前，请确保根据您的环境替换 `model_file` 和 `labels.txt` 目录。

!!! example "Python 脚本"

    === "目标检测"

        ```python
        import numpy as np
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_od_yolo_ultralytics


        class YOLO(Model):
            """用于 IMX500 部署的 YOLO 模型。"""

            def __init__(self):
                """初始化用于 IMX500 部署的 YOLO 模型。"""
                super().__init__(
                    model_file="yolo11n_imx_model/packerOut.zip",  # 替换为正确的目录
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

                self.labels = np.genfromtxt(
                    "yolo11n_imx_model/labels.txt",  # 替换为正确的目录
                    dtype=str,
                    delimiter="\n",
                )

            def post_process(self, output_tensors):
                """对目标检测的输出张量进行后处理。"""
                return pp_od_yolo_ultralytics(output_tensors)


        device = AiCamera(frame_rate=16)  # AI 摄像头上运行 YOLO 模型的最佳帧率，可获得最大 DPS
        model = YOLO()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                detections = frame.detections[frame.detections.confidence > 0.55]
                labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

                annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
                frame.display()
        ```

    === "姿态估计"

        ```python
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_yolo_pose_ultralytics


        class YOLOPose(Model):
            """用于 IMX500 部署的 YOLO 姿态估计模型。"""

            def __init__(self):
                """初始化用于 IMX500 部署的 YOLO 姿态估计模型。"""
                super().__init__(
                    model_file="yolo11n-pose_imx_model/packerOut.zip",  # 替换为正确的目录
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

            def post_process(self, output_tensors):
                """对姿态估计的输出张量进行后处理。"""
                return pp_yolo_pose_ultralytics(output_tensors)


        device = AiCamera(frame_rate=17)  # AI 摄像头上运行 YOLO-pose 模型的最佳帧率，可获得最大 DPS
        model = YOLOPose()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                detections = frame.detections[frame.detections.confidence > 0.4]

                annotator.annotate_keypoints(frame, detections)
                annotator.annotate_boxes(frame, detections, corner_length=20)
                frame.display()
        ```

    === "图像分类"

        ```python
        import cv2
        import numpy as np
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_cls


        class YOLOClassification(Model):
            """用于 IMX500 部署的 YOLO 分类模型。"""

            def __init__(self):
                """初始化用于 IMX500 部署的 YOLO 分类模型。"""
                super().__init__(
                    model_file="yolo11n-cls_imx_model/packerOut.zip",  # 替换为正确的目录
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

                self.labels = np.genfromtxt("yolo11n-cls_imx_model/labels.txt", dtype=str, delimiter="\n")

            def post_process(self, output_tensors):
                """对分类的输出张量进行后处理。"""
                return pp_cls(output_tensors)


        device = AiCamera()
        model = YOLOClassification()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                for i, label in enumerate([model.labels[id] for id in frame.detections.class_id[:3]]):
                    text = f"{i + 1}. {label}: {frame.detections.confidence[i]:.2f}"
                    cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

                frame.display()
        ```

    === "实例分割"

        ```python
        import numpy as np
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_yolo_segment_ultralytics


        class YOLOSegment(Model):
            """用于 IMX500 部署的 YOLO 分割模型。"""

            def __init__(self):
                """初始化用于 IMX500 部署的 YOLO 分割模型。"""
                super().__init__(
                    model_file="yolo11n-seg_imx_model/packerOut.zip",  # 替换为正确的目录
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

                self.labels = np.genfromtxt(
                    "yolo11n-seg_imx_model/labels.txt",  # 替换为正确的目录
                    dtype=str,
                    delimiter="\n",
                )

            def post_process(self, output_tensors):
                """对实例分割的输出张量进行后处理。"""
                return pp_yolo_segment_ultralytics(output_tensors)


        device = AiCamera(frame_rate=17)  # AI 摄像头上运行 YOLO-seg 模型的最佳帧率，可获得最大 DPS
        model = YOLOSegment()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                detections = frame.detections[frame.detections.confidence > 0.4]

                labels = [f"{model.labels[c]}" for m, c, s, _, _ in detections]
                annotator.annotate_instance_segments(frame, detections)
                annotator.annotate_boxes(frame, detections, labels=labels)
                frame.display()
        ```

## 基准测试

以下 YOLOv8n、YOLO11n、YOLOv8n-pose、YOLO11n-pose、YOLOv8n-cls 和 YOLO11n-cls 基准测试由 Ultralytics 团队在树莓派 AI 摄像头上使用 `imx` 模型格式进行，测量速度和精度。

| 模型         | 格式 | 尺寸（像素） | `packerOut.zip` 大小（MB） | mAP50-95(B) | 推理时间（毫秒/图像） |
| ------------ | ------ | ------------- | ---------------------------- | ----------- | ---------------------- |
| YOLOv8n      | imx    | 640           | 2.1                          | 0.470       | 58.79                  |
| YOLO11n      | imx    | 640           | 2.2                          | 0.517       | 58.82                  |
| YOLOv8n-pose | imx    | 640           | 2.0                          | 0.687       | 58.79                  |
| YOLO11n-pose | imx    | 640           | 2.1                          | 0.788       | 62.50                  |

| 模型        | 格式 | 尺寸（像素） | `packerOut.zip` 大小（MB） | acc (top1) | acc (top5) | 推理时间（毫秒/图像） |
| ----------- | ------ | ------------- | ---------------------------- | ---------- | ---------- | ---------------------- |
| YOLOv8n-cls | imx    | 224           | 2.3                          | 0.25       | 0.5        | 33.31                  |
| YOLO11n-cls | imx    | 224           | 2.3                          | 0.25       | 0.417      | 33.31                  |

!!! note

    上述基准测试的验证使用 COCO128 数据集进行检测模型验证，COCO8-Pose 数据集进行姿态估计模型验证，ImageNet10 进行分类模型验证。

## 底层原理

<p align="center">
  <img width="640" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/imx500-deploy.avif" alt="IMX500 部署">
</p>

### Sony 模型压缩工具包（MCT）

[Sony 的模型压缩工具包（MCT）](https://github.com/SonySemiconductorSolutions/mct-model-optimization)是一个强大的工具，用于通过量化和剪枝优化深度学习模型。它支持各种量化方法，并提供先进的算法来减少模型大小和计算复杂度，同时不会显著牺牲精度。MCT 对于在资源受限的设备上部署模型特别有用，可确保高效推理和减少延迟。

### MCT 支持的功能

Sony 的 MCT 提供了一系列旨在优化神经网络模型的功能：

1. **图优化：** 通过将批量归一化等层折叠到前面的层中，将模型转换为更高效的版本。
2. **量化参数搜索：** 使用均方误差、无裁剪和平均绝对误差等指标最小化量化噪声。
3. **高级量化算法：**
    - **负值偏移校正：** 解决对称激活量化带来的性能问题。
    - **异常值过滤：** 使用 z-score 检测和移除异常值。
    - **聚类：** 利用非均匀量化网格实现更好的分布匹配。
    - **混合精度搜索：** 根据敏感度为每层分配不同的量化位宽。
4. **可视化：** 使用 TensorBoard 观察模型性能洞察、量化阶段和位宽配置。

#### 量化

MCT 支持多种量化方法来减少模型大小并提高推理速度：

1. **训练后量化（PTQ）：**
    - 通过 Keras 和 PyTorch API 提供。
    - 复杂度：低
    - 计算成本：低（CPU 分钟级）
2. **基于梯度的训练后量化（GPTQ）：**
    - 通过 Keras 和 PyTorch API 提供。
    - 复杂度：中等
    - 计算成本：中等（2-3 GPU 小时）
3. **量化感知训练（QAT）：**
    - 复杂度：高
    - 计算成本：高（12-36 GPU 小时）

MCT 还支持各种权重和激活的量化方案：

1. 二的幂次方（硬件友好）
2. 对称
3. 均匀

#### 结构化剪枝

MCT 引入了针对特定硬件架构设计的结构化、硬件感知模型剪枝。该技术利用目标平台的单指令多数据（SIMD）功能，通过剪枝 SIMD 组来实现。这减少了模型大小和复杂度，同时优化了通道利用率，与 SIMD 架构对齐以实现权重内存占用的目标资源利用。通过 Keras 和 PyTorch API 提供。

### IMX500 转换器工具（编译器）

IMX500 转换器工具是 IMX500 工具集的重要组成部分，允许编译模型以部署到 Sony 的 IMX500 传感器（例如树莓派 AI 摄像头）。该工具促进了通过 Ultralytics 软件处理的 Ultralytics YOLO11 模型的转换，确保它们与指定硬件兼容并高效运行。模型量化后的导出过程涉及生成封装基本数据和设备特定配置的二进制文件，简化了在树莓派 AI 摄像头上的部署过程。

## 实际应用案例

导出为 IMX500 格式在各行业都有广泛的适用性。以下是一些示例：

- **边缘 AI 和物联网：** 在无人机或安防摄像头上启用目标检测，其中在低功耗设备上进行实时处理至关重要。
- **可穿戴设备：** 在健康监测可穿戴设备上部署针对小规模 AI 处理优化的模型。
- **智慧城市：** 使用 IMX500 导出的 YOLO11 模型进行交通监控和安全分析，处理速度更快，延迟最小。
- **零售分析：** 通过在销售点系统或智能货架中部署优化模型来增强店内监控。

## 总结

将 Ultralytics YOLO11 模型导出为 Sony 的 IMX500 格式允许您在基于 IMX500 的摄像头上部署模型进行高效推理。通过利用先进的量化技术，您可以减少模型大小并提高推理速度，同时不会显著影响精度。

有关更多信息和详细指南，请参阅 Sony 的 [IMX500 网站](https://www.aitrios.sony-semicon.com/edge-ai-devices/raspberry-pi-ai-camera)。

## 常见问题

### 如何将 YOLO11 模型导出为 IMX500 格式用于树莓派 AI 摄像头？

要将 YOLO11 模型导出为 IMX500 格式，请使用 Python API 或 CLI 命令：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="imx")  # 默认使用 PTQ 量化导出
```

导出过程将创建一个包含部署所需文件的目录，包括 `packerOut.zip`。

### 使用 IMX500 格式进行边缘 AI 部署有哪些主要优势？

IMX500 格式为边缘部署提供了几个重要优势：

- 片上 AI 处理减少延迟和功耗
- 输出图像和元数据（推理结果），而不仅仅是图像
- 通过本地处理数据增强隐私，无需依赖云端
- 实时处理能力，非常适合时间敏感的应用
- 优化的量化，可在资源受限的设备上高效部署模型

### IMX500 部署需要哪些硬件和软件先决条件？

部署 IMX500 模型需要：

硬件：

- 树莓派 5 或树莓派 4 Model B
- 配备 IMX500 传感器的树莓派 AI 摄像头

软件：

- Raspberry Pi OS Bookworm
- IMX500 固件和工具（`sudo apt install imx500-all`）

### YOLO11 模型在 IMX500 上可以期望什么性能？

根据 Ultralytics 在树莓派 AI 摄像头上的基准测试：

- YOLO11n 每张图像的推理时间为 62.50 毫秒
- 在 COCO128 数据集上的 mAP50-95 为 0.492
- 量化后模型大小仅为 3.2MB

这表明 IMX500 格式为边缘 AI 应用提供了高效的实时推理，同时保持良好的精度。
