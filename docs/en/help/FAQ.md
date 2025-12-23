---
comments: true
description: 探索与 Ultralytics YOLO 相关的常见问题和解决方案，从硬件要求到模型微调和实时检测。
keywords: Ultralytics, YOLO, FAQ, 目标检测, 硬件要求, 微调, ONNX, TensorFlow, 实时检测, 模型准确率
---

# Ultralytics YOLO 常见问题 (FAQ)

本 FAQ 部分解答用户在使用 [Ultralytics](https://www.ultralytics.com/) YOLO 仓库时可能遇到的常见问题和问题。

## 常见问题

### 什么是 Ultralytics，它提供什么？

Ultralytics 是一家专注于最先进目标检测和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)模型的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv) AI 公司，重点关注 YOLO（You Only Look Once）系列。他们的产品包括：

- [YOLO11](https://docs.ultralytics.com/models/yolo11/)（最新版）和 [YOLOv8](https://docs.ultralytics.com/models/yolov8/)（上一代）的开源实现
- 用于各种计算机视觉任务的广泛[预训练模型](https://docs.ultralytics.com/models/)
- 用于将 YOLO 模型无缝集成到项目中的综合 [Python 包](https://docs.ultralytics.com/usage/python/)
- 用于训练、测试和部署模型的多功能[工具](https://docs.ultralytics.com/modes/)
- [详尽的文档](https://docs.ultralytics.com/)和支持性社区

### 如何安装 Ultralytics 包？

使用 pip 安装 Ultralytics 包非常简单：

```
pip install ultralytics
```

要获取最新的开发版本，请直接从 GitHub 仓库安装：

```
pip install git+https://github.com/ultralytics/ultralytics.git
```

详细的安装说明可以在[快速入门指南](https://docs.ultralytics.com/quickstart/)中找到。

### 运行 Ultralytics 模型的系统要求是什么？

最低要求：

- Python 3.8+
- [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8+
- 兼容 CUDA 的 GPU（用于 GPU 加速）

推荐配置：

- Python 3.8+
- PyTorch 1.10+
- 带有 CUDA 11.2+ 的 NVIDIA GPU
- 8GB+ RAM
- 50GB+ 可用磁盘空间（用于数据集存储和模型训练）

有关常见问题的故障排除，请访问 [YOLO 常见问题](https://docs.ultralytics.com/guides/yolo-common-issues/)页面。

### 如何在自己的数据集上训练自定义 YOLO 模型？

要训练自定义 YOLO 模型：

1. 以 YOLO 格式准备数据集（图像和相应的标签 txt 文件）。
2. 创建描述数据集结构和类别的 YAML 文件。
3. 使用以下 Python 代码开始训练：

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo11n.yaml")  # 从头构建新模型
    model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

    # 训练模型
    results = model.train(data="path/to/your/data.yaml", epochs=100, imgsz=640)
    ```

有关更深入的指南，包括数据准备和高级训练选项，请参阅综合[训练指南](https://docs.ultralytics.com/modes/train/)。

### Ultralytics 提供哪些预训练模型？

Ultralytics 为各种任务提供多种预训练模型：

- 目标检测：YOLO11n、YOLO11s、YOLO11m、YOLO11l、YOLO11x
- [实例分割](https://www.ultralytics.com/glossary/instance-segmentation)：YOLO11n-seg、YOLO11s-seg、YOLO11m-seg、YOLO11l-seg、YOLO11x-seg
- 分类：YOLO11n-cls、YOLO11s-cls、YOLO11m-cls、YOLO11l-cls、YOLO11x-cls
- 姿态估计：YOLO11n-pose、YOLO11s-pose、YOLO11m-pose、YOLO11l-pose、YOLO11x-pose

这些模型在大小和复杂性上各不相同，在速度和[准确率](https://www.ultralytics.com/glossary/accuracy)之间提供不同的权衡。探索完整的[预训练模型](https://docs.ultralytics.com/models/)范围，找到最适合您项目的模型。

### 如何使用训练好的 Ultralytics 模型进行推理？

要使用训练好的模型进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("path/to/your/model.pt")

# 执行推理
results = model("path/to/image.jpg")

# 处理结果
for r in results:
    print(r.boxes)  # 打印边界框预测
    print(r.masks)  # 打印掩码预测
    print(r.probs)  # 打印类别概率
```

有关高级推理选项，包括批处理和视频推理，请查看详细的[预测指南](https://docs.ultralytics.com/modes/predict/)。

### Ultralytics 模型可以部署在边缘设备或生产环境中吗？

当然可以！Ultralytics 模型设计用于跨各种平台的多功能部署：

- 边缘设备：使用 TensorRT、ONNX 或 OpenVINO 在 NVIDIA Jetson 或 Intel Neural Compute Stick 等设备上优化推理。
- 移动端：通过将模型转换为 TFLite 或 Core ML 在 Android 或 iOS 设备上部署。
- 云端：利用 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Serving 或 PyTorch Serve 等框架进行可扩展的云部署。
- Web：使用 ONNX.js 或 TensorFlow.js 实现浏览器内推理。

Ultralytics 提供导出功能，可将模型转换为各种部署格式。探索广泛的[部署选项](https://docs.ultralytics.com/guides/model-deployment-options/)，找到最适合您用例的解决方案。

### YOLOv8 和 YOLO11 有什么区别？

主要区别包括：

- 架构：YOLO11 具有改进的骨干网络和头部设计，以增强性能。
- 性能：与 YOLOv8 相比，YOLO11 通常提供更高的准确率和速度。
- 效率：YOLO11m 在 COCO 数据集上实现更高的平均精度 (mAP)，参数比 YOLOv8m 少 22%。
- 任务：两种模型都在统一框架中支持[目标检测](https://www.ultralytics.com/glossary/object-detection)、实例分割、分类和姿态估计。
- 代码库：YOLO11 采用更模块化和可扩展的架构实现，便于更轻松的定制和扩展。

有关功能和性能指标的深入比较，请访问 [YOLO11 文档页面](https://docs.ultralytics.com/models/yolo11/)。

### 如何为 Ultralytics 开源项目做贡献？

为 Ultralytics 做贡献是改进项目和扩展技能的好方法。以下是参与方式：

1. 在 GitHub 上 Fork Ultralytics 仓库。
2. 为您的功能或错误修复创建新分支。
3. 进行更改并确保所有测试通过。
4. 提交拉取请求，清楚描述您的更改。
5. 参与代码审查过程。

您还可以通过报告错误、建议功能或改进文档来做贡献。有关详细指南和最佳实践，请参阅[贡献指南](https://docs.ultralytics.com/help/contributing/)。

### 如何在 Python 中安装 Ultralytics 包？

在 Python 中安装 Ultralytics 包很简单。在终端或命令提示符中运行以下命令使用 pip：

```bash
pip install ultralytics
```

要获取最新的开发版本，请直接从 GitHub 仓库安装：

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

有关特定环境的安装说明和故障排除提示，请参阅综合[快速入门指南](https://docs.ultralytics.com/quickstart/)。

### Ultralytics YOLO 的主要功能是什么？

Ultralytics YOLO 拥有丰富的高级计算机视觉任务功能：

- 实时检测：在实时场景中高效检测和分类目标。
- 多任务能力：使用统一框架执行目标检测、实例分割、分类和姿态估计。
- 预训练模型：访问各种[预训练模型](https://docs.ultralytics.com/models/)，在速度和准确率之间为不同用例提供平衡。
- 自定义训练：使用灵活的[训练管道](https://docs.ultralytics.com/modes/train/)轻松在自定义数据集上微调模型。
- 广泛的[部署选项](https://docs.ultralytics.com/guides/model-deployment-options/)：将模型导出为 TensorRT、ONNX 和 CoreML 等各种格式，以便跨不同平台部署。
- 详尽的文档：受益于全面的[文档](https://docs.ultralytics.com/)和支持性社区，指导您完成计算机视觉之旅。

### 如何提高 YOLO 模型的性能？

可以通过多种技术增强 YOLO 模型的性能：

1. [超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)：使用[超参数调优指南](https://docs.ultralytics.com/guides/hyperparameter-tuning/)尝试不同的超参数以优化模型性能。
2. [数据增强](https://www.ultralytics.com/glossary/data-augmentation)：实施翻转、缩放、旋转和颜色调整等技术来增强训练数据集并提高模型泛化能力。
3. [迁移学习](https://www.ultralytics.com/glossary/transfer-learning)：利用预训练模型并使用[训练指南](../modes/train.md)在特定数据集上进行微调。
4. 导出为高效格式：使用[导出指南](../modes/export.md)将模型转换为 TensorRT 或 ONNX 等优化格式以加快推理速度。
5. 基准测试：利用[基准模式](https://docs.ultralytics.com/modes/benchmark/)系统地测量和提高推理速度和准确率。

### 我可以在移动和边缘设备上部署 Ultralytics YOLO 模型吗？

是的，Ultralytics YOLO 模型设计用于多功能部署，包括移动和边缘设备：

- 移动端：将模型转换为 TFLite 或 CoreML，以无缝集成到 Android 或 iOS 应用中。有关特定平台的说明，请参阅 [TFLite 集成指南](https://docs.ultralytics.com/integrations/tflite/)和 [CoreML 集成指南](https://docs.ultralytics.com/integrations/coreml/)。
- 边缘设备：使用 TensorRT 或 ONNX 在 NVIDIA Jetson 或其他边缘硬件上优化推理。[Edge TPU 集成指南](https://docs.ultralytics.com/integrations/edge-tpu/)提供了边缘部署的详细步骤。

有关跨各种平台的部署策略的全面概述，请参阅[部署选项指南](https://docs.ultralytics.com/guides/model-deployment-options/)。

### 如何使用训练好的 Ultralytics YOLO 模型进行推理？

使用训练好的 Ultralytics YOLO 模型进行推理非常简单：

1. 加载模型：

    ```python
    from ultralytics import YOLO

    model = YOLO("path/to/your/model.pt")
    ```

2. 运行推理：

    ```python
    results = model("path/to/image.jpg")

    for r in results:
        print(r.boxes)  # 打印边界框预测
        print(r.masks)  # 打印掩码预测
        print(r.probs)  # 打印类别概率
    ```

有关高级推理技术，包括批处理、视频推理和自定义预处理，请参阅详细的[预测指南](https://docs.ultralytics.com/modes/predict/)。

### 在哪里可以找到使用 Ultralytics 的示例和教程？

Ultralytics 提供丰富的资源帮助您入门并掌握其工具：

- 📚 [官方文档](https://docs.ultralytics.com/)：全面的指南、API 参考和最佳实践。
- 💻 [GitHub 仓库](https://github.com/ultralytics/ultralytics)：源代码、示例脚本和社区贡献。
- ✍️ [Ultralytics 博客](https://www.ultralytics.com/blog)：深入的文章、用例和技术见解。
- 💬 [社区论坛](https://community.ultralytics.com/)：与其他用户联系、提问和分享经验。
- 🎥 [YouTube 频道](https://www.youtube.com/ultralytics?sub_confirmation=1)：关于各种 Ultralytics 主题的视频教程、演示和网络研讨会。

这些资源提供代码示例、实际用例和使用 Ultralytics 模型执行各种任务的分步指南。

如果您需要进一步帮助，请查阅 Ultralytics 文档或通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 或官方[讨论论坛](https://github.com/orgs/ultralytics/discussions)联系社区。
