---
comments: true
description: 了解如何使用 Ultralytics YOLO11 开始使用 Seeed Studio reCamera 进行边缘 AI 应用。学习其强大功能、实际应用以及如何将 YOLO11 模型导出为 ONNX 格式以实现无缝集成。
keywords: Seeed Studio reCamera, YOLO11, ONNX 导出, 边缘 AI, 计算机视觉, 实时检测, 个人防护设备检测, 火灾检测, 废物检测, 跌倒检测, 模块化 AI 设备, Ultralytics
---

# 快速入门指南：Seeed Studio reCamera 与 Ultralytics YOLO11

[reCamera](https://www.seeedstudio.com/recamera) 在 [YOLO Vision 2024 (YV24)](https://www.youtube.com/watch?v=rfI5vOo3-_A)（[Ultralytics](https://www.ultralytics.com/) 年度混合活动）上向 AI 社区推出。它主要为[边缘 AI 应用](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai)设计，提供强大的处理能力和轻松的部署。

凭借对多种硬件配置和开源资源的支持，它是在边缘原型设计和部署创新[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)[解决方案](https://docs.ultralytics.com/solutions/#solutions)的理想平台。

![Seeed Studio reCamera](https://github.com/ultralytics/docs/releases/download/0/saeed-studio-recamera.avif)

## 为什么选择 reCamera？

reCamera 系列专为边缘 AI 应用而构建，旨在满足开发者和创新者的需求。以下是它脱颖而出的原因：

- **RISC-V 驱动的性能**：其核心是基于 RISC-V 架构的 SG200X 处理器，为边缘 AI 任务提供卓越性能，同时保持能效。凭借每秒执行 1 万亿次运算（1 TOPS）的能力，它可以轻松处理实时[目标检测](https://docs.ultralytics.com/tasks/detect/)等要求苛刻的任务。

- **优化的视频技术**：支持先进的视频压缩标准，包括 H.264 和 H.265，在不牺牲质量的情况下减少存储和带宽需求。HDR 成像、3D 降噪和镜头校正等功能即使在具有挑战性的环境中也能确保专业的视觉效果。

- **节能双处理**：SG200X 处理复杂的 AI 任务，而较小的 8 位微控制器管理更简单的操作以节省电力，使 reCamera 成为电池供电或低功耗设置的理想选择。

- **模块化和可升级设计**：reCamera 采用模块化结构，由三个主要组件组成：核心板、传感器板和底板。这种设计允许开发者轻松更换或升级组件，确保灵活性和面向未来的项目。

## reCamera 快速硬件设置

请按照 [reCamera 快速入门指南](https://wiki.seeedstudio.com/recamera_getting_started/)进行设备的初始配置，例如将设备连接到 WiFi 网络并访问 [Node-RED](https://nodered.org/) Web UI 以快速预览检测结果。

## 使用预装的 YOLO11 模型进行推理

reCamera 预装了四个 Ultralytics YOLO11 模型，您可以在 Node-RED 仪表板中简单地选择所需的模型。

- [检测 (YOLO11n)](../tasks/detect.md)
- [分类 (YOLO11n-cls)](../tasks/classify.md)
- [分割 (YOLO11n-seg)](../tasks/segment.md)
- [姿态估计 (YOLO11n-pose)](../tasks/pose.md)

步骤 1：如果您已将 reCamera 连接到网络，请在 Web 浏览器中输入 reCamera 的 IP 地址以打开 Node-RED 仪表板。如果您通过 USB 将 reCamera 连接到 PC，可以输入 `192.168.42.1`。在这里您将看到默认加载的 YOLO11n 检测模型。

![reCamera YOLO11n 演示](https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-yolo11n-demo.avif)

步骤 2：点击右下角的绿色圆圈以访问 Node-RED 流程编辑器。

步骤 3：点击 `model` 节点并点击 `On Device`。

![Node-RED 模型选择](https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-nodered-model-select.avif)

步骤 4：从四个不同的预装 YOLO11n 模型中选择一个，然后点击 `Done`。例如，这里我们将选择 `YOLO11n Pose`

<p align="center">
  <img width="50%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-nodered-yolo11n-pose.avif" alt="Node-RED YOLO11n-pose 选择">
</p>

步骤 5：点击 `Deploy`，部署完成后，点击 `Dashboard`。

![reCamera Node-RED 部署](https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-nodered-deploy.avif)

现在您将能够看到 YOLO11n 姿态估计模型在运行！

![reCamera YOLO11n-pose 演示](https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-yolo11n-pose-demo.avif)

## 导出到 cvimodel：转换您的 YOLO11 模型

如果您想在 reCamera 上使用[自定义训练的 YOLO11 模型](../modes/train.md)，请按照以下步骤操作。

在这里，我们首先将 `PyTorch` 模型转换为 `ONNX`，然后将其转换为 `MLIR` 模型格式。最后，`MLIR` 将被转换为 `cvimodel` 以在设备上运行推理。

<p align="center">
  <img width="80%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-toolchain-workflow.avif" alt="reCamera 工具链">
</p>

### 导出到 ONNX

将 Ultralytics YOLO11 模型导出为 [ONNX 模型格式](https://docs.ultralytics.com/integrations/onnx/)。

#### 安装

要安装所需的包，请运行：

!!! Tip "安装"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

#### 使用方法

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx", opset=14)  # 创建 'yolo11n.onnx'
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 ONNX 格式
        yolo export model=yolo11n.pt format=onnx opset=14 # 创建 'yolo11n.onnx'
        ```

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

### 将 ONNX 导出为 MLIR 和 cvimodel

获得 ONNX 模型后，请参阅[转换和量化 AI 模型](https://wiki.seeedstudio.com/recamera_model_conversion/)页面将 ONNX 模型转换为 MLIR，然后转换为 cvimodel。

!!! note

    我们正在积极努力将 reCamera 支持直接添加到 Ultralytics 包中，它将很快可用。同时，请查看我们关于[将 Ultralytics YOLO 模型与 Seeed Studio reCamera 集成](https://www.ultralytics.com/blog/integrating-ultralytics-yolo-models-on-seeed-studios-recamera)的博客以获取更多见解。

## 基准测试

即将推出。

## reCamera 的实际应用

reCamera 先进的计算机视觉能力和模块化设计使其适用于各种实际场景，帮助开发者和企业轻松应对独特挑战。

- **跌倒检测**：专为安全和医疗保健应用设计，reCamera 可以实时检测跌倒，非常适合老年护理、医院和工业环境，这些场景需要快速响应。

- **个人防护设备检测**：reCamera 可用于通过实时检测 PPE 合规性来确保工作场所安全。它有助于识别工人是否佩戴头盔、手套或其他安全装备，降低工业环境中的风险。

![个人防护设备检测](https://github.com/ultralytics/docs/releases/download/0/personal-protective-equipment-detection.avif)

- **火灾检测**：reCamera 的实时处理能力使其成为工业和住宅区域[火灾检测](https://www.ultralytics.com/blog/computer-vision-in-fire-detection-and-prevention)的绝佳选择，提供早期预警以防止潜在灾难。

- **废物检测**：它还可用于废物检测应用，使其成为环境监测和[废物管理](https://www.ultralytics.com/blog/simplifying-e-waste-management-with-ai-innovations)的出色工具。

- **汽车零件检测**：在制造和汽车行业，它有助于检测和分析汽车零件，用于质量控制、装配线监控和库存管理。

![汽车零件检测](https://github.com/ultralytics/docs/releases/download/0/carparts-detection.avif)

## 常见问题

### 如何首次安装和设置 reCamera？

要首次设置 reCamera，请按照以下步骤操作：

1. 将 reCamera 连接到电源
2. 使用 [reCamera 快速入门指南](https://wiki.seeedstudio.com/recamera_getting_started/)将其连接到 WiFi 网络
3. 通过在 Web 浏览器中输入设备的 IP 地址访问 Node-RED Web UI（如果通过 USB 连接，则使用 `192.168.42.1`）
4. 通过仪表板界面立即开始使用预装的 YOLO11 模型

### 我可以在 reCamera 上使用自定义训练的 YOLO11 模型吗？

是的，您可以在 reCamera 上使用自定义训练的 YOLO11 模型。该过程包括：

1. 使用 `model.export(format="onnx", opset=14)` 将 PyTorch 模型导出为 ONNX 格式
2. 将 ONNX 模型转换为 MLIR 格式
3. 将 MLIR 转换为 cvimodel 格式以进行设备上推理
4. 将转换后的模型加载到 reCamera 上

有关详细说明，请参阅[转换和量化 AI 模型](https://wiki.seeedstudio.com/recamera_model_conversion/)指南。

### reCamera 与传统 IP 摄像头有什么不同？

与需要外部硬件进行处理的传统 IP 摄像头不同，reCamera：

- 通过其 RISC-V SG200X 处理器直接在设备上集成 AI 处理
- 为实时边缘 AI 应用提供 1 TOPS 的计算能力
- 采用模块化设计，允许组件升级和定制
- 支持 H.264/H.265 压缩、HDR 成像和 3D 降噪等先进视频技术
- 预装 Ultralytics YOLO11 模型可立即使用

这些功能使 reCamera 成为边缘 AI 应用的独立解决方案，无需额外的外部处理硬件。
