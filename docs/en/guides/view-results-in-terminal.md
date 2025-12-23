---
comments: true
description: 学习如何在 Linux 和 MacOS 上使用 sixel 直接在 VSCode 终端中可视化 YOLO 推理结果。
keywords: YOLO, 推理结果, VSCode 终端, sixel, 显示图像, Linux, MacOS
---

# 在终端中查看推理结果

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/sixel-example-terminal.avif" alt="终端中的 Sixel 图像示例">
</p>

图片来自 [libsixel](https://saitoha.github.io/libsixel/) 网站。

## 动机

当连接到远程机器时，通常无法可视化图像结果，或者需要将数据移动到具有 GUI 的本地设备。VSCode 集成终端允许直接渲染图像。这是一个简短的演示，展示如何将此功能与 `ultralytics` 的[预测结果](../modes/predict.md)结合使用。

!!! warning

    仅与 Linux 和 MacOS 兼容。查看 [VSCode 仓库](https://github.com/microsoft/vscode)、[Issue 状态](https://github.com/microsoft/vscode/issues/198622)或[文档](https://code.visualstudio.com/docs)以获取有关 Windows 支持使用 `sixel` 在终端中查看图像的更新。

VSCode 使用集成终端查看图像的兼容协议是 [`sixel`](https://en.wikipedia.org/wiki/Sixel) 和 [`iTerm`](https://iterm2.com/documentation-images.html)。本指南将演示 `sixel` 协议的使用。

## 过程

1. 首先，您必须在 VSCode 中启用设置 `terminal.integrated.enableImages` 和 `terminal.integrated.gpuAcceleration`。

    ```yaml
    "terminal.integrated.gpuAcceleration": "auto" # "auto" 是默认值，也可以使用 "on"
    "terminal.integrated.enableImages": true
    ```

    <p align="center">
      <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/vscode-enable-terminal-images-setting.avif" alt="VSCode 启用终端图像设置">
    </p>

2. 在虚拟环境中安装 `python-sixel` 库。这是 `PySixel` 库的一个[分支](https://github.com/lubosz/python-sixel?tab=readme-ov-file)，该库已不再维护。

    ```bash
    pip install sixel
    ```

3. 加载模型并执行推理，然后绘制结果并存储在变量中。有关推理参数和处理结果的更多信息，请参阅[预测模式](../modes/predict.md)页面。

    ```{ .py .annotate }
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo11n.pt")

    # 对图像运行推理
    results = model.predict(source="ultralytics/assets/bus.jpg")

    # 绘制推理结果
    plot = results[0].plot()  # (1)!
    ```

    1. 请参阅[绘图方法参数](../modes/predict.md#plot-method-parameters)以查看可能使用的参数。

4. 现在，使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) 将 `np.ndarray` 转换为 `bytes` 数据。然后使用 `io.BytesIO` 创建一个"类文件"对象。

    ```{ .py .annotate }
    import io

    import cv2

    # 将结果图像转换为字节
    im_bytes = cv2.imencode(
        ".png",  # (1)!
        plot,
    )[1].tobytes()  # (2)!

    # 将图像字节转换为类文件对象
    mem_file = io.BytesIO(im_bytes)
    ```

    1. 也可以使用其他图像扩展名。
    2. 只需要返回的索引 `1` 处的对象。

5. 创建一个 `SixelWriter` 实例，然后使用 `.draw()` 方法在终端中绘制图像。

    ```python
    from sixel import SixelWriter

    # 创建 sixel 写入器对象
    w = SixelWriter()

    # 在终端中绘制 sixel 图像
    w.draw(mem_file)
    ```

## 推理结果示例

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/view-image-in-terminal.avif" alt="在终端中查看图像">
</p>

!!! danger

    使用此示例处理视频或动画 GIF 帧**尚未**经过测试。请自行承担风险尝试。

## 完整代码示例

```{ .py .annotate }
import io

import cv2
from sixel import SixelWriter

from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 对图像运行推理
results = model.predict(source="ultralytics/assets/bus.jpg")

# 绘制推理结果
plot = results[0].plot()  # (3)!

# 将结果图像转换为字节
im_bytes = cv2.imencode(
    ".png",  # (1)!
    plot,
)[1].tobytes()  # (2)!

mem_file = io.BytesIO(im_bytes)
w = SixelWriter()
w.draw(mem_file)
```

1. 也可以使用其他图像扩展名。
2. 只需要返回的索引 `1` 处的对象。
3. 请参阅[绘图方法参数](../modes/predict.md#plot-method-parameters)以查看可能使用的参数。

---

!!! tip

    您可能需要使用 `clear` 来"擦除"终端中图像的视图。

## 常见问题

### 如何在 macOS 或 Linux 上的 VSCode 终端中查看 YOLO 推理结果？

要在 macOS 或 Linux 上的 VSCode 终端中查看 YOLO 推理结果，请按照以下步骤操作：

1. 启用必要的 VSCode 设置：

    ```yaml
    "terminal.integrated.enableImages": true
    "terminal.integrated.gpuAcceleration": "auto"
    ```

2. 安装 sixel 库：

    ```bash
    pip install sixel
    ```

3. 加载 YOLO 模型并运行推理：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.predict(source="path_to_image")
    plot = results[0].plot()
    ```

4. 将推理结果图像转换为字节并在终端中显示：

    ```python
    import io

    import cv2
    from sixel import SixelWriter

    im_bytes = cv2.imencode(".png", plot)[1].tobytes()
    mem_file = io.BytesIO(im_bytes)
    SixelWriter().draw(mem_file)
    ```

有关更多详细信息，请访问[预测模式](../modes/predict.md)页面。

### 为什么 sixel 协议只能在 Linux 和 macOS 上工作？

sixel 协议目前仅在 Linux 和 macOS 上受支持，因为这些平台具有与 sixel 图形兼容的原生终端功能。Windows 对使用 sixel 的终端图形支持仍在开发中。有关 Windows 兼容性的更新，请查看 [VSCode Issue 状态](https://github.com/microsoft/vscode/issues/198622)和[文档](https://code.visualstudio.com/docs)。

### 如果在 VSCode 终端中显示图像时遇到问题怎么办？

如果在使用 sixel 在 VSCode 终端中显示图像时遇到问题：

1. 确保 VSCode 中启用了必要的设置：

    ```yaml
    "terminal.integrated.enableImages": true
    "terminal.integrated.gpuAcceleration": "auto"
    ```

2. 验证 sixel 库安装：

    ```bash
    pip install sixel
    ```

3. 检查图像数据转换和绘图代码是否有错误。例如：

    ```python
    import io

    import cv2
    from sixel import SixelWriter

    im_bytes = cv2.imencode(".png", plot)[1].tobytes()
    mem_file = io.BytesIO(im_bytes)
    SixelWriter().draw(mem_file)
    ```

如果问题仍然存在，请查阅 [VSCode 仓库](https://github.com/microsoft/vscode)，并访问[绘图方法参数](../modes/predict.md#plot-method-parameters)部分获取更多指导。

### YOLO 可以使用 sixel 在终端中显示视频推理结果吗？

目前尚未测试使用 sixel 在终端中显示视频推理结果或动画 GIF 帧，可能不受支持。我们建议从静态图像开始并验证兼容性。请自行承担风险尝试视频结果，并注意性能限制。有关绘制推理结果的更多信息，请访问[预测模式](../modes/predict.md)页面。

### 如何排除 `python-sixel` 库的问题？

要排除 `python-sixel` 库的问题：

1. 确保库在虚拟环境中正确安装：

    ```bash
    pip install sixel
    ```

2. 验证您是否具有必要的 Python 和系统依赖项。

3. 参阅 [python-sixel GitHub 仓库](https://github.com/lubosz/python-sixel)获取更多文档和社区支持。

4. 仔细检查代码中的潜在错误，特别是 `SixelWriter` 的使用和图像数据转换步骤。

有关使用 YOLO 模型和 sixel 集成的更多帮助，请参阅[导出](../modes/export.md)和[预测模式](../modes/predict.md)文档页面。
