---
comments: true
description: 学习如何将 Ultralytics YOLO11 与 NVIDIA Triton 推理服务器集成，以实现可扩展、高性能的 AI 模型部署。
keywords: Triton 推理服务器, YOLO11, Ultralytics, NVIDIA, 深度学习, AI 模型部署, ONNX, 可扩展推理
---

# 使用 Ultralytics YOLO11 的 Triton 推理服务器

[Triton 推理服务器](https://developer.nvidia.com/dynamo)（以前称为 TensorRT 推理服务器）是由 NVIDIA 开发的开源软件解决方案。它提供了针对 NVIDIA GPU 优化的云推理解决方案。Triton 简化了 AI 模型在生产环境中的大规模部署。将 Ultralytics YOLO11 与 Triton 推理服务器集成，可以部署可扩展、高性能的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)推理工作负载。本指南提供了设置和测试集成的步骤。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NQDtfSi5QF4"
    title="NVIDIA Triton 推理服务器入门" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>NVIDIA Triton 推理服务器入门。
</p>

## 什么是 Triton 推理服务器？

Triton 推理服务器旨在在生产环境中部署各种 AI 模型。它支持广泛的深度学习和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)框架，包括 TensorFlow、[PyTorch](https://www.ultralytics.com/glossary/pytorch)、ONNX Runtime 等。其主要用例包括：

- 从单个服务器实例提供多个模型服务
- 无需重启服务器即可动态加载和卸载模型
- 集成推理，允许多个模型一起使用以获得结果
- 用于 A/B 测试和滚动更新的模型版本控制

## Triton 推理服务器的主要优势

将 Triton 推理服务器与 Ultralytics YOLO11 一起使用有几个优势：

- **自动批处理**：在处理之前将多个 AI 请求组合在一起，减少延迟并提高推理速度
- **Kubernetes 集成**：云原生设计与 Kubernetes 无缝配合，用于管理和扩展 AI 应用程序
- **硬件特定优化**：充分利用 NVIDIA GPU 以获得最大性能
- **框架灵活性**：支持多种 AI 框架，包括 TensorFlow、PyTorch、ONNX 和 TensorRT
- **开源且可定制**：可以修改以适应特定需求，确保各种 AI 应用程序的灵活性

## 先决条件

在继续之前，请确保您具备以下先决条件：

- 在您的机器上安装 Docker
- 安装 `tritonclient`：
    ```bash
    pip install tritonclient[all]
    ```

## 将 YOLO11 导出为 ONNX 格式

在 Triton 上部署模型之前，必须将其导出为 ONNX 格式。ONNX（开放神经网络交换）是一种允许在不同深度学习框架之间传输模型的格式。使用 `YOLO` 类的 `export` 函数：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")  # 加载官方模型

# 在导出期间检索元数据。元数据需要添加到 config.pbtxt。请参阅下一节。
metadata = []


def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# 导出模型
onnx_file = model.export(format="onnx", dynamic=True)
```

## 设置 Triton 模型仓库

Triton 模型仓库是 Triton 可以访问和加载模型的存储位置。

1. 创建必要的目录结构：

    ```python
    from pathlib import Path

    # 定义路径
    model_name = "yolo"
    triton_repo_path = Path("tmp") / "triton_repo"
    triton_model_path = triton_repo_path / model_name

    # 创建目录
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    ```

2. 将导出的 ONNX 模型移动到 Triton 仓库：

    ```python
    from pathlib import Path

    # 将 ONNX 模型移动到 Triton 模型路径
    Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

    # 创建配置文件
    (triton_model_path / "config.pbtxt").touch()

    data = """
    # 添加元数据
    parameters {
      key: "metadata"
      value {
        string_value: "%s"
      }
    }

    # （可选）为 GPU 推理启用 TensorRT
    # 由于 TensorRT 引擎转换，首次运行会较慢
    optimization {
      execution_accelerators {
        gpu_execution_accelerator {
          name: "tensorrt"
          parameters {
            key: "precision_mode"
            value: "FP16"
          }
          parameters {
            key: "max_workspace_size_bytes"
            value: "3221225472"
          }
          parameters {
            key: "trt_engine_cache_enable"
            value: "1"
          }
          parameters {
            key: "trt_engine_cache_path"
            value: "/models/yolo/1"
          }
        }
      }
    }
    """ % metadata[0]  # noqa

    with open(triton_model_path / "config.pbtxt", "w") as f:
        f.write(data)
    ```

## 运行 Triton 推理服务器

使用 Docker 运行 Triton 推理服务器：

```python
import contextlib
import subprocess
import time

from tritonclient.http import InferenceServerClient

# 定义镜像 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:24.09-py3"  # 8.57 GB

# 拉取镜像
subprocess.call(f"docker pull {tag}", shell=True)

# 运行 Triton 服务器并捕获容器 ID
container_id = (
    subprocess.check_output(
        f"docker run -d --rm --runtime=nvidia --gpus 0 -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

# 等待 Triton 服务器启动
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# 等待模型准备就绪
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
```

然后使用 Triton 服务器模型运行推理：

```python
from ultralytics import YOLO

# 加载 Triton 服务器模型
model = YOLO("http://localhost:8000/yolo", task="detect")

# 在服务器上运行推理
results = model("path/to/image.jpg")
```

清理容器：

```python
# 在测试结束时终止并删除容器
subprocess.call(f"docker kill {container_id}", shell=True)
```

## TensorRT 优化（可选）

为了获得更高的性能，您可以将 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) 与 Triton 推理服务器一起使用。TensorRT 是专为 NVIDIA GPU 构建的高性能深度学习优化器，可以显著提高推理速度。

将 TensorRT 与 Triton 一起使用的主要优势包括：

- 与未优化模型相比，推理速度提高多达 36 倍
- 针对最大 GPU 利用率的硬件特定优化
- 支持降低精度格式（INT8、FP16）同时保持准确性
- 层融合以减少计算开销

要直接使用 TensorRT，您可以将 YOLO11 模型导出为 TensorRT 格式：

```python
from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 将模型导出为 TensorRT 格式
model.export(format="engine")  # 创建 'yolo11n.engine'
```

有关 TensorRT 优化的更多信息，请参阅 [TensorRT 集成指南](https://docs.ultralytics.com/integrations/tensorrt/)。

---

通过遵循上述步骤，您可以在 Triton 推理服务器上高效部署和运行 Ultralytics YOLO11 模型，为深度学习推理任务提供可扩展和高性能的解决方案。如果您遇到任何问题或有进一步的疑问，请参阅[官方 Triton 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)或联系 Ultralytics 社区获取支持。

## 常见问题

### 如何使用 NVIDIA Triton 推理服务器设置 Ultralytics YOLO11？

使用 [NVIDIA Triton 推理服务器](https://developer.nvidia.com/dynamo)设置 [Ultralytics YOLO11](../models/yolo11.md) 涉及几个关键步骤：

1. **将 YOLO11 导出为 ONNX 格式**：

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo11n.pt")  # 加载官方模型

    # 将模型导出为 ONNX 格式
    onnx_file = model.export(format="onnx", dynamic=True)
    ```

2. **设置 Triton 模型仓库**：

    ```python
    from pathlib import Path

    # 定义路径
    model_name = "yolo"
    triton_repo_path = Path("tmp") / "triton_repo"
    triton_model_path = triton_repo_path / model_name

    # 创建目录
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    ```

3. **运行 Triton 服务器**：

    ```python
    import contextlib
    import subprocess
    import time

    from tritonclient.http import InferenceServerClient

    # 定义镜像
    tag = "nvcr.io/nvidia/tritonserver:24.09-py3"

    subprocess.call(f"docker pull {tag}", shell=True)

    container_id = (
        subprocess.check_output(
            f"docker run -d --rm --runtime=nvidia --gpus 0 -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)
    ```

此设置可以帮助您在 Triton 推理服务器上高效地大规模部署 YOLO11 模型，以实现高性能 AI 模型推理。

### 使用 Ultralytics YOLO11 与 NVIDIA Triton 推理服务器有什么好处？

将 Ultralytics YOLO11 与 [NVIDIA Triton 推理服务器](https://developer.nvidia.com/dynamo)集成提供了几个优势：

- **可扩展的 AI 推理**：Triton 允许从单个服务器实例提供多个模型服务，支持动态模型加载和卸载，使其对各种 AI 工作负载具有高度可扩展性。
- **高性能**：针对 NVIDIA GPU 优化，Triton 推理服务器确保高速推理操作，非常适合[目标检测](https://www.ultralytics.com/glossary/object-detection)等实时应用。
- **集成和模型版本控制**：Triton 的集成模式允许组合多个模型以改进结果，其模型版本控制支持 A/B 测试和滚动更新。
- **自动批处理**：Triton 自动将多个推理请求组合在一起，显著提高吞吐量并减少延迟。
- **简化部署**：无需完全系统改造即可逐步优化 AI 工作流程，使高效扩展更容易。

有关设置和运行 YOLO11 与 Triton 的详细说明，您可以参阅[设置指南](#设置-triton-模型仓库)。

### 为什么在使用 Triton 推理服务器之前应该将 YOLO11 模型导出为 ONNX 格式？

在 [NVIDIA Triton 推理服务器](https://developer.nvidia.com/dynamo)上部署 Ultralytics YOLO11 模型之前使用 ONNX（开放神经网络交换）格式有几个关键优势：

- **互操作性**：ONNX 格式支持在不同深度学习框架（如 PyTorch、TensorFlow）之间传输，确保更广泛的兼容性。
- **优化**：许多部署环境（包括 Triton）针对 ONNX 进行了优化，实现更快的推理和更好的性能。
- **易于部署**：ONNX 在各种框架和平台上得到广泛支持，简化了在各种操作系统和硬件配置中的部署过程。
- **框架独立性**：一旦转换为 ONNX，您的模型就不再绑定到其原始框架，使其更具可移植性。
- **标准化**：ONNX 提供了标准化表示，有助于克服不同 AI 框架之间的兼容性问题。

要导出模型，请使用：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
onnx_file = model.export(format="onnx", dynamic=True)
```

您可以按照 [ONNX 集成指南](https://docs.ultralytics.com/integrations/onnx/)中的步骤完成该过程。

### 我可以在 Triton 推理服务器上使用 Ultralytics YOLO11 模型运行推理吗？

是的，您可以在 [NVIDIA Triton 推理服务器](https://developer.nvidia.com/dynamo)上使用 Ultralytics YOLO11 模型运行推理。一旦您的模型在 Triton 模型仓库中设置好并且服务器正在运行，您可以按如下方式加载和运行模型推理：

```python
from ultralytics import YOLO

# 加载 Triton 服务器模型
model = YOLO("http://localhost:8000/yolo", task="detect")

# 在服务器上运行推理
results = model("path/to/image.jpg")
```

这种方法允许您利用 Triton 的优化，同时使用熟悉的 Ultralytics YOLO 界面。有关设置和运行 Triton 服务器与 YOLO11 的深入指南，请参阅[运行 Triton 推理服务器](#运行-triton-推理服务器)部分。

### Ultralytics YOLO11 与 TensorFlow 和 PyTorch 模型在部署方面有何比较？

[Ultralytics YOLO11](../models/yolo11.md) 相比 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 PyTorch 模型在部署方面提供了几个独特优势：

- **实时性能**：针对实时目标检测任务进行了优化，YOLO11 提供最先进的[准确性](https://www.ultralytics.com/glossary/accuracy)和速度，非常适合需要实时视频分析的应用。
- **易于使用**：YOLO11 与 Triton 推理服务器无缝集成，并支持多种导出格式（ONNX、TensorRT、CoreML），使其适用于各种部署场景。
- **高级功能**：YOLO11 包括动态模型加载、模型版本控制和集成推理等功能，这些对于可扩展和可靠的 AI 部署至关重要。
- **简化的 API**：Ultralytics API 在不同部署目标之间提供一致的界面，减少学习曲线和开发时间。
- **边缘优化**：YOLO11 模型在设计时考虑了边缘部署，即使在资源受限的设备上也能提供出色的性能。

有关更多详细信息，请比较[模型导出指南](../modes/export.md)中的部署选项。
