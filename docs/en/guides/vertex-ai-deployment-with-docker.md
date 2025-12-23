---
comments: true
description: 学习如何使用 Docker 容器和 FastAPI 在 Google Cloud Vertex AI 上部署预训练的 YOLO11 模型，实现可扩展的推理，并完全控制预处理和后处理。
keywords: YOLO11, Vertex AI, Docker, FastAPI, 部署, 容器, GCP, Artifact Registry, Ultralytics, 云部署
---

# 使用 Ultralytics 在 Vertex AI 上部署预训练的 YOLO 模型进行推理

本指南将向您展示如何使用 Ultralytics 将预训练的 YOLO11 模型容器化，为其构建 FastAPI 推理服务器，并在 Google Cloud Vertex AI 上部署带有推理服务器的模型。示例实现将涵盖 YOLO11 的对象检测用例，但相同的原则适用于使用[其他 YOLO 模式](../modes/index.md)。

在开始之前，您需要创建一个 Google Cloud Platform (GCP) 项目。作为新用户，您可以获得 300 美元的 GCP 免费额度，这足以测试一个运行中的设置，之后您可以将其扩展到任何其他 YOLO11 用例，包括训练、批量和流式推理。

## 您将学到什么

1. 使用 FastAPI 为 Ultralytics YOLO11 模型创建推理后端。
2. 创建 GCP Artifact Registry 仓库来存储您的 Docker 镜像。
3. 构建并将带有模型的 Docker 镜像推送到 Artifact Registry。
4. 在 Vertex AI 中导入您的模型。
5. 创建 Vertex AI 端点并部署模型。

!!! tip "为什么要部署容器化模型？"

    - **使用 Ultralytics 完全控制模型**：您可以使用自定义推理逻辑，完全控制预处理、后处理和响应格式。
    - **Vertex AI 处理其余部分**：它自动扩展，同时在配置计算资源、内存和 GPU 配置方面提供灵活性。
    - **原生 GCP 集成和安全性**：与 Cloud Storage、BigQuery、Cloud Functions、VPC 控制、IAM 策略和审计日志无缝设置。

## 前提条件

1. 在您的机器上安装 [Docker](https://docs.docker.com/engine/install/)。
2. 安装 [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) 并[认证以使用 gcloud CLI](https://cloud.google.com/docs/authentication/gcloud)。
3. 强烈建议您阅读 [Ultralytics Docker 快速入门指南](https://docs.ultralytics.com/guides/docker-quickstart/)，因为在遵循本指南时您需要扩展官方 Ultralytics Docker 镜像之一。

## 1. 使用 FastAPI 创建推理后端

首先，您需要创建一个 FastAPI 应用程序来处理 YOLO11 模型推理请求。此应用程序将处理模型加载、图像预处理和推理（预测）逻辑。

### Vertex AI 合规基础

Vertex AI 期望您的容器实现两个特定端点：

1. **健康检查**端点 (`/health`)：当服务就绪时必须返回 HTTP 状态 `200 OK`。
2. **预测**端点 (`/predict`)：接受带有 **base64 编码**图像和可选参数的结构化预测请求。根据端点类型，[有效负载大小限制](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type)有所不同。

    `/predict` 端点的请求有效负载应遵循以下 JSON 结构：

    ```json
    {
        "instances": [{ "image": "base64_encoded_image" }],
        "parameters": { "confidence": 0.5 }
    }
    ```

### 项目文件夹结构

我们的大部分构建工作将在 Docker 容器内进行，Ultralytics 也会加载预训练的 YOLO11 模型，因此您可以保持本地文件夹结构简单：

```txt
YOUR_PROJECT/
├── src/
│   ├── __init__.py
│   ├── app.py              # 核心 YOLO11 推理逻辑
│   └── main.py             # FastAPI 推理服务器
├── tests/
├── .env                    # 本地开发的环境变量
├── Dockerfile              # 容器配置
├── LICENSE                 # AGPL-3.0 许可证
└── pyproject.toml          # Python 依赖和项目配置
```

!!! note "重要许可证说明"

    Ultralytics YOLO11 模型和框架采用 AGPL-3.0 许可证，有重要的合规要求。请务必阅读 Ultralytics 文档中关于[如何遵守许可证条款](../help/contributing.md#how-to-comply-with-agpl-30)的内容。

### 创建带有依赖项的 pyproject.toml

为了方便管理您的项目，创建一个包含以下依赖项的 `pyproject.toml` 文件：

```toml
[project]
name = "YOUR_PROJECT_NAME"
version = "0.0.1"
description = "YOUR_PROJECT_DESCRIPTION"
requires-python = ">=3.10,<3.13"
dependencies = [
   "ultralytics>=8.3.0",
   "fastapi[all]>=0.89.1",
   "uvicorn[standard]>=0.20.0",
   "pillow>=9.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

- `uvicorn` 将用于运行 FastAPI 服务器。
- `pillow` 将用于图像处理，但您不仅限于 PIL 图像 — Ultralytics 支持[许多其他格式](../modes/predict.md#inference-sources)。


### 使用 Ultralytics YOLO11 创建推理逻辑

现在您已经设置好项目结构和依赖项，可以实现核心 YOLO11 推理逻辑。创建一个 `src/app.py` 文件，使用 Ultralytics Python API 处理模型加载、图像处理和预测。

```python
# src/app.py

from ultralytics import YOLO

# 模型初始化和就绪状态
model_yolo = None
_model_ready = False


def _initialize_model():
    """初始化 YOLO 模型。"""
    global model_yolo, _model_ready

    try:
        # 使用 Ultralytics 基础镜像中的预训练 YOLO11n 模型
        model_yolo = YOLO("yolo11n.pt")
        _model_ready = True

    except Exception as e:
        print(f"初始化 YOLO 模型时出错: {e}")
        _model_ready = False
        model_yolo = None


# 在模块导入时初始化模型
_initialize_model()


def is_model_ready() -> bool:
    """检查模型是否准备好进行推理。"""
    return _model_ready and model_yolo is not None
```

这将在容器启动时加载一次模型，模型将在所有请求之间共享。如果您的模型将处理繁重的推理负载，建议在稍后在 Vertex AI 中导入模型时选择具有更多内存的机器类型。

接下来，使用 `pillow` 创建两个用于输入和输出图像处理的实用函数。YOLO11 原生支持 PIL 图像。

```python
def get_image_from_bytes(binary_image: bytes) -> Image.Image:
    """将图像从字节转换为 PIL RGB 格式。"""
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image
```

```python
def get_bytes_from_image(image: Image.Image) -> bytes:
    """将 PIL 图像转换为字节。"""
    return_image = io.BytesIO()
    image.save(return_image, format="JPEG", quality=85)
    return_image.seek(0)
    return return_image.getvalue()
```

最后，实现 `run_inference` 函数来处理对象检测。在此示例中，我们将从模型预测中提取边界框、类别名称和置信度分数。该函数将返回一个包含检测结果和原始结果的字典，用于进一步处理或标注。

```python
def run_inference(input_image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """使用 YOLO11n 模型对图像进行推理。"""
    global model_yolo

    # 检查模型是否就绪
    if not is_model_ready():
        print("模型未准备好进行推理")
        return {"detections": [], "results": None}

    try:
        # 进行预测并获取原始结果
        results = model_yolo.predict(
            imgsz=640, source=input_image, conf=confidence_threshold, save=False, augment=False, verbose=False
        )

        # 提取检测结果（边界框、类别名称和置信度）
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                boxes = result.boxes

                # 将张量转换为 numpy 进行处理
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                # 创建检测字典
                for i in range(len(xyxy)):
                    detection = {
                        "xmin": float(xyxy[i][0]),
                        "ymin": float(xyxy[i][1]),
                        "xmax": float(xyxy[i][2]),
                        "ymax": float(xyxy[i][3]),
                        "confidence": float(conf[i]),
                        "class": int(cls[i]),
                        "name": model_yolo.names.get(int(cls[i]), f"class_{int(cls[i])}"),
                    }
                    detections.append(detection)

        return {
            "detections": detections,
            "results": results,  # 保留原始结果用于标注
        }
    except Exception as e:
        # 如果出错，返回空结构
        print(f"YOLO 检测出错: {e}")
        return {"detections": [], "results": None}
```

可选地，您可以添加一个函数，使用 Ultralytics 内置的绘图方法在图像上标注边界框和标签。如果您想在预测响应中返回标注图像，这将很有用。

```python
def get_annotated_image(results: list) -> Image.Image:
    """使用 Ultralytics 内置的 plot 方法获取标注图像。"""
    if not results or len(results) == 0:
        raise ValueError("未提供用于标注的结果")

    result = results[0]
    # 使用 Ultralytics 内置的 plot 方法，输出 PIL 格式
    return result.plot(pil=True)
```

### 使用 FastAPI 创建 HTTP 推理服务器

现在您已经有了核心 YOLO11 推理逻辑，可以创建一个 FastAPI 应用程序来提供服务。这将包括 Vertex AI 所需的健康检查和预测端点。

首先，添加导入并为 Vertex AI 配置日志记录。因为 Vertex AI 将 stderr 视为错误输出，所以将日志输出到 stdout 是有意义的。

```python
import sys

from loguru import logger

# 配置日志记录器
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")
```

为了完全符合 Vertex AI 要求，在环境变量中定义所需的端点并设置请求大小限制。建议在生产部署中使用[私有 Vertex AI 端点](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type)。这样您将获得更高的请求有效负载限制（私有端点为 10 MB，公共端点为 1.5 MB），以及强大的安全性和访问控制。

```python
# Vertex AI 环境变量
AIP_HTTP_PORT = int(os.getenv("AIP_HTTP_PORT", "8080"))
AIP_HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

# 请求大小限制（私有端点 10 MB，公共端点 1.5 MB）
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB（字节）
```

添加两个 Pydantic 模型用于验证请求和响应：

```python
# 请求/响应的 Pydantic 模型
class PredictionRequest(BaseModel):
    instances: list
    parameters: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    predictions: list
```

添加健康检查端点以验证模型就绪状态。**这对 Vertex AI 很重要**，因为没有专用的健康检查，其编排器将 ping 随机套接字，无法确定模型是否准备好进行推理。您的检查必须在成功时返回 `200 OK`，在失败时返回 `503 Service Unavailable`：

```python
# 健康检查端点
@app.get(AIP_HEALTH_ROUTE, status_code=status.HTTP_200_OK)
def health_check():
    """Vertex AI 的健康检查端点。"""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未就绪")
    return {"status": "healthy"}
```

现在您已经拥有实现预测端点所需的一切，该端点将处理推理请求。它将接受图像文件，运行推理，并返回结果。请注意，图像必须是 base64 编码的，这会额外增加有效负载大小最多 33%。

```python
@app.post(AIP_PREDICT_ROUTE, response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Vertex AI 的预测端点。"""
    try:
        predictions = []

        for instance in request.instances:
            if isinstance(instance, dict):
                if "image" in instance:
                    image_data = base64.b64decode(instance["image"])
                    input_image = get_image_from_bytes(image_data)
                else:
                    raise HTTPException(status_code=400, detail="实例必须包含 'image' 字段")
            else:
                raise HTTPException(status_code=400, detail="无效的实例格式")

            # 如果提供了 YOLO11 参数则提取
            parameters = request.parameters or {}
            confidence_threshold = parameters.get("confidence", 0.5)
            return_annotated_image = parameters.get("return_annotated_image", False)

            # 使用 YOLO11n 模型运行推理
            result = run_inference(input_image, confidence_threshold=confidence_threshold)
            detections_list = result["detections"]

            # 为 Vertex AI 格式化预测结果
            detections = []
            for detection in detections_list:
                formatted_detection = {
                    "class": detection["name"],
                    "confidence": detection["confidence"],
                    "bbox": {
                        "xmin": detection["xmin"],
                        "ymin": detection["ymin"],
                        "xmax": detection["xmax"],
                        "ymax": detection["ymax"],
                    },
                }
                detections.append(formatted_detection)

            # 构建预测响应
            prediction = {"detections": detections, "detection_count": len(detections)}

            # 如果请求且存在检测结果，添加标注图像
            if (
                return_annotated_image
                and result["results"]
                and result["results"][0].boxes is not None
                and len(result["results"][0].boxes) > 0
            ):
                import base64

                annotated_image = get_annotated_image(result["results"])
                img_bytes = get_bytes_from_image(annotated_image)
                prediction["annotated_image"] = base64.b64encode(img_bytes).decode("utf-8")

            predictions.append(prediction)

        logger.info(
            f"处理了 {len(request.instances)} 个实例，共发现 {sum(len(p['detections']) for p in predictions)} 个检测结果"
        )

        return PredictionResponse(predictions=predictions)

    except HTTPException:
        # 原样重新抛出 HTTPException（不要捕获并转换为 500）
        raise
    except Exception as e:
        logger.error(f"预测错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")
```

最后，添加应用程序入口点来运行 FastAPI 服务器。

```python
if __name__ == "__main__":
    import uvicorn

    logger.info(f"在端口 {AIP_HTTP_PORT} 上启动服务器")
    logger.info(f"健康检查路由: {AIP_HEALTH_ROUTE}")
    logger.info(f"预测路由: {AIP_PREDICT_ROUTE}")
    uvicorn.run(app, host="0.0.0.0", port=AIP_HTTP_PORT)
```

现在您有了一个完整的 FastAPI 应用程序，可以处理 YOLO11 推理请求。您可以通过安装依赖项并运行服务器在本地测试它，例如使用 uv。

```bash
# 安装依赖项
uv pip install -e .

# 直接运行 FastAPI 服务器
uv run src/main.py
```

要测试服务器，您可以使用 cURL 查询 `/health` 和 `/predict` 端点。将测试图像放在 `tests` 文件夹中。然后，在终端中运行以下命令：

```bash
# 测试健康检查端点
curl http://localhost:8080/health

# 使用 base64 编码图像测试预测端点
curl -X POST -H "Content-Type: application/json" -d "{\"instances\": [{\"image\": \"$(base64 -i tests/test_image.jpg)\"}]}" http://localhost:8080/predict
```

您应该会收到包含检测对象的 JSON 响应。在第一次请求时，预计会有短暂延迟，因为 Ultralytics 需要拉取和加载 YOLO11 模型。

## 2. 使用您的应用程序扩展 Ultralytics Docker 镜像

Ultralytics 提供了几个 Docker 镜像，您可以将其用作应用程序镜像的基础。Docker 将安装 Ultralytics 和必要的 GPU 驱动程序。

要使用 Ultralytics YOLO 模型的全部功能，您应该选择 CUDA 优化镜像进行 GPU 推理。但是，如果 CPU 推理足以满足您的任务，您也可以选择仅 CPU 镜像来节省计算资源：

- [Dockerfile](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile)：用于 YOLO11 单/多 GPU 训练和推理的 CUDA 优化镜像。
- [Dockerfile-cpu](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-cpu)：用于 YOLO11 推理的仅 CPU 镜像。


### 为您的应用程序创建 Docker 镜像

在项目根目录创建一个包含以下内容的 `Dockerfile`：

```dockerfile
# 扩展官方 Ultralytics Docker 镜像用于 YOLO11
FROM ultralytics/ultralytics:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装 FastAPI 和依赖项
RUN uv pip install fastapi[all] uvicorn[standard] loguru

WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./

# 安装应用程序包
RUN uv pip install -e .

RUN mkdir -p /app/logs
ENV PYTHONPATH=/app/src

# Vertex AI 端口
EXPOSE 8080

# 启动推理服务器
ENTRYPOINT ["python", "src/main.py"]
```

在示例中，官方 Ultralytics Docker 镜像 `ultralytics:latest` 用作基础。它已经包含 YOLO11 模型和所有必要的依赖项。服务器的入口点与我们在本地测试 FastAPI 应用程序时使用的相同。

### 构建和测试 Docker 镜像

现在您可以使用以下命令构建 Docker 镜像：

```bash
docker build --platform linux/amd64 -t IMAGE_NAME:IMAGE_VERSION .
```

将 `IMAGE_NAME` 和 `IMAGE_VERSION` 替换为您想要的值，例如 `yolo11-fastapi:0.1`。请注意，如果您要在 Vertex AI 上部署，必须为 `linux/amd64` 架构构建镜像。如果您在 Apple Silicon Mac 或任何其他非 x86 架构上构建镜像，需要显式设置 `--platform` 参数。

镜像构建完成后，您可以在本地测试 Docker 镜像：

```bash
docker run --platform linux/amd64 -p 8080:8080 IMAGE_NAME:IMAGE_VERSION
```

您的 Docker 容器现在正在端口 `8080` 上运行 FastAPI 服务器，准备接受推理请求。您可以使用与之前相同的 cURL 命令测试 `/health` 和 `/predict` 端点：

```bash
# 测试健康检查端点
curl http://localhost:8080/health

# 使用 base64 编码图像测试预测端点
curl -X POST -H "Content-Type: application/json" -d "{\"instances\": [{\"image\": \"$(base64 -i tests/test_image.jpg)\"}]}" http://localhost:8080/predict
```

## 3. 将 Docker 镜像上传到 GCP Artifact Registry

要在 Vertex AI 中导入您的容器化模型，您需要将 Docker 镜像上传到 Google Cloud Artifact Registry。如果您还没有 Artifact Registry 仓库，需要先创建一个。

### 在 Google Cloud Artifact Registry 中创建仓库

在 Google Cloud Console 中打开 [Artifact Registry 页面](https://console.cloud.google.com/artifacts)。如果您是第一次使用 Artifact Registry，可能会提示您先启用 Artifact Registry API。

<p align="center">
  <img width="70%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/create-artifact-registry-repo.png" alt="Google Cloud Artifact Registry 创建仓库界面，显示仓库名称、区域选择和格式选项">
</p>

1. 选择创建仓库。
2. 输入仓库名称。选择所需的区域，其他选项使用默认设置，除非您需要特别更改。

!!! note

    区域选择可能会影响机器的可用性以及非企业用户的某些计算限制。您可以在 Vertex AI 官方文档中找到更多信息：[Vertex AI 配额和限制](https://docs.cloud.google.com/vertex-ai/docs/quotas)

1. 仓库创建后，将您的 PROJECT_ID、位置（区域）和仓库名称保存到您的密钥库或 `.env` 文件中。稍后您需要它们来标记和推送 Docker 镜像到 Artifact Registry。

### 向 Artifact Registry 认证 Docker

向您刚创建的 Artifact Registry 仓库认证您的 Docker 客户端。在终端中运行以下命令：

```sh
gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
```

### 标记并推送镜像到 Artifact Registry

标记并将 Docker 镜像推送到 Google Artifact Registry。

!!! note "为您的镜像使用唯一标签"

    建议每次更新镜像时使用唯一标签。大多数 GCP 服务（包括 Vertex AI）依赖镜像标签进行自动版本控制和扩展，因此使用语义版本控制或基于日期的标签是一个好习惯。

使用 Artifact Registry 仓库 URL 标记您的镜像。将占位符替换为您之前保存的值。

```sh
docker tag IMAGE_NAME:IMAGE_VERSION YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/IMAGE_NAME:IMAGE_VERSION
```

将标记的镜像推送到 Artifact Registry 仓库。

```sh
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/IMAGE_NAME:IMAGE_VERSION
```

等待过程完成。您现在应该可以在 Artifact Registry 仓库中看到该镜像。

有关如何在 Artifact Registry 中处理镜像的更具体说明，请参阅 Artifact Registry 文档：[推送和拉取镜像](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)。

## 4. 在 Vertex AI 中导入您的模型

使用您刚推送的 Docker 镜像，现在可以在 Vertex AI 中导入模型。

1. 在 Google Cloud 导航菜单中，转到 Vertex AI > 模型注册表。或者，在 Google Cloud Console 顶部的搜索栏中搜索"Vertex AI"。
 <p align="center">
   <img width="80%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/vertex-ai-import.png" alt="Vertex AI 模型注册表界面，突出显示用于导入新模型的导入按钮">
 </p>
1. 点击导入。
1. 选择作为新模型导入。
1. 选择区域。您可以选择与 Artifact Registry 仓库相同的区域，但您的选择应该取决于您所在区域的机器类型和配额的可用性。
1. 选择导入现有模型容器。
 <p align="center">
   <img width="80%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/import-model.png" alt="Vertex AI 导入模型对话框，显示容器镜像选择和模型配置选项">
 </p>
1. 在容器镜像字段中，浏览您之前创建的 Artifact Registry 仓库并选择您刚推送的镜像。
1. 向下滚动到环境变量部分，输入您在 FastAPI 应用程序中定义的预测和健康检查端点以及端口。
 <p align="center">
   <img width="60%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/predict-health-port.png" alt="Vertex AI 环境变量配置，显示 FastAPI 端点的预测路由、健康检查路由和端口设置">
 </p>
1. 点击导入。Vertex AI 将需要几分钟来注册模型并准备部署。导入完成后，您将收到电子邮件通知。

## 5. 创建 Vertex AI 端点并部署您的模型

!!! note "Vertex AI 中的端点与模型"

    在 Vertex AI 术语中，**端点**指的是**已部署**的模型，因为它们代表您发送推理请求的 HTTP 端点，而**模型**是存储在模型注册表中的训练好的机器学习工件。

要部署模型，您需要在 Vertex AI 中创建一个端点。

1.  在 Vertex AI 导航菜单中，转到端点。选择您导入模型时使用的区域。点击创建。
<p align="center">
  <img width="60%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/endpoint-name.png" alt="Vertex AI 创建端点界面，显示端点名称输入字段和访问配置选项">
</p>
1.  输入端点名称。
1.  对于访问，Vertex AI 建议使用私有 Vertex AI 端点。除了安全优势外，如果选择私有端点，您还可以获得更高的有效负载限制，但您需要配置 VPC 网络和防火墙规则以允许访问端点。有关[私有端点](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type)的更多说明，请参阅 Vertex AI 文档。
1.  点击继续。
1.  在模型设置对话框中，选择您之前导入的模型。现在您可以为模型配置机器类型、内存和 GPU 设置。如果您预期有高推理负载，请分配足够的内存以确保没有 I/O 瓶颈，从而保证 YOLO11 的正常性能。
1.  在加速器类型中，选择您要用于推理的 GPU 类型。如果您不确定选择哪种 GPU，可以从支持 CUDA 的 NVIDIA T4 开始。

    !!! note "区域和机器类型配额"

        请记住，某些区域的计算配额非常有限，因此您可能无法在您的区域选择某些机器类型或 GPU。如果这很关键，请将部署区域更改为配额更大的区域。在 Vertex AI 官方文档中查找更多信息：[Vertex AI 配额和限制](https://docs.cloud.google.com/vertex-ai/docs/quotas)。

1.  选择机器类型后，您可以点击继续。此时，您可以选择在 Vertex AI 中启用模型监控——这是一项额外服务，将跟踪模型的性能并提供其行为的洞察。这是可选的，会产生额外费用，请根据您的需求选择。点击创建。

Vertex AI 将需要几分钟（在某些区域最多 30 分钟）来部署模型。部署完成后，您将收到电子邮件通知。

## 6. 测试您部署的模型

部署完成后，Vertex AI 将为您提供一个示例 API 界面来测试您的模型。

要测试远程推理，您可以使用提供的 cURL 命令或创建另一个 Python 客户端库来向部署的模型发送请求。请记住，在将图像发送到 `/predict` 端点之前，您需要将图像编码为 base64。

<p align="center">
  <img width="50%" src="https://github.com/ultralytics/docs/releases/download/0/vertex-ai-endpoint-test-curl-yolo11.avif" alt="Vertex AI 端点测试界面，显示用于向已部署的 YOLO11 模型发出预测请求的示例 cURL 命令">
</p>

!!! note "预计第一次请求会有短暂延迟"

    与本地测试类似，预计第一次请求会有短暂延迟，因为 Ultralytics 需要在运行的容器中拉取和加载 YOLO11 模型。

您已成功使用 Ultralytics 在 Google Cloud Vertex AI 上部署了预训练的 YOLO11 模型。

## 常见问题

### 我可以在没有 Docker 的情况下在 Vertex AI 上使用 Ultralytics YOLO 模型吗？

可以；但是，您首先需要将模型导出为与 Vertex AI 兼容的格式，如 TensorFlow、Scikit-learn 或 XGBoost。Google Cloud 提供了在 Vertex 上运行 `.pt` 模型的指南，其中包含转换过程的完整概述：[在 Vertex AI 上运行 PyTorch 模型](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)。

请注意，生成的设置将仅依赖于 Vertex AI 标准服务层，不支持高级 Ultralytics 框架功能。由于 Vertex AI 完全支持容器化模型，并可以根据您的部署配置自动扩展它们，因此它允许您利用 Ultralytics YOLO 模型的全部功能，而无需将它们转换为不同的格式。

### 为什么 FastAPI 是服务 YOLO11 推理的好选择？

FastAPI 为推理工作负载提供高吞吐量。异步支持允许处理多个并发请求而不阻塞主线程，这在服务计算机视觉模型时很重要。

FastAPI 的自动请求/响应验证减少了生产推理服务中的运行时错误。这对于输入格式一致性至关重要的对象检测 API 特别有价值。

FastAPI 为您的推理管道增加的计算开销最小，为模型执行和图像处理任务留下更多资源。

FastAPI 还支持 SSE（服务器发送事件），这对于流式推理场景很有用。

### 为什么我必须多次选择区域？

这实际上是 Google Cloud Platform 的一个灵活性功能，您需要为使用的每项服务选择一个区域。对于在 Vertex AI 上部署容器化模型的任务，最重要的区域选择是模型注册表的区域。它将决定您模型部署的机器类型和配额的可用性。

此外，如果您要扩展设置并将预测数据或结果存储在 Cloud Storage 或 BigQuery 中，您需要使用与模型注册表相同的区域，以最小化延迟并确保数据访问的高吞吐量。
