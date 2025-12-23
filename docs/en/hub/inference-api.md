---
comments: true
description: 了解如何使用 Ultralytics HUB 推理 API 运行推理。包含 Python 和 cURL 示例，便于快速集成。
keywords: Ultralytics, HUB, 推理 API, Python, cURL, REST API, YOLO, 图像处理, 机器学习, AI 集成
---

# Ultralytics HUB 推理 API

[训练模型](./models.md#训练模型)后，您可以免费使用[共享推理 API](#共享推理-api)。如果您是 [Pro](./pro.md) 用户，可以访问[专用推理 API](#专用推理-api)。[Ultralytics HUB](https://www.ultralytics.com/hub) 推理 API 允许您通过 REST API 运行推理，无需在本地安装和设置 Ultralytics YOLO 环境。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向专用推理 API 卡片和共享推理 API 卡片](https://github.com/ultralytics/docs/releases/download/0/hub-inference-api-card.avif)

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/OpWpBI35A5Y"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics HUB 推理 API 演练
</p>

## 专用推理 API

为响应高需求和广泛兴趣，我们很高兴推出 [Ultralytics HUB](https://www.ultralytics.com/hub) 专用推理 API，为我们的 [Pro](./pro.md) 用户提供在专用环境中一键部署的功能！

!!! note "注意"

    我们很高兴在公测期间作为 [Pro 计划](./pro.md)的一部分免费提供此功能，未来可能会有付费层级。

- **全球覆盖**：部署在全球 38 个区域，确保从任何位置都能低延迟访问。[查看 Google Cloud 区域完整列表](https://cloud.google.com/about/locations)。
- **Google Cloud Run 支持**：由 Google Cloud Run 提供支持，提供无限可扩展且高度可靠的基础设施。
- **高速**：根据 Ultralytics 测试，从附近区域进行 640 分辨率的 YOLOv8n 推理可实现低于 100ms 的延迟。
- **增强安全性**：提供强大的安全功能来保护您的数据并确保符合行业标准。[了解更多关于 Google Cloud 安全性](https://cloud.google.com/security)。

要使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 专用推理 API，请点击**启动端点**按钮。然后，按照以下指南中的说明使用唯一的端点 URL。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向专用推理 API 卡片中的启动端点按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dedicated-inference-api.avif)

!!! tip "提示"

    如[文档](https://docs.ultralytics.com/reference/hub/google/__init__/)中所述，选择延迟最低的区域以获得最佳性能。

要关闭专用端点，请点击**停止端点**按钮。

![Ultralytics HUB 模型页面部署选项卡截图，箭头指向专用推理 API 卡片中的停止端点按钮](https://github.com/ultralytics/docs/releases/download/0/deploy-tab-model-page-stop-endpoint.avif)

## 共享推理 API

要使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 共享推理 API，请按照以下指南操作。

[Ultralytics HUB](https://www.ultralytics.com/hub) 共享推理 API 有以下使用限制：

- 100 次调用 / 小时

## Python

要使用 Python 访问 [Ultralytics HUB](https://www.ultralytics.com/hub) 推理 API，请使用以下代码：

```python
import requests

# API URL
url = "https://predict.ultralytics.com"

# 请求头，使用实际的 API_KEY
headers = {"x-api-key": "API_KEY"}

# 推理参数（使用实际的 MODEL_ID）
data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

# 加载图像并发送请求
with open("path/to/image.jpg", "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())
```

!!! note "注意"

    将 `MODEL_ID` 替换为所需的模型 ID，将 `API_KEY` 替换为您的实际 API 密钥，将 `path/to/image.jpg` 替换为要运行推理的图像路径。

    如果您使用的是[专用推理 API](#专用推理-api)，也请替换 `url`。

## cURL

要使用 cURL 访问 [Ultralytics HUB](https://www.ultralytics.com/hub) 推理 API，请使用以下代码：

```bash
curl -X POST "https://predict.ultralytics.com" \
  -H "x-api-key: API_KEY" \
  -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
  -F "file=@/path/to/image.jpg" \
  -F "imgsz=640" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

!!! note "注意"

    将 `MODEL_ID` 替换为所需的模型 ID，将 `API_KEY` 替换为您的实际 API 密钥，将 `path/to/image.jpg` 替换为要运行推理的图像路径。

    如果您使用的是[专用推理 API](#专用推理-api)，也请替换 `url`。

## 参数

请参阅下表了解所有可用推理参数的完整列表。

| 参数     | 默认值   | 类型    | 描述                                                                                                                                     |
| -------- | ------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `file`   |         | `file`  | 用于推理的图像或视频文件。                                                                                                                |
| `imgsz`  | `640`   | `int`   | 输入图像的大小，有效范围为 `32` - `1280` 像素。                                                                                           |
| `conf`   | `0.25`  | `float` | 预测的置信度阈值，有效范围 `0.01` - `1.0`。                                                                                               |
| `iou`    | `0.45`  | `float` | [交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) 阈值，有效范围 `0.0` - `0.95`。                          |

## 响应

[Ultralytics HUB](https://www.ultralytics.com/hub) 推理 API 返回 JSON 响应。

### 分类

!!! example "分类模型"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n-cls.pt")

        # 运行推理
        results = model("image.jpg")

        # 以 JSON 格式打印 image.jpg 结果
        print(results[0].to_json())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://predict.ultralytics.com" \
          -H "x-api-key: API_KEY" \
          -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
          -F "file=@/path/to/image.jpg" \
          -F "imgsz=640" \
          -F "conf=0.25" \
          -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL
        url = "https://predict.ultralytics.com"

        # 请求头，使用实际的 API_KEY
        headers = {"x-api-key": "API_KEY"}

        # 推理参数（使用实际的 MODEL_ID）
        data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

        # 加载图像并发送请求
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "响应"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### 检测

!!! example "检测模型"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n.pt")

        # 运行推理
        results = model("image.jpg")

        # 以 JSON 格式打印 image.jpg 结果
        print(results[0].to_json())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://predict.ultralytics.com" \
          -H "x-api-key: API_KEY" \
          -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
          -F "file=@/path/to/image.jpg" \
          -F "imgsz=640" \
          -F "conf=0.25" \
          -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL
        url = "https://predict.ultralytics.com"

        # 请求头，使用实际的 API_KEY
        headers = {"x-api-key": "API_KEY"}

        # 推理参数（使用实际的 MODEL_ID）
        data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

        # 加载图像并发送请求
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "响应"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### 定向边界框

!!! example "定向边界框模型"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n-obb.pt")

        # 运行推理
        results = model("image.jpg")

        # 以 JSON 格式打印 image.jpg 结果
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://predict.ultralytics.com" \
          -H "x-api-key: API_KEY" \
          -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
          -F "file=@/path/to/image.jpg" \
          -F "imgsz=640" \
          -F "conf=0.25" \
          -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL
        url = "https://predict.ultralytics.com"

        # 请求头，使用实际的 API_KEY
        headers = {"x-api-key": "API_KEY"}

        # 推理参数（使用实际的 MODEL_ID）
        data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

        # 加载图像并发送请求
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "响应"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 374.85565,
                    "x2": 392.31824,
                    "x3": 412.81805,
                    "x4": 395.35547,
                    "y1": 264.40704,
                    "y2": 267.45728,
                    "y3": 150.0966,
                    "y4": 147.04634
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### 分割

!!! example "分割模型"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n-seg.pt")

        # 运行推理
        results = model("image.jpg")

        # 以 JSON 格式打印 image.jpg 结果
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://predict.ultralytics.com" \
          -H "x-api-key: API_KEY" \
          -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
          -F "file=@/path/to/image.jpg" \
          -F "imgsz=640" \
          -F "conf=0.25" \
          -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL
        url = "https://predict.ultralytics.com"

        # 请求头，使用实际的 API_KEY
        headers = {"x-api-key": "API_KEY"}

        # 推理参数（使用实际的 MODEL_ID）
        data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

        # 加载图像并发送请求
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "响应"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  },
                  "segments": {
                    "x": [
                      266.015625,
                      266.015625,
                      258.984375,
                      ...
                    ],
                    "y": [
                      110.15625,
                      113.67188262939453,
                      120.70311737060547,
                      ...
                    ]
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### 姿态估计

!!! example "姿态估计模型"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n-pose.pt")

        # 运行推理
        results = model("image.jpg")

        # 以 JSON 格式打印 image.jpg 结果
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://predict.ultralytics.com" \
          -H "x-api-key: API_KEY" \
          -F "model=https://hub.ultralytics.com/models/MODEL_ID" \
          -F "file=@/path/to/image.jpg" \
          -F "imgsz=640" \
          -F "conf=0.25" \
          -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL
        url = "https://predict.ultralytics.com"

        # 请求头，使用实际的 API_KEY
        headers = {"x-api-key": "API_KEY"}

        # 推理参数（使用实际的 MODEL_ID）
        data = {"model": "https://hub.ultralytics.com/models/MODEL_ID", "imgsz": 640, "conf": 0.25, "iou": 0.45}

        # 加载图像并发送请求
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "响应"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  },
                  "keypoints": {
                    "visible": [
                      0.9909399747848511,
                      0.8162999749183655,
                      0.9872099757194519,
                      ...
                    ],
                    "x": [
                      316.3871765136719,
                      315.9374694824219,
                      304.878173828125,
                      ...
                    ],
                    "y": [
                      156.4207763671875,
                      148.05775451660156,
                      144.93240356445312,
                      ...
                    ]
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```
