---
comments: true
description: 学习如何将 YOLOv5 模型导出为 TFLite、ONNX、CoreML 和 TensorRT 等各种格式。通过我们的分步指南提高模型效率和部署灵活性。
keywords: YOLOv5 导出, TFLite, ONNX, CoreML, TensorRT, 模型转换, YOLOv5 教程, PyTorch 导出
---

# TFLite、ONNX、CoreML、TensorRT 导出

📚 本指南介绍如何将训练好的 YOLOv5 🚀 模型从 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 导出为各种部署格式，包括 ONNX、TensorRT、CoreML 等。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

有关 [TensorRT](https://developer.nvidia.com/tensorrt) 导出示例（需要 GPU），请参阅我们的 Colab [notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=VTRwsvA9u7ln&line=2&uniqifier=1) 附录部分。<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>

## 支持的导出格式

YOLOv5 推理官方支持 12 种格式：

!!! tip "性能提示"

    - 导出为 ONNX 或 OpenVINO 可获得高达 3 倍的 CPU 加速。参见 [CPU 基准测试](https://github.com/ultralytics/yolov5/pull/6613)。
    - 导出为 TensorRT 可获得高达 5 倍的 GPU 加速。参见 [GPU 基准测试](https://github.com/ultralytics/yolov5/pull/6963)。

| 格式                                                         | `export.py --include` | 模型                      |
| :----------------------------------------------------------- | :-------------------- | :------------------------ |
| [PyTorch](https://pytorch.org/)                              | -                     | `yolov5s.pt`              |
| [TorchScript](../../integrations/torchscript.md)             | `torchscript`         | `yolov5s.torchscript`     |
| [ONNX](../../integrations/onnx.md)                           | `onnx`                | `yolov5s.onnx`            |
| [OpenVINO](../../integrations/openvino.md)                   | `openvino`            | `yolov5s_openvino_model/` |
| [TensorRT](../../integrations/tensorrt.md)                   | `engine`              | `yolov5s.engine`          |
| [CoreML](../../integrations/coreml.md)                       | `coreml`              | `yolov5s.mlmodel`         |
| [TensorFlow SavedModel](../../integrations/tf-savedmodel.md) | `saved_model`         | `yolov5s_saved_model/`    |
| [TensorFlow GraphDef](../../integrations/tf-graphdef.md)     | `pb`                  | `yolov5s.pb`              |
| [TensorFlow Lite](../../integrations/tflite.md)              | `tflite`              | `yolov5s.tflite`          |
| [TensorFlow Edge TPU](../../integrations/edge-tpu.md)        | `edgetpu`             | `yolov5s_edgetpu.tflite`  |
| [TensorFlow.js](../../integrations/tfjs.md)                  | `tfjs`                | `yolov5s_web_model/`      |
| [PaddlePaddle](../../integrations/paddlepaddle.md)           | `paddle`              | `yolov5s_paddle_model/`   |

## 基准测试

以下基准测试在 Colab Pro 上使用 YOLOv5 教程 notebook 运行 <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>。要复现：

```bash
python benchmarks.py --weights yolov5s.pt --imgsz 640 --device 0
```

### Colab Pro V100 GPU

```
benchmarks: weights=/content/yolov5/yolov5s.pt, imgsz=640, batch_size=1, data=/content/yolov5/data/coco128.yaml, device=0, half=False, test=False
Checking setup...
YOLOv5 🚀 v6.1-135-g7926afc torch 1.10.0+cu111 CUDA:0 (Tesla V100-SXM2-16GB, 16160MiB)
Setup complete ✅ (8 CPUs, 51.0 GB RAM, 46.7/166.8 GB disk)

Benchmarks complete (458.07s)
                   Format  mAP@0.5:0.95  Inference time (ms)
0                 PyTorch        0.4623                10.19
1             TorchScript        0.4623                 6.85
2                    ONNX        0.4623                14.63
3                OpenVINO           NaN                  NaN
4                TensorRT        0.4617                 1.89
5                  CoreML           NaN                  NaN
6   TensorFlow SavedModel        0.4623                21.28
7     TensorFlow GraphDef        0.4623                21.22
8         TensorFlow Lite           NaN                  NaN
9     TensorFlow Edge TPU           NaN                  NaN
10          TensorFlow.js           NaN                  NaN
```


### Colab Pro CPU

```
benchmarks: weights=/content/yolov5/yolov5s.pt, imgsz=640, batch_size=1, data=/content/yolov5/data/coco128.yaml, device=cpu, half=False, test=False
Checking setup...
YOLOv5 🚀 v6.1-135-g7926afc torch 1.10.0+cu111 CPU
Setup complete ✅ (8 CPUs, 51.0 GB RAM, 41.5/166.8 GB disk)

Benchmarks complete (241.20s)
                   Format  mAP@0.5:0.95  Inference time (ms)
0                 PyTorch        0.4623               127.61
1             TorchScript        0.4623               131.23
2                    ONNX        0.4623                69.34
3                OpenVINO        0.4623                66.52
4                TensorRT           NaN                  NaN
5                  CoreML           NaN                  NaN
6   TensorFlow SavedModel        0.4623               123.79
7     TensorFlow GraphDef        0.4623               121.57
8         TensorFlow Lite        0.4623               316.61
9     TensorFlow Edge TPU           NaN                  NaN
10          TensorFlow.js           NaN                  NaN
```

## 导出训练好的 YOLOv5 模型

此命令将预训练的 YOLOv5s 模型导出为 TorchScript 和 ONNX 格式。`yolov5s.pt` 是"小型"模型，是第二小的可用模型。其他选项有 `yolov5n.pt`、`yolov5m.pt`、`yolov5l.pt` 和 `yolov5x.pt`，以及它们的 P6 对应版本如 `yolov5s6.pt`，或您自己的自定义训练检查点如 `runs/exp/weights/best.pt`。有关所有可用模型的详细信息，请参阅我们的 README [表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)。

```bash
python export.py --weights yolov5s.pt --include torchscript onnx
```

!!! tip

    添加 `--half` 以 FP16 半[精度](https://www.ultralytics.com/glossary/precision)导出模型，以获得更小的文件大小

输出：

```text
export: data=data/coco128.yaml, weights=['yolov5s.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['torchscript', 'onnx']
YOLOv5 🚀 v6.2-104-ge3e5122 Python-3.8.0 torch-1.12.1+cu113 CPU

Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 274MB/s]

Fusing layers...
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients

PyTorch: starting from yolov5s.pt with output shape (1, 25200, 85) (14.1 MB)

TorchScript: starting export with torch 1.12.1+cu113...
TorchScript: export success ✅ 1.7s, saved as yolov5s.torchscript (28.1 MB)

ONNX: starting export with onnx 1.12.0...
ONNX: export success ✅ 2.3s, saved as yolov5s.onnx (28.0 MB)

Export complete (5.5s)
Results saved to /content/yolov5
Detect:          python detect.py --weights yolov5s.onnx
Validate:        python val.py --weights yolov5s.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')
Visualize:       https://netron.app/
```

3 个导出的模型将与原始 PyTorch 模型一起保存：

<p align="center"><img width="700" src="https://github.com/ultralytics/docs/releases/download/0/yolo-export-locations.avif" alt="YOLO 导出位置"></p>

推荐使用 [Netron Viewer](https://github.com/lutzroeder/netron) 可视化导出的模型：

<p align="center"><img width="850" src="https://github.com/ultralytics/docs/releases/download/0/yolo-model-visualization.avif" alt="YOLO 模型可视化"></p>

## 导出模型使用示例

`detect.py` 在导出的模型上运行推理：

```bash
python detect.py --weights yolov5s.pt             # PyTorch
python detect.py --weights yolov5s.torchscript    # TorchScript
python detect.py --weights yolov5s.onnx           # ONNX Runtime 或 OpenCV DNN（使用 dnn=True）
python detect.py --weights yolov5s_openvino_model # OpenVINO
python detect.py --weights yolov5s.engine         # TensorRT
python detect.py --weights yolov5s.mlmodel        # CoreML（仅 macOS）
python detect.py --weights yolov5s_saved_model    # TensorFlow SavedModel
python detect.py --weights yolov5s.pb             # TensorFlow GraphDef
python detect.py --weights yolov5s.tflite         # TensorFlow Lite
python detect.py --weights yolov5s_edgetpu.tflite # TensorFlow Edge TPU
python detect.py --weights yolov5s_paddle_model   # PaddlePaddle
```

`val.py` 在导出的模型上运行验证：

```bash
python val.py --weights yolov5s.pt             # PyTorch
python val.py --weights yolov5s.torchscript    # TorchScript
python val.py --weights yolov5s.onnx           # ONNX Runtime 或 OpenCV DNN（使用 dnn=True）
python val.py --weights yolov5s_openvino_model # OpenVINO
python val.py --weights yolov5s.engine         # TensorRT
python val.py --weights yolov5s.mlmodel        # CoreML（仅 macOS）
python val.py --weights yolov5s_saved_model    # TensorFlow SavedModel
python val.py --weights yolov5s.pb             # TensorFlow GraphDef
python val.py --weights yolov5s.tflite         # TensorFlow Lite
python val.py --weights yolov5s_edgetpu.tflite # TensorFlow Edge TPU
python val.py --weights yolov5s_paddle_model   # PaddlePaddle
```

使用 PyTorch Hub 加载导出的 YOLOv5 模型：

```python
import torch

# 模型
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.pt")
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.torchscript")  # TorchScript
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.onnx")  # ONNX Runtime
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_openvino_model")  # OpenVINO
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.engine")  # TensorRT
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.mlmodel")  # CoreML（仅 macOS）
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_saved_model")  # TensorFlow SavedModel
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.pb")  # TensorFlow GraphDef
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.tflite")  # TensorFlow Lite
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_edgetpu.tflite")  # TensorFlow Edge TPU
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s_paddle_model")  # PaddlePaddle

# 图像
img = "https://ultralytics.com/images/zidane.jpg"  # 或文件、Path、PIL、OpenCV、numpy、列表

# 推理
results = model(img)

# 结果
results.print()  # 或 .show()、.save()、.crop()、.pandas() 等
```

## OpenCV DNN 推理

使用 ONNX 模型进行 [OpenCV](https://www.ultralytics.com/glossary/opencv) 推理：

```bash
python export.py --weights yolov5s.pt --include onnx

python detect.py --weights yolov5s.onnx --dnn # 检测
python val.py --weights yolov5s.onnx --dnn    # 验证
```

## C++ 推理

YOLOv5 OpenCV DNN C++ 在导出的 ONNX 模型上的推理示例：

- [https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp](https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp)
- [https://github.com/doleron/yolov5-opencv-cpp-python](https://github.com/doleron/yolov5-opencv-cpp-python)

YOLOv5 OpenVINO C++ 推理示例：

- [https://github.com/dacquaviva/yolov5-openvino-cpp-python](https://github.com/dacquaviva/yolov5-openvino-cpp-python)
- [https://github.com/UNeedCryDear/yolov5-seg-opencv-dnn-cpp](https://github.com/UNeedCryDear/yolov5-seg-opencv-onnxruntime-cpp)

## TensorFlow.js Web 浏览器推理

- [https://aukerul-shuvo.github.io/YOLOv5_TensorFlow-JS/](https://aukerul-shuvo.github.io/YOLOv5_TensorFlow-JS/)

## 支持的环境

Ultralytics 提供一系列开箱即用的环境，每个环境都预装了 [CUDA](https://developer.nvidia.com/cuda)、[CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 等基本依赖项，以便快速启动您的项目。

- **免费 GPU Notebook**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](../environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](../environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均成功通过。这些 CI 测试严格检查 YOLOv5 在各个关键方面的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致可靠运行，测试每 24 小时进行一次，并在每次新提交时进行。
