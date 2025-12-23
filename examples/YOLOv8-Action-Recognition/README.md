# 使用 Ultralytics YOLOv8 的零样本动作识别（视频推理）

动作识别是一种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)技术，用于识别和分类视频中个体执行的动作。当考虑多个动作时，此过程能够实现更高级的分析。使用 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 等模型，可以实时检测和分类动作。该系统利用**零样本学习**，通过使用描述性标签来识别未经明确训练的动作。在 [Wikipedia](https://en.wikipedia.org/wiki/Zero-shot_learning) 上了解更多零样本概念。

该系统可以通过提供不同的文本标签，根据用户的偏好和需求进行自定义，以识别特定动作。

## 🎬 目录

- [步骤 1：安装所需库](#步骤-1安装所需库)
- [步骤 2：使用 Ultralytics YOLOv8 运行动作识别](#步骤-2使用-ultralytics-yolov8-运行动作识别)
- [使用选项](#使用选项)
- [常见问题](#常见问题)

## ⚙️ 步骤 1：安装所需库

使用 [Git](https://git-scm.com/) 克隆 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)，使用 [pip](https://pip.pypa.io/en/stable/) 安装依赖，然后导航（`cd`）到此本地目录以执行步骤 2 中的命令。

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入本地目录
cd ultralytics/examples/YOLOv8-Action-Recognition

# 使用 Python 包管理器安装依赖
pip install -U -r requirements.txt
```

## 🚀 步骤 2：使用 Ultralytics YOLOv8 运行动作识别

以下是运行推理的基本命令：

### 注意

动作识别模型将自动对视频中的人物执行[目标检测](https://www.ultralytics.com/glossary/object-detection)和[跟踪](https://docs.ultralytics.com/modes/track/)，并根据指定的标签对其动作进行分类。结果将在视频输出上实时显示。你可以通过在运行 [Python](https://www.python.org/) 脚本时修改 `--labels` 参数来自定义动作标签。这利用了视频分类器模型，通常来自 [Hugging Face Models](https://huggingface.co/models) 等平台。

```bash
# 使用默认视频和标签快速开始
python action_recognition.py

# 使用 YouTube 视频和自定义标签的基本用法
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --labels "dancing" "singing a song"

# 使用本地视频文件
python action_recognition.py --source path/to/video.mp4

# 使用中等大小的 YOLOv8 模型以获得更好的检测器性能
python action_recognition.py --weights yolov8m.pt

# 在 CPU 而非 GPU 上运行推理
python action_recognition.py --device cpu

# 使用 TorchVision 的不同视频分类器模型
python action_recognition.py --video-classifier-model "s3d"

# 使用 FP16（半精度）加速推理（仅适用于 HuggingFace 模型）
python action_recognition.py --fp16

# 将带有识别动作的输出视频导出为 mp4 文件
python action_recognition.py --output-path output.mp4

# 组合多个选项：特定 YouTube 源、GPU 设备 0、特定 HuggingFace 模型、自定义标签和 FP16
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device 0 --video-classifier-model "microsoft/xclip-base-patch32" --labels "dancing" "singing a song" --fp16
```


## 🛠️ 使用选项

- `--weights`：YOLO [模型权重](https://www.ultralytics.com/glossary/model-weights)文件路径（默认：`"yolov8n.pt"`）。你可以选择其他模型如 `yolov8s.pt`、`yolov8m.pt` 等。
- `--device`：CUDA 设备标识符（如 `0` 或 `0,1,2,3`）或 `cpu` 以在 [CPU](https://www.ultralytics.com/glossary/cpu) 上运行（默认：自动检测可用 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)）。
- `--source`：本地视频文件路径或 YouTube URL（默认："[rickroll](https://www.youtube.com/watch?v=dQw4w9WgXcQ)"）。
- `--output-path`：保存输出视频文件的路径（如 `output.mp4`）。如未指定，视频将在窗口中显示。
- `--crop-margin-percentage`：裁剪检测到的目标进行分类前添加的边距百分比（默认：`10`）。
- `--num-video-sequence-samples`：从序列中采样并输入分类器的视频帧数（默认：`8`）。
- `--skip-frame`：检测之间跳过的帧数以加速处理（默认：`1`）。
- `--video-cls-overlap-ratio`：发送进行分类的连续视频序列之间的重叠比率（默认：`0.25`）。
- `--fp16`：使用 [FP16（半精度）](https://www.ultralytics.com/glossary/half-precision)进行推理，可能在兼容硬件上加速（仅适用于 Hugging Face 模型）。
- `--video-classifier-model`：视频分类器模型的名称或路径（默认：`"microsoft/xclip-base-patch32"`）。可以是 Hugging Face 模型名称或 [TorchVision 模型](https://docs.pytorch.org/vision/stable/models.html)名称。
- `--labels`：零样本视频分类的文本标签列表（默认：`["dancing", "singing a song"]`）。

## 🤔 常见问题

### 1. 动作识别涉及什么？

动作识别是一种计算方法，用于识别和分类录制视频或实时流中个体执行的动作或活动。此技术广泛应用于视频分析、监控和人机交互，能够基于运动模式和上下文检测和理解人类行为。它通常将[目标跟踪](https://www.ultralytics.com/glossary/object-tracking)与分类相结合。在 [arxiv](https://arxiv.org/) 上探索更多视频分类研究。

### 2. 动作识别是否支持自定义动作标签？

是的，支持自定义动作标签。`action_recognition.py` 脚本允许用户为**零样本视频分类**指定自己的自定义标签。这通过 `--labels` 参数完成。例如：

```bash
python action_recognition.py --source https://www.youtube.com/watch?v=dQw4w9WgXcQ --labels "walking" "running" "jumping"
```

你可以调整这些标签以匹配你希望系统在视频中识别的特定动作。然后系统将尝试使用从大型数据集中获得的理解，基于这些自定义标签对检测到的动作进行分类。

此外，你可以在不同的视频分类模型之间选择：

1.  **Hugging Face 模型**：你可以使用 Hugging Face Hub 上任何兼容的视频分类模型。默认是：
    - `"microsoft/xclip-base-patch32"`
2.  **TorchVision 模型**：这些模型不支持使用自定义文本标签的零样本分类，但提供预训练的分类能力。选项包括：
    - `"s3d"`
    - `"r3d_18"`
    - `"swin3d_t"`
    - `"swin3d_b"`
    - `"mvit_v1_b"`
    - `"mvit_v2_s"`

### 3. 为什么将动作识别与 YOLOv8 结合？

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 擅长在视频流中进行快速准确的[目标检测](https://docs.ultralytics.com/tasks/detect/)和跟踪。将其与动作识别结合，使系统不仅能定位个体（使用 YOLOv8 的检测能力），还能理解他们_正在做什么_。这种协同作用为视频内容提供了更丰富的分析，对于自动监控、体育分析或人机交互等应用至关重要。查看我们关于[目标检测和跟踪](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8)的博客文章。

### 4. 我可以使用其他 YOLO 版本吗？

当然可以！虽然本示例默认使用 `yolov8n.pt`，但你可以灵活地使用 `--weights` 选项指定不同的 Ultralytics YOLO 模型权重。例如，你可以使用 `yolov8s.pt`、`yolov8m.pt`、`yolov8l.pt` 或 `yolov8x.pt` 以获得更高的检测精度，但会牺牲推理速度。如果适用，你甚至可以使用为其他任务训练的模型，尽管检测模型在这里是标准的。查看 [Ultralytics 文档](https://docs.ultralytics.com/)了解可用模型及其性能指标。

---

我们希望本指南能帮助你使用 Ultralytics YOLOv8 实现零样本动作识别！随时探索代码并尝试不同的选项。如果你遇到问题或有建议，请考虑在 [GitHub 仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多详情。
