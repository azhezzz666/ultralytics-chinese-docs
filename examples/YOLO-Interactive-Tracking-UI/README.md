# Ultralytics YOLO 交互式目标跟踪界面 🚀

一个基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 和 [OpenCV](https://opencv.org/) 构建的实时[目标检测](https://docs.ultralytics.com/tasks/detect/)和[跟踪](https://docs.ultralytics.com/modes/track/)界面，专为交互式演示和无缝集成跟踪叠加层而设计。无论你是刚开始接触目标跟踪还是希望增强其附加功能，本项目都提供了坚实的基础。

https://github.com/user-attachments/assets/723e919e-555b-4cca-8e60-18e711d4f3b2

## ✨ 功能特性

- 实时目标检测和视觉跟踪
- 点击任意检测到的目标即可开始跟踪
- 活动跟踪目标显示瞄准线和粗体[边界框](https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes)
- 非跟踪目标显示虚线框
- [终端实时输出](https://docs.ultralytics.com/guides/view-results-in-terminal/)：目标 ID、标签、[置信度](https://www.ultralytics.com/glossary/confidence)和中心坐标
- 可调节的目标跟踪算法（[ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/)、[BoT-SORT](https://docs.ultralytics.com/reference/trackers/bot_sort/)）
- 支持：
  - [PyTorch](https://pytorch.org/) `.pt` 模型（适用于 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) 或支持 [CUDA](https://developer.nvidia.com/cuda) 的桌面设备等 GPU 设备）
  - [NCNN](https://docs.ultralytics.com/integrations/ncnn/) `.param + .bin` 模型（适用于 [Raspberry Pi](https://www.raspberrypi.org/) 或 ARM 开发板等纯 CPU 设备）

## 🏗️ 项目结构

```
YOLO-Interactive-Tracking-UI/
├── interactive_tracker.py   # 主 Python 跟踪界面脚本
└── README.md                # 你正在阅读的文件！
```

## 💻 硬件与模型兼容性

| 平台             | 模型格式           | 示例模型             | GPU 加速     | 备注                            |
| ---------------- | ------------------ | -------------------- | ------------ | ------------------------------- |
| Raspberry Pi 4/5 | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌ 仅 CPU    | Pi/ARM 推荐格式                 |
| Jetson Nano      | PyTorch (.pt)      | `yolov8n.pt`         | ✅ CUDA      | 可实现实时性能                  |
| 带 GPU 的桌面    | PyTorch (.pt)      | `yolov8s.pt`         | ✅ CUDA      | 最佳性能                        |
| 纯 CPU 笔记本    | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌           | 性能尚可（约 10-15 FPS）        |

_注意：性能可能因具体硬件、模型复杂度和输入分辨率而异。_

## 🛠️ 安装

### 基础依赖

安装核心 `ultralytics` 包：

```bash
pip install ultralytics
```

> **提示：** 建议使用 `venv` 或 [`conda`](https://docs.ultralytics.com/guides/conda-quickstart/)（推荐）等虚拟环境来管理依赖。

> **GPU 支持：** 根据你的系统和 CUDA 版本，按照官方指南安装 PyTorch：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 🚀 快速开始

### 步骤 1：下载、转换或指定模型

- 对于预训练的 Ultralytics YOLO [模型](https://docs.ultralytics.com/models/)（如 `yolo11s.pt` 或 `yolov8s.pt`），只需在脚本参数（`model_file`）中指定模型名称。这些模型将自动下载并缓存。你也可以从 [Ultralytics Assets Releases](https://github.com/ultralytics/assets/releases) 手动下载并放置在项目文件夹中。
- 如果使用自定义训练的 YOLO 模型，请确保模型文件在项目文件夹中或提供其相对路径。
- 对于纯 CPU 设备，使用 Ultralytics `export` 模式将所选模型（如 `yolov8n.pt`）导出为 [NCNN 格式](https://docs.ultralytics.com/integrations/ncnn/)。

- **支持的格式：**
  - `yolo11s.pt`（用于带 PyTorch 的 GPU）
  - `yolov8n_ncnn_model`（包含 `.param` 和 `.bin` 文件的目录，用于带 NCNN 的 CPU）

### 步骤 2：配置脚本

编辑 `interactive_tracker.py` 顶部的全局参数：

```python
# --- 配置 ---
enable_gpu = False  # 如果使用 CUDA 和 PyTorch 模型，设置为 True
model_file = "yolo11s.pt"  # 模型文件路径（GPU 用 .pt，CPU 用 _ncnn_model 目录）
show_fps = True  # 在左上角显示当前 FPS
show_conf = False  # 显示每个检测的置信度分数
save_video = False  # 设置为 True 以保存输出视频流
video_output_path = "interactive_tracker_output.avi"  # 输出视频文件名

# --- 检测与跟踪参数 ---
conf = 0.3  # 目标检测的最小置信度阈值
iou = 0.3  # 非极大值抑制（NMS）的 IoU 阈值
max_det = 20  # 每帧检测的最大目标数

tracker = "bytetrack.yaml"  # 跟踪器配置：'bytetrack.yaml' 或 'botsort.yaml'
track_args = {
    "persist": True,  # 跨帧保持跟踪历史
    "verbose": False,  # 抑制详细的跟踪器调试输出
}

window_name = "Ultralytics YOLO Interactive Tracking"  # OpenCV 显示窗口名称
# --- 配置结束 ---
```

- **`enable_gpu`**：如果你有兼容 CUDA 的 GPU 并使用 `.pt` 模型，设置为 `True`。对于 NCNN 模型或纯 CPU 执行，保持 `False`。
- **`model_file`**：确保根据 `enable_gpu` 指向正确的模型文件或目录。
- **`conf`**：调整[置信度](https://www.ultralytics.com/glossary/confidence)阈值。较低的值会检测更多目标，但可能增加误检。
- **`iou`**：设置[非极大值抑制（NMS）](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)的[交并比（IoU）](https://www.ultralytics.com/glossary/intersection-over-union-iou)阈值。较高的值允许更多重叠框。
- **`tracker`**：在可用的跟踪器配置文件之间选择（[ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/)、[BoT-SORT](https://docs.ultralytics.com/reference/trackers/bot_sort/)）。

### 步骤 3：运行目标跟踪

从终端执行脚本：

```bash
python interactive_tracker.py
```

### 控制方式

- 🖱️ **左键点击**检测到的目标边界框以开始跟踪。
- 🔄 按 **`c`** 键取消当前跟踪并选择新目标。
- ❌ 按 **`q`** 键退出应用程序。

### 保存输出视频（可选）

如果要录制跟踪会话，在配置中启用 `save_video` 选项：

```python
save_video = True  # 启用视频录制
video_output_path = "output.avi"  # 自定义输出文件名（如 .mp4、.avi）
```

当你按 `q` 退出应用程序时，视频文件将保存在项目工作目录中。

## 👤 作者

- **Alireza**
- [LinkedIn 联系](https://www.linkedin.com/in/alireza787b)
- 发布日期：2025-04-01

## 📜 许可证与免责声明

本项目基于 [AGPL-3.0 许可证](https://www.ultralytics.com/legal/agpl-3-0-software-license)发布。完整许可详情请参阅 [Ultralytics 许可页面](https://www.ultralytics.com/license)。

本软件按"原样"提供，仅用于教育和演示目的。请负责任地使用，风险自负。作者不对滥用或意外后果承担任何责任。

## 🤝 贡献

欢迎贡献、反馈和错误报告！如果你有改进或建议，请随时在原始仓库上提交 issue 或 pull request。
