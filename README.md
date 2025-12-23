# Ultralytics YOLO 中文文档

<div align="center">
  <a href="https://www.ultralytics.com/" target="_blank">
    <img width="800" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner">
  </a>
</div>

<div align="center">
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/官方仓库-ultralytics-blue?logo=github" alt="Official Repo"></a>
  <a href="https://docs.ultralytics.com/zh/"><img src="https://img.shields.io/badge/官方文档-中文-green" alt="Official Docs"></a>
  <a href="https://github.com/azhezzz666/ultralytics-chinese-docs/blob/main/LICENSE"><img src="https://img.shields.io/badge/许可证-AGPL--3.0-orange" alt="License"></a>
</div>

## 📖 项目简介

本仓库是 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 官方文档的**简体中文翻译版本**，基于 **v8.3.241** 版本。

旨在帮助中文开发者更好地理解和使用 Ultralytics YOLO 系列模型，包括 YOLOv8、YOLO11 等最新版本。

## 📁 目录结构

```
├── docs/                    # 文档目录
│   ├── en/                  # 英文文档（已翻译为中文）
│   │   ├── datasets/        # 数据集文档
│   │   ├── guides/          # 使用指南
│   │   ├── help/            # 帮助文档
│   │   ├── hub/             # Ultralytics HUB 文档
│   │   ├── integrations/    # 集成文档
│   │   ├── models/          # 模型文档
│   │   ├── modes/           # 模式文档
│   │   ├── reference/       # API 参考
│   │   ├── solutions/       # 解决方案
│   │   ├── tasks/           # 任务文档
│   │   ├── usage/           # 使用说明
│   │   └── yolov5/          # YOLOv5 文档
│   └── macros/              # 宏定义文件
├── examples/                # 示例代码
│   ├── *.ipynb              # Jupyter 笔记本教程
│   └── */                   # 各种语言/框架的示例
├── CONTRIBUTING.md          # 贡献指南
└── README.zh-CN.md          # 中文 README
```

## ✨ 主要内容

### 📚 文档翻译

- **模型文档**：YOLO11、YOLOv8、YOLOv5、SAM、RT-DETR 等
- **任务文档**：目标检测、实例分割、图像分类、姿态估计、旋转边界框
- **模式文档**：训练、验证、预测、导出、跟踪、基准测试
- **集成文档**：TensorRT、ONNX、OpenVINO、CoreML、TFLite 等
- **指南文档**：数据集准备、模型部署、性能优化等

### 📓 示例笔记本

- `tutorial.ipynb` - YOLO11 入门教程
- `heatmaps.ipynb` - 热力图可视化
- `object_counting.ipynb` - 目标计数
- `object_tracking.ipynb` - 目标跟踪
- `hub.ipynb` - Ultralytics HUB 使用

### 💻 代码示例

- Python/ONNX Runtime 推理
- C++/OpenVINO 推理
- Rust/ONNX Runtime 推理
- TensorRT 部署
- 更多...

## 🚀 快速开始

### 安装

```bash
pip install ultralytics
```

### 基本使用

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo11n.pt')

# 对图像进行预测
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
```

### CLI 使用

```bash
# 预测
yolo predict model=yolo11n.pt source='path/to/image.jpg'

# 训练
yolo train model=yolo11n.pt data=coco8.yaml epochs=100

# 验证
yolo val model=yolo11n.pt data=coco8.yaml

# 导出
yolo export model=yolo11n.pt format=onnx
```

## 📌 版本信息

- **基于版本**：Ultralytics v8.3.241
- **翻译日期**：2024年12月
- **翻译语言**：简体中文

## 🔗 相关链接

- [Ultralytics 官方仓库](https://github.com/ultralytics/ultralytics)
- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [Ultralytics 中文文档](https://docs.ultralytics.com/zh/)
- [Ultralytics HUB](https://hub.ultralytics.com/)

## 📜 许可证

本项目遵循 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。

## 🙏 致谢

感谢 [Ultralytics](https://www.ultralytics.com/) 团队开发的优秀开源项目！

---

<div align="center">
  <p>如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！</p>
</div>
