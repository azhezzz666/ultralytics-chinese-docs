<div align="center">
  <p>
    <a href="https://www.ultralytics.com/" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>
</div>
<br>

[Ultralytics](https://www.ultralytics.com/) 基于多年在计算机视觉和人工智能领域的基础研究，创造了尖端的、最先进的 (SOTA) [YOLO 模型](https://www.ultralytics.com/yolo)。我们的模型不断更新以提高性能和灵活性，具有**速度快**、**精度高**和**易于使用**的特点。

## 📄 文档

请参阅下文了解快速安装和使用示例。有关训练、验证、预测和部署的全面指南，请参阅我们的完整 [Ultralytics 文档](https://docs.ultralytics.com/)。

### 安装

```bash
pip install ultralytics
```

### 使用方法

#### CLI

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

#### Python

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

## 📜 许可证

Ultralytics 提供两种许可选项：

- **AGPL-3.0 许可证**：适合学生、研究人员和爱好者。
- **Ultralytics 企业许可证**：专为商业用途设计。

## 📞 联系方式

- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Discord](https://discord.com/invite/ultralytics)
- [Ultralytics 社区论坛](https://community.ultralytics.com/)
