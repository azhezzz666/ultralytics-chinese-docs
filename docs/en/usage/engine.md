---
comments: true
description: 学习如何为特定任务自定义 Ultralytics YOLO 训练器。包含 Python 示例的分步说明，以实现最佳模型性能。
keywords: Ultralytics, YOLO, 训练器自定义, Python, 机器学习, 人工智能, 模型训练, DetectionTrainer, 自定义模型
---

# 高级自定义

Ultralytics YOLO 命令行和 Python 接口都是构建在基础引擎执行器之上的高级抽象。本指南重点介绍 `Trainer` 引擎，解释如何根据您的特定需求进行自定义。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=104"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：高级自定义
</p>

## BaseTrainer

`BaseTrainer` 类提供了一个可适应各种任务的通用训练例程。通过覆盖特定函数或操作来自定义它，同时遵循所需的格式。例如，通过覆盖以下函数来集成您自己的自定义模型和数据加载器：

- `get_model(cfg, weights)`：构建要训练的模型。
- `get_dataloader()`：构建数据加载器。

有关更多详情和源代码，请参阅 [`BaseTrainer` 参考](../reference/engine/trainer.md)。

## DetectionTrainer

以下是如何使用和自定义 Ultralytics YOLO `DetectionTrainer`：

```python
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # 获取最佳模型
```

### 自定义 DetectionTrainer

要训练不直接支持的自定义检测模型，请重载现有的 `get_model` 功能：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """根据配置和权重文件加载自定义检测模型。"""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

通过修改[损失函数](https://www.ultralytics.com/glossary/loss-function)或添加[回调](callbacks.md)来进一步自定义训练器，例如每 10 个[训练周期](https://www.ultralytics.com/glossary/epoch)将模型上传到 Google Drive。以下是示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self):
        """初始化损失函数并添加回调，每 10 个训练周期将模型上传到 Google Drive。"""
        ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """返回使用指定配置和权重配置的自定义检测模型实例。"""
        return MyCustomModel(...)


# 上传模型权重的回调
def log_model(trainer):
    """记录训练器使用的最后一个模型权重的路径。"""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # 添加到现有回调
trainer.train()
```

有关回调触发事件和入口点的更多信息，请参阅[回调指南](../usage/callbacks.md)。

## 其他引擎组件

类似地自定义其他组件，如 `Validators` 和 `Predictors`。有关更多信息，请参阅 [Validators](../reference/engine/validator.md) 和 [Predictors](../reference/engine/predictor.md) 的文档。

## 将 YOLO 与自定义训练器一起使用

`YOLO` 模型类为 Trainer 类提供了高级封装。您可以利用此架构在机器学习工作流中获得更大的灵活性：

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# 创建自定义训练器
class MyCustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """自定义代码实现。"""
        ...


# 初始化 YOLO 模型
model = YOLO("yolo11n.pt")

# 使用自定义训练器进行训练
results = model.train(trainer=MyCustomTrainer, data="coco8.yaml", epochs=3)
```

这种方法允许您在自定义底层训练过程以满足特定需求的同时，保持 YOLO 接口的简洁性。

## 常见问题

### 如何为特定任务自定义 Ultralytics YOLO DetectionTrainer？

通过覆盖其方法来自定义 `DetectionTrainer` 以适应您的自定义模型和数据加载器。首先从 `DetectionTrainer` 继承并重新定义 `get_model` 等方法来实现自定义功能。以下是示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """根据配置和权重文件加载自定义检测模型。"""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # 获取最佳模型
```

有关进一步自定义，如更改[损失函数](https://www.ultralytics.com/glossary/loss-function)或添加[回调](https://www.ultralytics.com/glossary/callback)，请参阅[回调指南](../usage/callbacks.md)。

### Ultralytics YOLO 中 BaseTrainer 的关键组件是什么？

`BaseTrainer` 作为训练例程的基础，可通过覆盖其通用方法来自定义各种任务。关键组件包括：

- `get_model(cfg, weights)`：构建要训练的模型。
- `get_dataloader()`：构建数据加载器。
- `preprocess_batch()`：在模型前向传递之前处理批次预处理。
- `set_model_attributes()`：根据数据集信息设置模型属性。
- `get_validator()`：返回用于模型评估的验证器。

有关自定义和源代码的更多详情，请参阅 [`BaseTrainer` 参考](../reference/engine/trainer.md)。

### 如何向 Ultralytics YOLO DetectionTrainer 添加回调？

添加回调以监控和修改 `DetectionTrainer` 中的训练过程。以下是如何添加回调以在每个训练[训练周期](https://www.ultralytics.com/glossary/epoch)后记录模型权重：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


# 上传模型权重的回调
def log_model(trainer):
    """记录训练器使用的最后一个模型权重的路径。"""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = DetectionTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # 添加到现有回调
trainer.train()
```

有关回调事件和入口点的更多详情，请参阅[回调指南](../usage/callbacks.md)。

### 为什么应该使用 Ultralytics YOLO 进行模型训练？

Ultralytics YOLO 在强大的引擎执行器之上提供高级抽象，非常适合快速开发和自定义。主要优势包括：

- **易于使用**：命令行和 Python 接口都简化了复杂任务。
- **性能**：针对实时[目标检测](https://www.ultralytics.com/glossary/object-detection)和各种视觉 AI 应用进行了优化。
- **自定义**：易于扩展以支持自定义模型、[损失函数](https://www.ultralytics.com/glossary/loss-function)和数据加载器。
- **模块化**：组件可以独立修改而不影响整个流水线。
- **集成**：与机器学习生态系统中的流行框架和工具无缝协作。

通过探索 [Ultralytics YOLO](https://www.ultralytics.com/yolo) 主页了解更多关于 YOLO 功能的信息。

### 我可以将 Ultralytics YOLO DetectionTrainer 用于非标准模型吗？

是的，`DetectionTrainer` 高度灵活，可针对非标准模型进行自定义。从 `DetectionTrainer` 继承并重载方法以支持您特定模型的需求。以下是一个简单示例：

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """加载自定义检测模型。"""
        ...


trainer = CustomDetectionTrainer(overrides={...})
trainer.train()
```

有关全面说明和示例，请查看 [`DetectionTrainer` 参考](../reference/models/yolo/detect/train.md)。
