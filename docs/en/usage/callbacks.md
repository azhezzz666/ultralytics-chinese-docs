---
comments: true
description: 探索 Ultralytics 回调函数，用于训练、验证、导出和预测。学习如何为您的机器学习模型使用和自定义它们。
keywords: Ultralytics, 回调, 训练, 验证, 导出, 预测, 机器学习模型, YOLO, Python, 机器学习
---

# 回调

Ultralytics 框架支持回调，它们作为 `train`、`val`、`export` 和 `predict` 模式中战略阶段的入口点。每个回调接受一个 `Trainer`、`Validator` 或 `Predictor` 对象，具体取决于操作类型。这些对象的所有属性在文档的[参考部分](../reference/cfg/__init__.md)中有详细说明。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ENQXiK7HF5o"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 回调 | 预测、训练、验证和导出回调 | Ultralytics YOLO🚀
</p>

## 示例

### 在预测时返回额外信息

在此示例中，我们演示如何将原始帧与每个结果对象一起返回：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """将预测结果与相应的帧组合。"""
    _, image, _, _ = predictor.batch

    # 确保 image 是列表
    image = image if isinstance(image, list) else [image]

    # 将预测结果与相应的帧组合
    predictor.results = zip(predictor.results, image)


# 创建 YOLO 模型实例
model = YOLO("yolo11n.pt")

# 将自定义回调添加到模型
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# 遍历结果和帧
for result, frame in model.predict():  # 或 model.track()
    pass
```

### 使用 `on_model_save` 回调访问模型指标

此示例展示如何在使用 `on_model_save` 回调保存检查点后检索训练详情，如 best_fitness 分数、total_loss 和其他指标。

```python
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("yolo11n.pt")


def print_checkpoint_metrics(trainer):
    """在每次保存检查点后打印训练器指标和损失详情。"""
    print(
        f"模型详情\n"
        f"最佳适应度: {trainer.best_fitness}, "
        f"损失名称: {trainer.loss_names}, "  # 损失名称列表
        f"指标: {trainer.metrics}, "
        f"总损失: {trainer.tloss}"  # 总损失值
    )


if __name__ == "__main__":
    # 添加 on_model_save 回调
    model.add_callback("on_model_save", print_checkpoint_metrics)

    # 在自定义数据集上运行模型训练
    results = model.train(data="coco8.yaml", epochs=3)
```

## 所有回调

以下是所有支持的回调。更多详情请参阅回调[源代码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py)。

### 训练器回调

| 回调 | 描述 |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `on_pretrain_routine_start` | 在预训练例程开始时触发。 |
| `on_pretrain_routine_end`   | 在预训练例程结束时触发。 |
| `on_train_start`            | 在训练开始时触发。 |
| `on_train_epoch_start`      | 在每个训练[训练周期](https://www.ultralytics.com/glossary/epoch)开始时触发。 |
| `on_train_batch_start`      | 在每个训练批次开始时触发。 |
| `optimizer_step`            | 在优化器步骤期间触发。 |
| `on_before_zero_grad`       | 在梯度归零之前触发。 |
| `on_train_batch_end`        | 在每个训练批次结束时触发。 |
| `on_train_epoch_end`        | 在每个训练训练周期结束时触发。 |
| `on_fit_epoch_end`          | 在每个拟合训练周期结束时触发。 |
| `on_model_save`             | 在模型保存时触发。 |
| `on_train_end`              | 在训练过程结束时触发。 |
| `on_params_update`          | 在模型参数更新时触发。 |
| `teardown`                  | 在训练过程清理时触发。 |

### 验证器回调

| 回调 | 描述 |
| -------------------- | ------------------------------------------------ |
| `on_val_start`       | 在验证开始时触发。 |
| `on_val_batch_start` | 在每个验证批次开始时触发。 |
| `on_val_batch_end`   | 在每个验证批次结束时触发。 |
| `on_val_end`         | 在验证结束时触发。 |

### 预测器回调

| 回调 | 描述 |
| ---------------------------- | --------------------------------------------------- |
| `on_predict_start`           | 在预测过程开始时触发。 |
| `on_predict_batch_start`     | 在每个预测批次开始时触发。 |
| `on_predict_postprocess_end` | 在预测后处理结束时触发。 |
| `on_predict_batch_end`       | 在每个预测批次结束时触发。 |
| `on_predict_end`             | 在预测过程结束时触发。 |

### 导出器回调

| 回调 | 描述 |
| ----------------- | ----------------------------------------- |
| `on_export_start` | 在导出过程开始时触发。 |
| `on_export_end`   | 在导出过程结束时触发。 |

## 常见问题

### 什么是 Ultralytics 回调，如何使用它们？

Ultralytics 回调是在模型操作（如训练、验证、导出和预测）的关键阶段触发的专门入口点。这些回调允许在过程中的特定点实现自定义功能，从而增强和修改工作流。每个回调接受一个 `Trainer`、`Validator` 或 `Predictor` 对象，具体取决于操作类型。有关这些对象的详细属性，请参阅[参考部分](../reference/cfg/__init__.md)。

要使用回调，定义一个函数并使用 [`model.add_callback()`](../reference/engine/model.md#ultralytics.engine.model.Model.add_callback) 方法将其添加到模型。以下是在预测期间返回额外信息的示例：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """通过将结果与相应帧组合来处理预测批次结束；修改预测器结果。"""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

### 如何使用回调自定义 Ultralytics 训练例程？

通过在训练过程的特定阶段注入逻辑来自定义您的 Ultralytics 训练例程。Ultralytics YOLO 提供各种训练回调，如 `on_train_start`、`on_train_end` 和 `on_train_batch_end`，允许您添加自定义指标、处理或日志记录。

以下是在冻结层时使用回调将冻结层置于评估模式以防止 BN 值更改的方法：

```python
from ultralytics import YOLO


# 添加回调将冻结层置于评估模式以防止 BN 值更改
def put_in_eval_mode(trainer):
    n_layers = trainer.args.freeze
    if not isinstance(n_layers, int):
        return

    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split(".")[1]) < n_layers:
            module.eval()
            module.track_running_stats = False


model = YOLO("yolo11n.pt")
model.add_callback("on_train_epoch_start", put_in_eval_mode)
model.train(data="coco.yaml", epochs=10)
```

有关有效使用训练回调的更多详情，请参阅[训练指南](../modes/train.md)。

### 为什么应该在 Ultralytics YOLO 验证期间使用回调？

在 Ultralytics YOLO 验证期间使用回调可以通过启用自定义处理、日志记录或指标计算来增强模型评估。`on_val_start`、`on_val_batch_end` 和 `on_val_end` 等回调提供入口点来注入自定义逻辑，确保详细和全面的验证过程。

例如，要绘制所有验证批次而不仅仅是前三个：

```python
import inspect

from ultralytics import YOLO


def plot_samples(validator):
    frame = inspect.currentframe().f_back.f_back
    v = frame.f_locals
    validator.plot_val_samples(v["batch"], v["batch_i"])
    validator.plot_predictions(v["batch"], v["preds"], v["batch_i"])


model = YOLO("yolo11n.pt")
model.add_callback("on_val_batch_end", plot_samples)
model.val(data="coco.yaml")
```

有关将回调纳入验证过程的更多见解，请参阅[验证指南](../modes/val.md)。

### 如何在 Ultralytics YOLO 中为预测模式附加自定义回调？

要在 Ultralytics YOLO 中为预测模式附加自定义回调，定义一个回调函数并将其注册到预测过程。常见的预测回调包括 `on_predict_start`、`on_predict_batch_end` 和 `on_predict_end`。这些允许修改预测输出和集成额外功能，如数据日志记录或结果转换。

以下是一个示例，其中自定义回调根据是否存在特定类别的目标来保存预测：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

class_id = 2


def save_on_object(predictor):
    r = predictor.results[0]
    if class_id in r.boxes.cls:
        predictor.args.save = True
    else:
        predictor.args.save = False


model.add_callback("on_predict_postprocess_end", save_on_object)
results = model("pedestrians.mp4", stream=True, save=True)

for results in results:
    pass
```

有关更全面的用法，请参阅[预测指南](../modes/predict.md)，其中包含详细说明和其他自定义选项。

### 在 Ultralytics YOLO 中使用回调有哪些实际示例？

Ultralytics YOLO 支持各种回调的实际实现，以增强和自定义训练、验证和预测等不同阶段。一些实际示例包括：

- **记录自定义指标**：在不同阶段记录额外指标，例如在训练或验证[训练周期](https://www.ultralytics.com/glossary/epoch)结束时。
- **[数据增强](https://www.ultralytics.com/glossary/data-augmentation)**：在预测或训练批次期间实现自定义数据转换或增强。
- **中间结果**：保存中间结果，如预测或帧，以供进一步分析或可视化。

示例：在预测期间使用 `on_predict_batch_end` 将帧与预测结果组合：

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """将预测结果与帧组合。"""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

探索[回调源代码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py)了解更多选项和示例。
