---
comments: true
description: 学习如何确保 Python 中 YOLO 模型推理的线程安全。避免竞态条件，使用最佳实践可靠地运行多线程任务。
keywords: YOLO 模型, 线程安全, Python 线程, 模型推理, 并发, 竞态条件, 多线程, 并行, Python GIL
---

# YOLO 模型的线程安全推理

在多线程环境中运行 YOLO 模型需要仔细考虑以确保线程安全。Python 的 `threading` 模块允许您同时运行多个线程，但在跨这些线程使用 YOLO 模型时，需要注意一些重要的安全问题。本页将指导您创建线程安全的 YOLO 模型推理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/jMbvN6uCIos"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Python 中使用 Ultralytics YOLO 模型执行线程安全推理 | 多线程 🚀
</p>

## 理解 Python 线程

Python 线程是一种并行形式，允许您的程序同时运行多个操作。然而，Python 的全局解释器锁 (GIL) 意味着一次只有一个线程可以执行 Python 字节码。

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/single-vs-multi-thread-examples.avif" alt="单线程与多线程示例">
</p>

虽然这听起来像是一个限制，但线程仍然可以提供并发性，特别是对于 I/O 密集型操作或使用释放 GIL 的操作时，例如 YOLO 底层 C 库执行的操作。

## 共享模型实例的危险

在线程外部实例化 YOLO 模型并在多个线程之间共享此实例可能会导致竞态条件，其中模型的内部状态由于并发访问而被不一致地修改。当模型或其组件持有非线程安全设计的状态时，这尤其成问题。

### 非线程安全示例：单个模型实例

在 Python 中使用线程时，识别可能导致并发问题的模式很重要。以下是您应该避免的：在多个线程之间共享单个 YOLO 模型实例。

```python
# 不安全：在线程之间共享单个模型实例
from threading import Thread

from ultralytics import YOLO

# 在线程外部实例化模型
shared_model = YOLO("yolo11n.pt")


def predict(image_path):
    """使用预加载的 YOLO 模型预测图像中的对象，接受图像路径字符串作为参数。"""
    results = shared_model.predict(image_path)
    # 处理结果


# 启动共享相同模型实例的线程
Thread(target=predict, args=("image1.jpg",)).start()
Thread(target=predict, args=("image2.jpg",)).start()
```

在上面的示例中，`shared_model` 被多个线程使用，这可能导致不可预测的结果，因为 `predict` 可能被多个线程同时执行。

### 非线程安全示例：多个模型实例

类似地，以下是使用多个 YOLO 模型实例的不安全模式：

```python
# 不安全：在线程之间共享多个模型实例仍可能导致问题
from threading import Thread

from ultralytics import YOLO

# 在线程外部实例化多个模型
shared_model_1 = YOLO("yolo11n_1.pt")
shared_model_2 = YOLO("yolo11n_2.pt")


def predict(model, image_path):
    """使用指定的 YOLO 模型对图像运行预测，返回结果。"""
    results = model.predict(image_path)
    # 处理结果


# 使用单独的模型实例启动线程
Thread(target=predict, args=(shared_model_1, "image1.jpg")).start()
Thread(target=predict, args=(shared_model_2, "image2.jpg")).start()
```

即使有两个单独的模型实例，并发问题的风险仍然存在。如果 `YOLO` 的内部实现不是线程安全的，使用单独的实例可能无法防止竞态条件，特别是如果这些实例共享任何非线程本地的底层资源或状态。

## 线程安全推理

要执行线程安全推理，您应该在每个线程内实例化单独的 YOLO 模型。这确保每个线程都有自己的隔离模型实例，消除了竞态条件的风险。

### 线程安全示例

以下是如何在每个线程内实例化 YOLO 模型以实现安全的并行推理：

```python
# 安全：在每个线程内实例化单个模型
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """以线程安全的方式使用新的 YOLO 模型实例对图像进行预测；接受图像路径作为输入。"""
    local_model = YOLO("yolo11n.pt")
    results = local_model.predict(image_path)
    # 处理结果


# 启动各自拥有自己模型实例的线程
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

在此示例中，每个线程创建自己的 `YOLO` 实例。这防止任何线程干扰另一个线程的模型状态，从而确保每个线程安全地执行推理，而不会与其他线程产生意外交互。

## 使用 ThreadingLocked 装饰器

Ultralytics 提供了一个 `ThreadingLocked` 装饰器，可用于确保函数的线程安全执行。此装饰器使用锁来确保一次只有一个线程可以执行被装饰的函数。

```python
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked

# 创建模型实例
model = YOLO("yolo11n.pt")


# 装饰 predict 方法使其线程安全
@ThreadingLocked()
def thread_safe_predict(image_path):
    """使用共享模型实例的线程安全预测。"""
    results = model.predict(image_path)
    return results


# 现在您可以安全地从多个线程调用此函数
```

`ThreadingLocked` 装饰器在您需要跨线程共享模型实例但希望确保一次只有一个线程可以访问它时特别有用。与为每个线程创建新模型实例相比，这种方法可以节省内存，但可能会降低并发性，因为线程需要等待锁释放。

## 结论

在 Python 的 `threading` 中使用 YOLO 模型时，始终在将使用它们的线程内实例化模型以确保线程安全。这种做法避免了竞态条件，并确保您的推理任务可靠运行。

对于更高级的场景以及进一步优化多线程推理性能，请考虑使用 [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) 进行基于进程的并行处理，或利用具有专用工作进程的任务队列。

## 常见问题

### 如何在多线程 Python 环境中使用 YOLO 模型时避免竞态条件？

要在多线程 Python 环境中使用 Ultralytics YOLO 模型时防止竞态条件，请在每个线程内实例化单独的 YOLO 模型。这确保每个线程都有自己的隔离模型实例，避免模型状态的并发修改。

示例：

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """以线程安全的方式对图像进行预测。"""
    local_model = YOLO("yolo11n.pt")
    results = local_model.predict(image_path)
    # 处理结果


Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

有关确保线程安全的更多信息，请访问[线程安全推理](#线程安全推理)。

### 在 Python 中安全运行多线程 YOLO 模型推理的最佳实践是什么？

要在 Python 中安全运行多线程 YOLO 模型推理，请遵循以下最佳实践：

1. 在每个线程内实例化 YOLO 模型，而不是跨线程共享单个模型实例。
2. 使用 Python 的 `multiprocessing` 模块进行并行处理，以避免与全局解释器锁 (GIL) 相关的问题。
3. 通过使用 YOLO 底层 C 库执行的操作来释放 GIL。
4. 当内存是问题时，考虑对共享模型实例使用 `ThreadingLocked` 装饰器。

线程安全模型实例化示例：

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """使用新的 YOLO 模型实例以线程安全的方式运行推理。"""
    model = YOLO("yolo11n.pt")
    results = model.predict(image_path)
    # 处理结果


# 启动多个线程
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

有关更多上下文，请参考[线程安全推理](#线程安全推理)部分。

### 为什么每个线程应该有自己的 YOLO 模型实例？

每个线程应该有自己的 YOLO 模型实例以防止竞态条件。当单个模型实例在多个线程之间共享时，并发访问可能导致不可预测的行为和模型内部状态的修改。通过使用单独的实例，您可以确保线程隔离，使您的多线程任务可靠和安全。

有关详细指导，请查看[非线程安全示例：单个模型实例](#非线程安全示例单个模型实例)和[线程安全示例](#线程安全示例)部分。

### Python 的全局解释器锁 (GIL) 如何影响 YOLO 模型推理？

Python 的全局解释器锁 (GIL) 一次只允许一个线程执行 Python 字节码，这可能会限制 CPU 密集型多线程任务的性能。然而，对于 I/O 密集型操作或使用释放 GIL 的库（如 YOLO 的底层 C 库）的进程，您仍然可以实现并发。为了增强性能，请考虑使用 Python 的 `multiprocessing` 模块进行基于进程的并行处理。

有关 Python 线程的更多信息，请参阅[理解 Python 线程](#理解-python-线程)部分。

### 对于 YOLO 模型推理，使用基于进程的并行处理是否比线程更安全？

是的，使用 Python 的 `multiprocessing` 模块对于并行运行 YOLO 模型推理通常更安全且更高效。基于进程的并行处理创建单独的内存空间，避免全局解释器锁 (GIL) 并降低并发问题的风险。每个进程将独立运行，拥有自己的 YOLO 模型实例。

有关使用 YOLO 模型进行基于进程的并行处理的更多详细信息，请参考[线程安全推理](#线程安全推理)页面。
