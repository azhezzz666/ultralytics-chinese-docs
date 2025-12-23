---
comments: true
description: 探索 YOLOv7，这款突破性的实时目标检测器具有顶级速度和精度。了解关键特性、使用方法和性能指标。
keywords: YOLOv7, 实时目标检测, Ultralytics, AI, 计算机视觉, 模型训练, 目标检测器
---

# YOLOv7：可训练的免费技巧包

YOLOv7 是一款最先进的实时目标检测器，在 5 FPS 到 160 FPS 范围内的速度和[精度](https://www.ultralytics.com/glossary/accuracy)方面超越了所有已知的目标检测器。它在 GPU V100 上以 30 FPS 或更高帧率运行时，在所有已知实时目标检测器中具有最高精度（56.8% AP）。此外，YOLOv7 在速度和精度方面优于其他目标检测器，如 YOLOR、YOLOX、Scaled-YOLOv4、YOLOv5 等。该模型从头开始在 MS COCO 数据集上训练，未使用任何其他数据集或预训练权重。YOLOv7 的源代码可在 GitHub 上获取。

![YOLOv7 与 SOTA 目标检测器的比较](https://github.com/ultralytics/docs/releases/download/0/yolov7-comparison-sota-object-detectors.avif)

## SOTA 目标检测器比较

从 YOLO 比较表的结果中，我们知道所提出的方法在综合速度-精度权衡方面表现最佳。如果我们将 YOLOv7-tiny-SiLU 与 YOLOv5-N (r6.1) 进行比较，我们的方法快 127 fps，AP 精度提高 10.7%。此外，YOLOv7 在 161 fps 帧率下达到 51.4% AP，而具有相同 AP 的 PPYOLOE-L 仅有 78 fps 帧率。在参数使用方面，YOLOv7 比 PPYOLOE-L 少 41%。

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv7"]'></canvas>

如果我们将 114 fps 推理速度的 YOLOv7-X 与 99 fps 推理速度的 YOLOv5-L (r6.1) 进行比较，YOLOv7-X 可以将 AP 提高 3.9%。如果将 YOLOv7-X 与类似规模的 YOLOv5-X (r6.1) 进行比较，YOLOv7-X 的推理速度快 31 fps。此外，在参数量和计算量方面，YOLOv7-X 与 YOLOv5-X (r6.1) 相比减少了 22% 的参数和 8% 的计算量，但 AP 提高了 2.2%（[来源](https://arxiv.org/pdf/2207.02696)）。

!!! tip "性能"

    === "检测 (COCO)"

        | 模型                  | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(G)</sup> | 尺寸<br><sup>(像素)</sup> | FPS     | AP<sup>test / val<br>50-95</sup> | AP<sup>test<br>50</sup> | AP<sup>test<br>75</sup> | AP<sup>test<br>S</sup> | AP<sup>test<br>M</sup> | AP<sup>test<br>L</sup> |
        | --------------------- | ------------------ | ----------------- | --------------------- | ------- | -------------------------- | ----------------- | ----------------- | ---------------- | ---------------- | ---------------- |
        | [YOLOX-S][1]          | **9.0**           | **26.8**         | 640                   | **102** | 40.5% / 40.5%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-M][1]          | 25.3              | 73.8             | 640                   | 81      | 47.2% / 46.9%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-L][1]          | 54.2              | 155.6            | 640                   | 69      | 50.1% / 49.7%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-X][1]          | 99.1              | 281.9            | 640                   | 58      | **51.5% / 51.1%**          | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [PPYOLOE-S][2]        | **7.9**           | **17.4**         | 640                   | **208** | 43.1% / 42.7%              | 60.5%             | 46.6%             | 23.2%            | 46.4%            | 56.9%            |
        | [PPYOLOE-M][2]        | 23.4              | 49.9             | 640                   | 123     | 48.9% / 48.6%              | 66.5%             | 53.0%             | 28.6%            | 52.9%            | 63.8%            |
        | [PPYOLOE-L][2]        | 52.2              | 110.1            | 640                   | 78      | 51.4% / 50.9%              | 68.9%             | 55.6%             | 31.4%            | 55.3%            | 66.1%            |
        | [PPYOLOE-X][2]        | 98.4              | 206.6            | 640                   | 45      | **52.2% / 51.9%**          | **69.9%**         | **56.5%**         | **33.3%**        | **56.3%**        | **66.4%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv5-N (r6.1)][3]  | **1.9**           | **4.5**          | 640                   | **159** | - / 28.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-S (r6.1)][3]  | 7.2               | 16.5             | 640                   | 156     | - / 37.4%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-M (r6.1)][3]  | 21.2              | 49.0             | 640                   | 122     | - / 45.4%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-L (r6.1)][3]  | 46.5              | 109.1            | 640                   | 99      | - / 49.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-X (r6.1)][3]  | 86.7              | 205.7            | 640                   | 83      | - / **50.7%**              | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOR-CSP][4]        | 52.9              | 120.4            | 640                   | 106     | 51.1% / 50.8%              | 69.6%             | 55.7%             | 31.7%            | 55.3%            | 64.7%            |
        | [YOLOR-CSP-X][4]      | 96.9              | 226.8            | 640                   | 87      | 53.0% / 52.7%              | 71.4%             | 57.9%             | 33.7%            | 57.1%            | 66.8%            |
        | [YOLOv7-tiny-SiLU][5] | **6.2**           | **13.8**         | 640                   | **286** | 38.7% / 38.7%              | 56.7%             | 41.7%             | 18.8%            | 42.4%            | 51.9%            |
        | [YOLOv7][5]           | 36.9              | 104.7            | 640                   | 161     | 51.4% / 51.2%              | 69.7%             | 55.9%             | 31.8%            | 55.5%            | 65.0%            |
        | [YOLOv7-X][5]         | 71.3              | 189.9            | 640                   | 114     | **53.1% / 52.9%**          | **71.2%**         | **57.8%**         | **33.8%**        | **57.1%**        | **67.4%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv5-N6 (r6.1)][3] | **3.2**           | **18.4**         | 1280                  | **123** | - / 36.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-S6 (r6.1)][3] | 12.6              | 67.2             | 1280                  | 122     | - / 44.8%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-M6 (r6.1)][3] | 35.7              | 200.0            | 1280                  | 90      | - / 51.3%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-L6 (r6.1)][3] | 76.8              | 445.6            | 1280                  | 63      | - / 53.7%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-X6 (r6.1)][3] | 140.7             | 839.2            | 1280                  | 38      | - / **55.0%**              | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOR-P6][4]         | **37.2**          | **325.6**        | 1280                  | **76**  | 53.9% / 53.5%              | 71.4%             | 58.9%             | 36.1%            | 57.7%            | 65.6%            |
        | [YOLOR-W6][4]         | 79.8              | 453.2            | 1280                  | 66      | 55.2% / 54.8%              | 72.7%             | 60.5%             | 37.7%            | 59.1%            | 67.1%            |
        | [YOLOR-E6][4]         | 115.8             | 683.2            | 1280                  | 45      | 55.8% / 55.7%              | 73.4%             | 61.1%             | 38.4%            | 59.7%            | 67.7%            |
        | [YOLOR-D6][4]         | 151.7             | 935.6            | 1280                  | 34      | **56.5% / 56.1%**          | **74.1%**         | **61.9%**         | **38.9%**        | **60.4%**        | **68.7%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv7-W6][5]        | **70.4**          | **360.0**        | 1280                  | **84**  | 54.9% / 54.6%              | 72.6%             | 60.1%             | 37.3%            | 58.7%            | 67.1%            |
        | [YOLOv7-E6][5]        | 97.2              | 515.2            | 1280                  | 56      | 56.0% / 55.9%              | 73.5%             | 61.2%             | 38.0%            | 59.9%            | 68.4%            |
        | [YOLOv7-D6][5]        | 154.7             | 806.8            | 1280                  | 44      | 56.6% / 56.3%              | 74.0%             | 61.8%             | 38.8%            | 60.1%            | 69.5%            |
        | [YOLOv7-E6E][5]       | 151.7             | 843.2            | 1280                  | 36      | **56.8% / 56.8%**          | **74.4%**         | **62.1%**         | **39.3%**        | **60.5%**        | **69.0%**        |

        [1]: https://github.com/Megvii-BaseDetection/YOLOX
        [2]: https://github.com/PaddlePaddle/PaddleDetection
        [3]: https://github.com/ultralytics/yolov5
        [4]: https://github.com/WongKinYiu/yolor
        [5]: https://github.com/WongKinYiu/yolov7

## 概述

实时目标检测是许多[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)系统的重要组成部分，包括多[目标跟踪](https://www.ultralytics.com/glossary/object-tracking)、自动驾驶、[机器人技术](https://www.ultralytics.com/glossary/robotics)和[医学图像分析](https://www.ultralytics.com/glossary/medical-image-analysis)。近年来，实时目标检测的发展重点是设计高效的架构，并提高各种 CPU、GPU 和神经处理单元 (NPU) 的推理速度。YOLOv7 支持移动 GPU 和 GPU 设备，从边缘到云端。

与传统的专注于架构优化的实时目标检测器不同，YOLOv7 引入了对训练过程优化的关注。这包括旨在提高目标检测精度而不增加推理成本的模块和优化方法，这一概念被称为"可训练的免费技巧包"。

## 关键特性

YOLOv7 引入了几个关键特性：

1. **模型重参数化**：YOLOv7 提出了一种计划性的重参数化模型，这是一种适用于不同网络层的策略，具有梯度传播路径的概念。

2. **动态标签分配**：具有多个输出层的模型训练提出了一个新问题："如何为不同分支的输出分配动态目标？"为了解决这个问题，YOLOv7 引入了一种新的标签分配方法，称为从粗到细的引导标签分配。

3. **扩展和复合缩放**：YOLOv7 为实时目标检测器提出了"扩展"和"复合缩放"方法，可以有效利用参数和计算量。

4. **效率**：YOLOv7 提出的方法可以有效减少约 40% 的参数和 50% 的计算量，同时具有更快的推理速度和更高的检测精度。

## 使用示例

截至撰写本文时，Ultralytics 仅支持 YOLOv7 的 ONNX 和 TensorRT 推理。

### ONNX 导出

要在 Ultralytics 中使用 YOLOv7 ONNX 模型：

0. （可选）安装 Ultralytics 并导出 ONNX 模型以自动安装所需依赖项：

    ```bash
    pip install ultralytics
    yolo export model=yolo11n.pt format=onnx
    ```

1. 使用 [YOLOv7 仓库](https://github.com/WongKinYiu/yolov7)中的导出器导出所需的 YOLOv7 模型：

    ```bash
    git clone https://github.com/WongKinYiu/yolov7
    cd yolov7
    python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
    ```

2. 使用以下脚本修改 ONNX 模型图以与 Ultralytics 兼容：

    ```python
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    # 加载 ONNX 模型
    model_path = "yolov7/yolov7-tiny.onnx"  # 替换为你的模型路径
    model = onnx.load(model_path)
    graph = model.graph

    # 将输入形状固定为批次大小 1
    input_shape = graph.input[0].type.tensor_type.shape
    input_shape.dim[0].dim_value = 1

    # 定义原始模型的输出
    original_output_name = graph.output[0].name

    # 创建切片节点
    sliced_output_name = f"{original_output_name}_sliced"

    # 定义切片的初始化器（移除第一个值）
    start = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_start")
    end = numpy_helper.from_array(np.array([7], dtype=np.int64), name="slice_end")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_steps")

    graph.initializer.extend([start, end, axes, steps])

    slice_node = helper.make_node(
        "Slice",
        inputs=[original_output_name, "slice_start", "slice_end", "slice_axes", "slice_steps"],
        outputs=[sliced_output_name],
        name="SliceNode",
    )
    graph.node.append(slice_node)

    # 定义段切片
    seg1_start = numpy_helper.from_array(np.array([0], dtype=np.int64), name="seg1_start")
    seg1_end = numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg1_end")
    seg2_start = numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg2_start")
    seg2_end = numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg2_end")
    seg3_start = numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg3_start")
    seg3_end = numpy_helper.from_array(np.array([6], dtype=np.int64), name="seg3_end")

    graph.initializer.extend([seg1_start, seg1_end, seg2_start, seg2_end, seg3_start, seg3_end])

    # 为段创建中间张量
    segment_1_name = f"{sliced_output_name}_segment1"
    segment_2_name = f"{sliced_output_name}_segment2"
    segment_3_name = f"{sliced_output_name}_segment3"

    # 添加段切片节点
    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg1_start", "seg1_end", "slice_axes", "slice_steps"],
                outputs=[segment_1_name],
                name="SliceSegment1",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg2_start", "seg2_end", "slice_axes", "slice_steps"],
                outputs=[segment_2_name],
                name="SliceSegment2",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg3_start", "seg3_end", "slice_axes", "slice_steps"],
                outputs=[segment_3_name],
                name="SliceSegment3",
            ),
        ]
    )

    # 连接段
    concat_output_name = f"{sliced_output_name}_concat"
    concat_node = helper.make_node(
        "Concat",
        inputs=[segment_1_name, segment_3_name, segment_2_name],
        outputs=[concat_output_name],
        axis=1,
        name="ConcatSwapped",
    )
    graph.node.append(concat_node)

    # 重塑为 [1, -1, 6]
    reshape_shape = numpy_helper.from_array(np.array([1, -1, 6], dtype=np.int64), name="reshape_shape")
    graph.initializer.append(reshape_shape)

    final_output_name = f"{concat_output_name}_batched"
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[concat_output_name, "reshape_shape"],
        outputs=[final_output_name],
        name="AddBatchDimension",
    )
    graph.node.append(reshape_node)

    # 获取重塑张量的形状
    shape_node_name = f"{final_output_name}_shape"
    shape_node = helper.make_node(
        "Shape",
        inputs=[final_output_name],
        outputs=[shape_node_name],
        name="GetShapeDim",
    )
    graph.node.append(shape_node)

    # 提取第二个维度
    dim_1_index = numpy_helper.from_array(np.array([1], dtype=np.int64), name="dim_1_index")
    graph.initializer.append(dim_1_index)

    second_dim_name = f"{final_output_name}_dim1"
    gather_node = helper.make_node(
        "Gather",
        inputs=[shape_node_name, "dim_1_index"],
        outputs=[second_dim_name],
        name="GatherSecondDim",
    )
    graph.node.append(gather_node)

    # 从 100 减去以确定需要填充多少值
    target_size = numpy_helper.from_array(np.array([100], dtype=np.int64), name="target_size")
    graph.initializer.append(target_size)

    pad_size_name = f"{second_dim_name}_padsize"
    sub_node = helper.make_node(
        "Sub",
        inputs=["target_size", second_dim_name],
        outputs=[pad_size_name],
        name="CalculatePadSize",
    )
    graph.node.append(sub_node)

    # 构建 [2, 3] 填充数组：
    # 第 1 行 -> [0, 0, 0]（任何维度的开始都不填充）
    # 第 2 行 -> [0, pad_size, 0]（仅在第二个维度的末尾填充）
    pad_starts = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), name="pad_starts")
    graph.initializer.append(pad_starts)

    zero_scalar = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_scalar")
    graph.initializer.append(zero_scalar)

    pad_ends_name = "pad_ends"
    concat_pad_ends_node = helper.make_node(
        "Concat",
        inputs=["zero_scalar", pad_size_name, "zero_scalar"],
        outputs=[pad_ends_name],
        axis=0,
        name="ConcatPadEnds",
    )
    graph.node.append(concat_pad_ends_node)

    pad_values_name = "pad_values"
    concat_pad_node = helper.make_node(
        "Concat",
        inputs=["pad_starts", pad_ends_name],
        outputs=[pad_values_name],
        axis=0,
        name="ConcatPadStartsEnds",
    )
    graph.node.append(concat_pad_node)

    # 创建 Pad 操作符以用零填充
    pad_output_name = f"{final_output_name}_padded"
    pad_constant_value = numpy_helper.from_array(
        np.array([0.0], dtype=np.float32),
        name="pad_constant_value",
    )
    graph.initializer.append(pad_constant_value)

    pad_node = helper.make_node(
        "Pad",
        inputs=[final_output_name, pad_values_name, "pad_constant_value"],
        outputs=[pad_output_name],
        mode="constant",
        name="PadToFixedSize",
    )
    graph.node.append(pad_node)

    # 将图的最终输出更新为 [1, 100, 6]
    new_output_type = onnx.helper.make_tensor_type_proto(
        elem_type=graph.output[0].type.tensor_type.elem_type, shape=[1, 100, 6]
    )
    new_output = onnx.helper.make_value_info(name=pad_output_name, type_proto=new_output_type)

    # 用新输出替换旧输出
    graph.output.pop()
    graph.output.extend([new_output])

    # 保存修改后的模型
    onnx.save(model, "yolov7-ultralytics.onnx")
    ```

3. 然后你可以正常在 Ultralytics 中加载修改后的 ONNX 模型并运行推理：

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolov7-ultralytics.onnx", task="detect")

    results = model(ASSETS / "bus.jpg")
    ```

### TensorRT 导出

1. 按照 [ONNX 导出](#onnx-导出)部分的步骤 1-2 操作。

2. 安装 `TensorRT` Python 包：

    ```bash
    pip install tensorrt
    ```

3. 运行以下脚本将修改后的 ONNX 模型转换为 TensorRT 引擎：

    ```python
    from ultralytics.utils.export import export_engine

    export_engine("yolov7-ultralytics.onnx", half=True)
    ```

4. 在 Ultralytics 中加载并运行模型：

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolov7-ultralytics.engine", task="detect")

    results = model(ASSETS / "bus.jpg")
    ```

## 引用和致谢

我们要感谢 YOLOv7 作者在实时目标检测领域做出的重大贡献：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

原始 YOLOv7 论文可在 [arXiv](https://arxiv.org/pdf/2207.02696) 上找到。作者已公开其工作，代码库可在 [GitHub](https://github.com/WongKinYiu/yolov7) 上访问。我们感谢他们在推进该领域发展并使其工作对更广泛社区可用方面所做的努力。

## 常见问题

### 什么是 YOLOv7，为什么它被认为是实时[目标检测](https://www.ultralytics.com/glossary/object-detection)的突破？

YOLOv7 是一款尖端的实时目标检测模型，实现了无与伦比的速度和精度。它在参数使用和推理速度方面超越了其他模型，如 YOLOX、YOLOv5 和 PPYOLOE。YOLOv7 的显著特点包括其模型重参数化和动态标签分配，这些优化了其性能而不增加推理成本。有关其架构和与其他最先进目标检测器的比较指标的更多技术细节，请参阅 [YOLOv7 论文](https://arxiv.org/pdf/2207.02696)。

### YOLOv7 相比之前的 YOLO 模型（如 YOLOv4 和 YOLOv5）有哪些改进？

YOLOv7 引入了多项创新，包括模型重参数化和动态标签分配，这些增强了训练过程并提高了推理精度。与 YOLOv5 相比，YOLOv7 显著提高了速度和精度。例如，YOLOv7-X 与 YOLOv5-X 相比，精度提高了 2.2%，参数减少了 22%。详细比较可在性能表 [YOLOv7 与 SOTA 目标检测器的比较](#sota-目标检测器比较)中找到。

### 我可以在 Ultralytics 工具和平台上使用 YOLOv7 吗？

目前，Ultralytics 仅支持 YOLOv7 的 ONNX 和 TensorRT 推理。要在 Ultralytics 中运行 ONNX 和 TensorRT 导出版本的 YOLOv7，请查看[使用示例](#使用示例)部分。

### 如何使用我的数据集训练自定义 YOLOv7 模型？

要安装和训练自定义 YOLOv7 模型，请按照以下步骤操作：

1. 克隆 YOLOv7 仓库：
    ```bash
    git clone https://github.com/WongKinYiu/yolov7
    ```
2. 导航到克隆的目录并安装依赖项：
    ```bash
    cd yolov7
    pip install -r requirements.txt
    ```
3. 根据仓库中提供的[使用说明](https://github.com/WongKinYiu/yolov7)准备数据集并配置模型参数。
   有关更多指导，请访问 YOLOv7 GitHub 仓库获取最新信息和更新。

4. 训练完成后，你可以按照[使用示例](#使用示例)中所示将模型导出为 ONNX 或 TensorRT 以在 Ultralytics 中使用。

### YOLOv7 引入的关键特性和优化有哪些？

YOLOv7 提供了几个革命性的实时目标检测关键特性：

- **模型重参数化**：通过优化梯度传播路径来增强模型性能。
- **动态标签分配**：使用从粗到细的引导方法为不同分支的输出分配动态目标，提高精度。
- **扩展和复合缩放**：有效利用参数和计算量来缩放模型以适应各种实时应用。
- **效率**：与其他最先进模型相比，减少了 40% 的参数量和 50% 的计算量，同时实现了更快的推理速度。

有关这些特性的更多详细信息，请参阅 [YOLOv7 概述](#概述)部分。
