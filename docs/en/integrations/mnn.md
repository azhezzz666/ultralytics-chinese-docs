---
comments: true
description: 通过导出到 MNN 格式优化 YOLO11 模型，用于移动和嵌入式设备。学习如何转换、部署和使用 MNN 运行推理。
keywords: Ultralytics, YOLO11, MNN, 模型导出, 机器学习, 部署, 移动, 嵌入式系统, 深度学习, AI 模型, 推理, 量化
---

# YOLO11 模型的 MNN 导出和部署

## MNN

<p align="center">
  <img width="100%" src="https://mnn-docs.readthedocs.io/en/latest/_images/architecture.png" alt="MNN 架构">
</p>

[MNN](https://github.com/alibaba/MNN) 是一个高效轻量的深度学习框架。它支持深度学习模型的推理和训练，在设备端推理和训练方面具有业界领先的性能。目前，MNN 已集成到阿里巴巴集团的 30 多个应用中，如淘宝、天猫、优酷、钉钉、闲鱼等，覆盖直播、短视频拍摄、搜索推荐、以图搜商品、互动营销、权益分发、安全风控等 70 多个使用场景。此外，MNN 还用于嵌入式设备，如物联网设备。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/i34PacLIlq8"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO11 导出为 MNN 格式 | 在移动设备上加速推理📱
</p>

## 导出到 MNN：转换你的 YOLO11 模型

你可以通过将 [Ultralytics YOLO](../models/yolo11.md) 模型转换为 MNN 格式来扩展模型兼容性和部署灵活性。此转换优化你的模型用于移动和嵌入式环境，确保在资源受限设备上的高效性能。

### 安装

要安装所需的包，运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 和 MNN 所需的包
        pip install ultralytics
        pip install MNN
        ```

### 用法

所有 [Ultralytics YOLO11 模型](../models/index.md)都设计为开箱即用支持导出，使其易于集成到你首选的部署工作流程中。你可以[查看支持的导出格式和配置选项的完整列表](../modes/export.md)，为你的应用选择最佳设置。

!!! example "用法"

    === "Python"

          ```python
          from ultralytics import YOLO

          # 加载 YOLO11 模型
          model = YOLO("yolo11n.pt")

          # 将模型导出为 MNN 格式
          model.export(format="mnn")  # 创建 'yolo11n.mnn'

          # 加载导出的 MNN 模型
          mnn_model = YOLO("yolo11n.mnn")

          # 运行推理
          results = mnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # 将 YOLO11n PyTorch 模型导出为 MNN 格式
          yolo export model=yolo11n.pt format=mnn # 创建 'yolo11n.mnn'

          # 使用导出的模型运行推理
          yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg'
          ```

### 导出参数

| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str` | `'mnn'` | 导出模型的目标格式，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入所需的图像大小。可以是整数（用于正方形图像）或元组 `(height, width)`（用于特定尺寸）。 |
| `half` | `bool` | `False` | 启用 FP16（半精度）量化，减小模型大小并可能在支持的硬件上加速推理。 |
| `int8` | `bool` | `False` | 激活 INT8 量化，进一步压缩模型并加速推理，同时[准确率](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。 |
| `batch` | `int` | `1` | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。 |
| `device` | `str` | `None` | 指定导出设备：GPU (`device=0`)、CPU (`device=cpu`)、Apple silicon 的 MPS (`device=mps`)。 |

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

### 仅 MNN 推理

实现了一个仅依赖 MNN 进行 YOLO11 推理和预处理的函数，提供 Python 和 C++ 版本，便于在任何场景中轻松部署。

!!! example "MNN"

    === "Python"

        ```python
        import argparse

        import MNN
        import MNN.cv as cv2
        import MNN.numpy as np


        def inference(model, img, precision, backend, thread):
            config = {}
            config["precision"] = precision
            config["backend"] = backend
            config["numThread"] = thread
            rt = MNN.nn.create_runtime_manager((config,))
            # net = MNN.nn.load_module_from_file(model, ['images'], ['output0'], runtime_manager=rt)
            net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)
            original_image = cv2.imread(img)
            ih, iw, _ = original_image.shape
            length = max((ih, iw))
            scale = length / 640
            image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], "constant")
            image = cv2.resize(
                image, (640, 640), 0.0, 0.0, cv2.INTER_LINEAR, -1, [0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            )
            image = image[..., ::-1]  # BGR 转 RGB
            input_var = np.expand_dims(image, 0)
            input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
            output_var = net.forward(input_var)
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
            output_var = output_var.squeeze()
            # output_var 形状: [84, 8400]; 84 表示: [cx, cy, w, h, prob * 80]
            cx = output_var[0]
            cy = output_var[1]
            w = output_var[2]
            h = output_var[3]
            probs = output_var[4:]
            # [cx, cy, w, h] -> [y0, x0, y1, x1]
            x0 = cx - w * 0.5
            y0 = cy - h * 0.5
            x1 = cx + w * 0.5
            y1 = cy + h * 0.5
            boxes = np.stack([x0, y0, x1, y1], axis=1)
            # 确保比例在有效范围 [0.0, 1.0] 内
            boxes = np.clip(boxes, 0, 1)
            # 获取最大概率和索引
            scores = np.max(probs, 0)
            class_ids = np.argmax(probs, 0)
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
            print(result_ids.shape)
            # nms 结果框、分数、id
            result_boxes = boxes[result_ids]
            result_scores = scores[result_ids]
            result_class_ids = class_ids[result_ids]
            for i in range(len(result_boxes)):
                x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
                y0 = int(y0 * scale)
                y1 = int(y1 * scale)
                x0 = int(x0 * scale)
                x1 = int(x1 * scale)
                # 裁剪到原始图像大小以处理应用填充的情况
                x1 = min(iw, x1)
                y1 = min(ih, y1)
                print(result_class_ids[i])
                cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imwrite("res.jpg", original_image)


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", type=str, required=True, help="yolo11 模型路径")
            parser.add_argument("--img", type=str, required=True, help="输入图像路径")
            parser.add_argument("--precision", type=str, default="normal", help="推理精度: normal, low, high, lowBF")
            parser.add_argument(
                "--backend",
                type=str,
                default="CPU",
                help="推理后端: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI",
            )
            parser.add_argument("--thread", type=int, default=4, help="推理使用的线程数: int")
            args = parser.parse_args()
            inference(args.model, args.img, args.precision, args.backend, args.thread)
        ```

    === "CPP"

        ```cpp
        #include <stdio.h>
        #include <MNN/ImageProcess.hpp>
        #include <MNN/expr/Module.hpp>
        #include <MNN/expr/Executor.hpp>
        #include <MNN/expr/ExprCreator.hpp>
        #include <MNN/expr/Executor.hpp>

        #include <cv/cv.hpp>

        using namespace MNN;
        using namespace MNN::Express;
        using namespace MNN::CV;

        int main(int argc, const char* argv[]) {
            if (argc < 3) {
                MNN_PRINT("用法: ./yolo11_demo.out model.mnn input.jpg [forwardType] [precision] [thread]\n");
                return 0;
            }
            int thread = 4;
            int precision = 0;
            int forwardType = MNN_FORWARD_CPU;
            if (argc >= 4) {
                forwardType = atoi(argv[3]);
            }
            if (argc >= 5) {
                precision = atoi(argv[4]);
            }
            if (argc >= 6) {
                thread = atoi(argv[5]);
            }
            MNN::ScheduleConfig sConfig;
            sConfig.type = static_cast<MNNForwardType>(forwardType);
            sConfig.numThread = thread;
            BackendConfig bConfig;
            bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
            sConfig.backendConfig = &bConfig;
            std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
            if(rtmgr == nullptr) {
                MNN_ERROR("空的 RuntimeManger\n");
                return 0;
            }
            rtmgr->setCache(".cachefile");

            std::shared_ptr<Module> net(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
            auto original_image = imread(argv[2]);
            auto dims = original_image->getInfo()->dim;
            int ih = dims[0];
            int iw = dims[1];
            int len = ih > iw ? ih : iw;
            float scale = len / 640.0;
            std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
            auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
            auto image = _Pad(original_image, pads, CONSTANT);
            image = resize(image, Size(640, 640), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
            image = cvtColor(image, COLOR_BGR2RGB);
            auto input = _Unsqueeze(image, {0});
            input = _Convert(input, NC4HW4);
            auto outputs = net->onForward({input});
            auto output = _Convert(outputs[0], NCHW);
            output = _Squeeze(output);
            // output 形状: [84, 8400]; 84 表示: [cx, cy, w, h, prob * 80]
            auto cx = _Gather(output, _Scalar<int>(0));
            auto cy = _Gather(output, _Scalar<int>(1));
            auto w = _Gather(output, _Scalar<int>(2));
            auto h = _Gather(output, _Scalar<int>(3));
            std::vector<int> startvals { 4, 0 };
            auto start = _Const(static_cast<void*>(startvals.data()), {2}, NCHW, halide_type_of<int>());
            std::vector<int> sizevals { -1, -1 };
            auto size = _Const(static_cast<void*>(sizevals.data()), {2}, NCHW, halide_type_of<int>());
            auto probs = _Slice(output, start, size);
            // [cx, cy, w, h] -> [y0, x0, y1, x1]
            auto x0 = cx - w * _Const(0.5);
            auto y0 = cy - h * _Const(0.5);
            auto x1 = cx + w * _Const(0.5);
            auto y1 = cy + h * _Const(0.5);
            auto boxes = _Stack({x0, y0, x1, y1}, 1);
            // 确保比例在有效范围 [0.0, 1.0] 内
            boxes = _Maximum(boxes, _Scalar<float>(0.0f));
            boxes = _Minimum(boxes, _Scalar<float>(1.0f));
            auto scores = _ReduceMax(probs, {0});
            auto ids = _ArgMax(probs, 0);
            auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);
            auto result_ptr = result_ids->readMap<int>();
            auto box_ptr = boxes->readMap<float>();
            auto ids_ptr = ids->readMap<int>();
            auto score_ptr = scores->readMap<float>();
            for (int i = 0; i < 100; i++) {
                auto idx = result_ptr[i];
                if (idx < 0) break;
                auto x0 = box_ptr[idx * 4 + 0] * scale;
                auto y0 = box_ptr[idx * 4 + 1] * scale;
                auto x1 = box_ptr[idx * 4 + 2] * scale;
                auto y1 = box_ptr[idx * 4 + 3] * scale;
                // 裁剪到原始图像大小以处理应用填充的情况
                x1 = std::min(static_cast<float>(iw), x1);
                y1 = std::min(static_cast<float>(ih), y1);
                auto class_idx = ids_ptr[idx];
                auto score = score_ptr[idx];
                rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
            }
            if (imwrite("res.jpg", original_image)) {
                MNN_PRINT("结果图像写入 `res.jpg`。\n");
            }
            rtmgr->updateCache();
            return 0;
        }
        ```

## 总结

在本指南中，我们介绍了如何将 Ultralytics YOLO11 模型导出为 MNN 以及使用 MNN 进行推理。MNN 格式为[边缘 AI](https://www.ultralytics.com/glossary/edge-ai) 应用提供了出色的性能，使其非常适合在资源受限设备上部署计算机视觉模型。

有关更多用法，请参阅 [MNN 文档](https://mnn-docs.readthedocs.io/en/latest)。

## 常见问题

### 如何将 Ultralytics YOLO11 模型导出为 MNN 格式？

要将 Ultralytics YOLO11 模型导出为 MNN 格式，请按照以下步骤操作：

!!! example "导出"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 导出为 MNN 格式
        model.export(format="mnn")  # 创建带 fp32 权重的 'yolo11n.mnn'
        model.export(format="mnn", half=True)  # 创建带 fp16 权重的 'yolo11n.mnn'
        model.export(format="mnn", int8=True)  # 创建带 int8 权重的 'yolo11n.mnn'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=mnn           # 创建带 fp32 权重的 'yolo11n.mnn'
        yolo export model=yolo11n.pt format=mnn half=True # 创建带 fp16 权重的 'yolo11n.mnn'
        yolo export model=yolo11n.pt format=mnn int8=True # 创建带 int8 权重的 'yolo11n.mnn'
        ```

有关详细的导出选项，请查看文档中的[导出](../modes/export.md)页面。

### 如何使用导出的 YOLO11 MNN 模型进行预测？

要使用导出的 YOLO11 MNN 模型进行预测，使用 YOLO 类的 `predict` 函数。

!!! example "预测"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 MNN 模型
        model = YOLO("yolo11n.mnn")

        # 导出为 MNN 格式
        results = model("https://ultralytics.com/images/bus.jpg")  # 使用 `fp32` 预测
        results = model("https://ultralytics.com/images/bus.jpg", half=True)  # 如果设备支持，使用 `fp16` 预测

        for result in results:
            result.show()  # 显示到屏幕
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

    === "CLI"

        ```bash
        yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg'             # 使用 `fp32` 预测
        yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg' --half=True # 如果设备支持，使用 `fp16` 预测
        ```

### MNN 支持哪些平台？

MNN 功能多样，支持各种平台：

- **移动端**：Android、iOS、Harmony。
- **嵌入式系统和物联网设备**：如 [Raspberry Pi](../guides/raspberry-pi.md) 和 NVIDIA Jetson 等设备。
- **桌面和服务器**：Linux、Windows 和 macOS。

### 如何在移动设备上部署 Ultralytics YOLO11 MNN 模型？

要在移动设备上部署 YOLO11 模型：

1. **Android 构建**：按照 [MNN Android](https://github.com/alibaba/MNN/tree/master/project/android) 指南。
2. **iOS 构建**：按照 [MNN iOS](https://github.com/alibaba/MNN/tree/master/project/ios) 指南。
3. **Harmony 构建**：按照 [MNN Harmony](https://github.com/alibaba/MNN/tree/master/project/harmony) 指南。
