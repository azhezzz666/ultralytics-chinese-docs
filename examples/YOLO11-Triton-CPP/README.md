# YOLO11 Triton 推理服务器 C++ 客户端

[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-orange)](https://github.com/ultralytics/ultralytics)
[![Triton](https://img.shields.io/badge/NVIDIA-Triton-green)](https://github.com/triton-inference-server/server)

本示例演示如何使用部署在 NVIDIA Triton 推理服务器上的 Ultralytics YOLO11 模型执行目标检测。该实现展示了高效的图像预处理、FP16（半精度）数据转换、通过 gRPC 与 Triton 服务器的无缝通信，以及使用边界框和置信度分数可视化检测结果。

## ⚡ 功能特性

- **高性能推理**：使用 FP16（半精度）数据格式优化内存使用并加速推理。
- **非极大值抑制（NMS）**：移除重复检测以确保精确的目标检测结果。
- **无缝 Triton 集成**：通过 gRPC 与 NVIDIA Triton 推理服务器通信，实现高效可扩展的模型服务。
- **检测可视化**：使用边界框、类别标签和置信度分数标注图像，便于直观解读结果。

## 🛠️ 依赖项

在继续之前，请确保已安装以下依赖项：

| 依赖项                  | 版本    | 描述                                  |
| ----------------------- | ------- | ------------------------------------- |
| Triton 推理服务器       | 22.06   | 运行已部署的 FP16 YOLO11 模型         |
| Triton 客户端库         | 2.23    | 与 Triton 服务器通信所需              |
| C++ 编译器              | C++ 17+ | 用于编译 C++ 客户端应用程序           |
| OpenCV 库               | 3.4.15  | 用于图像处理和可视化                  |
| CMake                   | 3.5+    | 用于构建项目                          |

有关 Triton 的更多信息，请参阅 [NVIDIA Triton 推理服务器文档](https://github.com/triton-inference-server/server)并探索 [Ultralytics 模型部署选项](https://docs.ultralytics.com/guides/model-deployment-options/)。

## 🏗️ 构建项目

1. **安装 Triton 客户端库：**

   ```bash
   wget https://github.com/triton-inference-server/server/releases/download/v2.23.0/v2.23.0_ubuntu2004.clients.tar.gz
   mkdir tritonclient
   tar -xvf v2.23.0_ubuntu2004.clients.tar.gz -C tritonclient
   rm -f v2.23.0_ubuntu2004.clients.tar.gz
   ```

2. **克隆 Ultralytics 仓库：**

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/examples/YOLO11-Triton-CPP
   ```

3. **使用 CMake 配置和构建项目：**

   ```bash
   mkdir build
   cd build
   cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient
   make
   ```

有关将 Ultralytics YOLO 模型与各种平台集成的更多指导，请查看 [Ultralytics 集成文档](https://docs.ultralytics.com/integrations/)。

## 🚀 使用方法

1. **在 Triton 推理服务器上部署你的 FP16（半精度）YOLO11 模型。**
   了解更多关于使用 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/) 部署模型的信息。

2. **运行 YOLO11-Triton-CPP 应用程序：**

   ```bash
   ./YOLO11TritonCPP
   ```

默认情况下，应用程序将：

- 连接到 `localhost:8001` 的 Triton 服务器
- 使用名为 `yolo11` 版本为 `1` 的模型
- 处理图像文件 `test.jpg`
- 将检测结果保存到 `output.jpg`

有关目标检测工作流程的更多信息，请参阅 [Ultralytics 目标检测任务](https://docs.ultralytics.com/tasks/detect/)。

## ⚙️ 配置

你可以在 [main.cpp](main.cpp) 中修改以下参数：

```cpp
std::string triton_address = "localhost:8001";
std::string model_name = "yolo11";
std::string model_version = "1";
std::string image_path = "test.jpg";
std::string output_path = "output.jpg";
std::vector<std::string> object_class_list = {"class1", "class2"};
```

要了解更多关于配置和自定义 YOLO 模型的信息，请访问 [Ultralytics 配置指南](https://docs.ultralytics.com/usage/cfg/)。

## 🌟 贡献者

欢迎贡献！如果你发现任何问题或有改进建议，请在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。

- Ahmet Selim Demirel
- Doğan Mehmet Başoğlu
- Enes Uzun
- Elif Cansu Ada
- Mevlüt Ardıç
- Serhat Karaca

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

---

更多资源，请探索 [Ultralytics 文档](https://docs.ultralytics.com/)、[Ultralytics 博客](https://www.ultralytics.com/blog)和 [Ultralytics HUB](https://docs.ultralytics.com/hub/)。

**我们鼓励你的贡献来帮助改进这个项目。**
