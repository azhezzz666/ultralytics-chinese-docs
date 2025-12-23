---
comments: true
description: 学习如何将 Ultralytics YOLO 与运行 ROS Noetic 的机器人集成，利用 RGB 图像、深度图像和点云进行高效的目标检测、分割和增强机器人感知。
keywords: Ultralytics, YOLO, 目标检测, 深度学习, 机器学习, 指南, ROS, 机器人操作系统, 机器人, ROS Noetic, Python, Ubuntu, 仿真, 可视化, 通信, 中间件, 硬件抽象, 工具, 实用程序, 生态系统, Noetic Ninjemys, 自动驾驶车辆, AMV
---

# ROS（机器人操作系统）快速入门指南

<p align="center"> <iframe src="https://player.vimeo.com/video/639236696?h=740f412ce5" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><a href="https://vimeo.com/639236696">ROS 介绍（带字幕）</a> 来自 <a href="https://vimeo.com/osrfoundation">Open Robotics</a> 在 <a href="https://vimeo.com/">Vimeo</a>。</p>

## 什么是 ROS？

[机器人操作系统（ROS）](https://www.ros.org/)是一个广泛用于机器人研究和工业的开源框架。ROS 提供了一系列[库和工具](https://www.ros.org/blog/ecosystem/)来帮助开发者创建机器人应用程序。ROS 设计为与各种[机器人平台](https://robots.ros.org/)配合使用，使其成为机器人专家的灵活而强大的工具。

### ROS 的主要特性

1. **模块化架构**：ROS 具有模块化架构，允许开发者通过组合称为[节点](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes)的更小、可重用组件来构建复杂系统。每个节点通常执行特定功能，节点之间通过[话题](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics)或[服务](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams)上的消息进行通信。

2. **通信中间件**：ROS 提供强大的通信基础设施，支持进程间通信和分布式计算。这通过数据流（话题）的发布-订阅模型和服务调用的请求-回复模型实现。

3. **硬件抽象**：ROS 在硬件上提供了一层抽象，使开发者能够编写与设备无关的代码。这允许相同的代码用于不同的硬件设置，便于更容易的集成和实验。

4. **工具和实用程序**：ROS 附带了丰富的可视化、调试和仿真工具和实用程序。例如，RViz 用于可视化传感器数据和机器人状态信息，而 Gazebo 提供了强大的仿真环境来测试算法和机器人设计。

5. **广泛的生态系统**：ROS 生态系统庞大且不断增长，有许多可用于不同机器人应用的包，包括导航、操作、感知等。社区积极参与这些包的开发和维护。

???+ note "ROS 版本的演变"

    自 2007 年开发以来，ROS 经历了[多个版本](https://wiki.ros.org/Distributions)的演变，每个版本都引入了新功能和改进以满足机器人社区不断增长的需求。ROS 的开发可以分为两个主要系列：ROS 1 和 ROS 2。本指南重点介绍 ROS 1 的长期支持（LTS）版本，称为 ROS Noetic Ninjemys，代码也应该适用于早期版本。

    ### ROS 1 与 ROS 2

    虽然 ROS 1 为机器人开发提供了坚实的基础，但 ROS 2 通过提供以下功能解决了其不足：

    - **实时性能**：改进了对实时系统和确定性行为的支持。
    - **安全性**：增强了安全功能，可在各种环境中安全可靠地运行。
    - **可扩展性**：更好地支持多机器人系统和大规模部署。
    - **跨平台支持**：扩展了与 Linux 以外各种操作系统的兼容性，包括 Windows 和 macOS。
    - **灵活通信**：使用 DDS 实现更灵活高效的进程间通信。

### ROS 消息和话题

在 ROS 中，节点之间的通信通过[消息](https://wiki.ros.org/Messages)和[话题](https://wiki.ros.org/Topics)实现。消息是定义节点之间交换信息的数据结构，而话题是发送和接收消息的命名通道。节点可以向话题发布消息或从话题订阅消息，使它们能够相互通信。这种发布-订阅模型允许节点之间的异步通信和解耦。机器人系统中的每个传感器或执行器通常将数据发布到话题，然后其他节点可以使用这些数据进行处理或控制。在本指南中，我们将重点关注图像、深度和点云消息以及相机话题。

## 使用 ROS 设置 Ultralytics YOLO

本指南已使用[此 ROS 环境](https://github.com/ambitious-octopus/rosbot_ros/tree/noetic)进行测试，这是 [ROSbot ROS 仓库](https://github.com/husarion/rosbot_ros)的一个分支。此环境包括 Ultralytics YOLO 包、用于轻松设置的 Docker 容器、全面的 ROS 包和用于快速测试的 Gazebo 世界。它设计为与 [Husarion ROSbot 2 PRO](https://husarion.com/manuals/rosbot/) 配合使用。提供的代码示例将在任何 ROS Noetic/Melodic 环境中工作，包括仿真和真实世界。

<p align="center">
  <img width="50%" src="https://github.com/ultralytics/docs/releases/download/0/husarion-rosbot-2-pro.avif" alt="Husarion ROSbot 2 PRO">
</p>

### 依赖安装

除了 ROS 环境外，您还需要安装以下依赖：

- **[ROS Numpy 包](https://github.com/eric-wieser/ros_numpy)**：这是在 ROS 图像消息和 numpy 数组之间快速转换所必需的。

    ```bash
    pip install ros_numpy
    ```

- **Ultralytics 包**：

    ```bash
    pip install ultralytics
    ```

## 将 Ultralytics 与 ROS `sensor_msgs/Image` 一起使用

`sensor_msgs/Image` [消息类型](https://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)在 ROS 中常用于表示图像数据。它包含编码、高度、宽度和像素数据的字段，适合传输相机或其他传感器捕获的图像。图像消息广泛用于机器人应用中的视觉感知、[目标检测](https://www.ultralytics.com/glossary/object-detection)和导航等任务。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/detection-segmentation-ros-gazebo.avif" alt="ROS Gazebo 中的检测和分割">
</p>

### 图像分步使用

以下代码片段演示了如何将 Ultralytics YOLO 包与 ROS 一起使用。在此示例中，我们订阅相机话题，使用 YOLO 处理传入的图像，并将检测到的对象发布到新话题以进行[检测](../tasks/detect.md)和[分割](../tasks/segment.md)。

首先，导入必要的库并实例化两个模型：一个用于[分割](../tasks/segment.md)，一个用于[检测](../tasks/detect.md)。初始化一个 ROS 节点（名称为 `ultralytics`）以启用与 ROS 主节点的通信。为确保稳定连接，我们包含一个短暂的暂停，让节点有足够的时间在继续之前建立连接。

```python
import time

import rospy

from ultralytics import YOLO

detection_model = YOLO("yolo11m.pt")
segmentation_model = YOLO("yolo11m-seg.pt")
rospy.init_node("ultralytics")
time.sleep(1)
```

初始化两个 ROS 话题：一个用于[检测](../tasks/detect.md)，一个用于[分割](../tasks/segment.md)。这些话题将用于发布标注图像，使其可供进一步处理。节点之间的通信使用 `sensor_msgs/Image` 消息实现。

```python
from sensor_msgs.msg import Image

det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
```

最后，创建一个订阅者，监听 `/camera/color/image_raw` 话题上的消息，并为每条新消息调用回调函数。此回调函数接收 `sensor_msgs/Image` 类型的消息，使用 `ros_numpy` 将其转换为 numpy 数组，使用先前实例化的 YOLO 模型处理图像，标注图像，然后将它们发布回相应的话题：`/ultralytics/detection/image` 用于检测，`/ultralytics/segmentation/image` 用于分割。

```python
import ros_numpy


def callback(data):
    """处理图像并发布标注图像的回调函数。"""
    array = ros_numpy.numpify(data)
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    if seg_image_pub.get_num_connections():
        seg_result = segmentation_model(array)
        seg_annotated = seg_result[0].plot(show=False)
        seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? example "完整代码"

    ```python
    import time

    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image

    from ultralytics import YOLO

    detection_model = YOLO("yolo11m.pt")
    segmentation_model = YOLO("yolo11m-seg.pt")
    rospy.init_node("ultralytics")
    time.sleep(1)

    det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
    seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


    def callback(data):
        """处理图像并发布标注图像的回调函数。"""
        array = ros_numpy.numpify(data)
        if det_image_pub.get_num_connections():
            det_result = detection_model(array)
            det_annotated = det_result[0].plot(show=False)
            det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        if seg_image_pub.get_num_connections():
            seg_result = segmentation_model(array)
            seg_annotated = seg_result[0].plot(show=False)
            seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


    rospy.Subscriber("/camera/color/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

???+ tip "调试"

    由于系统的分布式特性，调试 ROS（机器人操作系统）节点可能具有挑战性。以下几个工具可以帮助完成此过程：

    1. `rostopic echo <TOPIC-NAME>`：此命令允许您查看在特定话题上发布的消息，帮助您检查数据流。
    2. `rostopic list`：使用此命令列出 ROS 系统中所有可用的话题，让您了解活动数据流的概况。
    3. `rqt_graph`：此可视化工具显示节点之间的通信图，提供节点如何互连以及如何交互的洞察。
    4. 对于更复杂的可视化，如 3D 表示，您可以使用 [RViz](https://wiki.ros.org/rviz)。RViz（ROS 可视化）是 ROS 的强大 3D 可视化工具。它允许您实时可视化机器人及其环境的状态。使用 RViz，您可以查看传感器数据（例如 `sensor_msgs/Image`）、机器人模型状态和各种其他类型的信息，使调试和理解机器人系统的行为变得更容易。

### 使用 `std_msgs/String` 发布检测到的类别

标准 ROS 消息还包括 `std_msgs/String` 消息。在许多应用中，不需要重新发布整个标注图像；相反，只需要机器人视野中存在的类别。以下示例演示了如何使用 `std_msgs/String` [消息](https://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)将检测到的类别重新发布到 `/ultralytics/detection/classes` 话题。这些消息更轻量级并提供基本信息，使其对各种应用很有价值。

#### 示例用例

考虑一个配备相机和目标[检测模型](../tasks/detect.md)的仓库机器人。机器人可以将检测到的类别列表作为 `std_msgs/String` 消息发布，而不是通过网络发送大型标注图像。例如，当机器人检测到"箱子"、"托盘"和"叉车"等对象时，它会将这些类别发布到 `/ultralytics/detection/classes` 话题。然后，中央监控系统可以使用此信息实时跟踪库存、优化机器人的路径规划以避开障碍物，或触发特定操作（如拾取检测到的箱子）。这种方法减少了通信所需的带宽，并专注于传输关键数据。

### 字符串分步使用

此示例演示了如何将 Ultralytics YOLO 包与 ROS 一起使用。在此示例中，我们订阅相机话题，使用 YOLO 处理传入的图像，并使用 `std_msgs/String` 消息将检测到的对象发布到新话题 `/ultralytics/detection/classes`。`ros_numpy` 包用于将 ROS 图像消息转换为 numpy 数组以供 YOLO 处理。

```python
import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

detection_model = YOLO("yolo11m.pt")
rospy.init_node("ultralytics")
time.sleep(1)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)


def callback(data):
    """处理图像并发布检测到的类别的回调函数。"""
    array = ros_numpy.numpify(data)
    if classes_pub.get_num_connections():
        det_result = detection_model(array)
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
while True:
    rospy.spin()
```

## 将 Ultralytics 与 ROS 深度图像一起使用

除了 RGB 图像外，ROS 还支持[深度图像](https://en.wikipedia.org/wiki/Depth_map)，它提供对象与相机距离的信息。深度图像对于避障、3D 建图和定位等机器人应用至关重要。

深度图像是每个像素表示从相机到对象距离的图像。与捕获颜色的 RGB 图像不同，深度图像捕获空间信息，使机器人能够感知其环境的 3D 结构。

!!! tip "获取深度图像"

    深度图像可以使用各种传感器获取：

    1. [立体相机](https://en.wikipedia.org/wiki/Stereo_camera)：使用两个相机根据图像视差计算深度。
    2. [飞行时间（ToF）相机](https://en.wikipedia.org/wiki/Time-of-flight_camera)：测量光从对象返回所需的时间。
    3. [结构光传感器](https://en.wikipedia.org/wiki/Structured-light_3D_scanner)：投射图案并测量其在表面上的变形。

### 将 YOLO 与深度图像一起使用

在 ROS 中，深度图像由 `sensor_msgs/Image` 消息类型表示，其中包含编码、高度、宽度和像素数据的字段。深度图像的编码字段通常使用"16UC1"等格式，表示每像素 16 位无符号整数，其中每个值表示到对象的距离。深度图像通常与 RGB 图像结合使用，以提供更全面的环境视图。

使用 YOLO，可以从 RGB 和深度图像中提取和组合信息。例如，YOLO 可以检测 RGB 图像中的对象，此检测可用于精确定位深度图像中的相应区域。这允许提取检测对象的精确深度信息，增强机器人在三维空间中理解其环境的能力。

!!! warning "RGB-D 相机"

    使用深度图像时，必须确保 RGB 和深度图像正确对齐。RGB-D 相机（如 [Intel RealSense](https://realsenseai.com/) 系列）提供同步的 RGB 和深度图像，使组合两个来源的信息更容易。如果使用单独的 RGB 和深度相机，则必须校准它们以确保准确对齐。

#### 深度分步使用

在此示例中，我们使用 YOLO 分割图像并将提取的掩码应用于深度图像中的对象分割。这允许我们确定感兴趣对象的每个像素与相机焦点中心的距离。通过获取此距离信息，我们可以计算相机与场景中特定对象之间的距离。首先导入必要的库，创建 ROS 节点，并实例化分割模型和 ROS 话题。

```python
import time

import rospy
from std_msgs.msg import String

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)

segmentation_model = YOLO("yolo11m-seg.pt")

classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
```

接下来，定义一个处理传入深度图像消息的回调函数。该函数等待深度图像和 RGB 图像消息，将它们转换为 numpy 数组，并将分割模型应用于 RGB 图像。然后，它为每个检测到的对象提取分割掩码，并使用深度图像计算对象与相机的平均距离。大多数传感器都有最大距离，称为裁剪距离，超过该距离的值表示为 inf（`np.inf`）。在处理之前，重要的是过滤掉这些空值并将它们赋值为 `0`。最后，它将检测到的对象及其平均距离发布到 `/ultralytics/detection/distance` 话题。

```python
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image


def callback(data):
    """处理深度图像和 RGB 图像的回调函数。"""
    image = rospy.wait_for_message("/camera/color/image_raw", Image)
    image = ros_numpy.numpify(image)
    depth = ros_numpy.numpify(data)
    result = segmentation_model(image)

    all_objects = []
    for index, cls in enumerate(result[0].boxes.cls):
        class_index = int(cls.cpu().numpy())
        name = result[0].names[class_index]
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        obj = depth[mask == 1]
        obj = obj[~np.isnan(obj)]
        avg_distance = np.mean(obj) if len(obj) else np.inf
        all_objects.append(f"{name}: {avg_distance:.2f}m")

    classes_pub.publish(String(data=str(all_objects)))


rospy.Subscriber("/camera/depth/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? example "完整代码"

    ```python
    import time

    import numpy as np
    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from ultralytics import YOLO

    rospy.init_node("ultralytics")
    time.sleep(1)

    segmentation_model = YOLO("yolo11m-seg.pt")

    classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)


    def callback(data):
        """处理深度图像和 RGB 图像的回调函数。"""
        image = rospy.wait_for_message("/camera/color/image_raw", Image)
        image = ros_numpy.numpify(image)
        depth = ros_numpy.numpify(data)
        result = segmentation_model(image)

        all_objects = []
        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            obj = depth[mask == 1]
            obj = obj[~np.isnan(obj)]
            avg_distance = np.mean(obj) if len(obj) else np.inf
            all_objects.append(f"{name}: {avg_distance:.2f}m")

        classes_pub.publish(String(data=str(all_objects)))


    rospy.Subscriber("/camera/depth/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

## 将 Ultralytics 与 ROS `sensor_msgs/PointCloud2` 一起使用

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/detection-segmentation-ros-gazebo-1.avif" alt="ROS Gazebo 中的检测和分割">
</p>

`sensor_msgs/PointCloud2` [消息类型](https://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html)是 ROS 中用于表示 3D 点云数据的数据结构。此消息类型是机器人应用的组成部分，支持 3D 建图、对象识别和定位等任务。

点云是在三维坐标系中定义的数据点集合。这些数据点表示通过 3D 扫描技术捕获的对象或场景的外表面。云中的每个点都有 `X`、`Y` 和 `Z` 坐标，对应于其在空间中的位置，还可能包括颜色和强度等附加信息。

!!! warning "参考坐标系"

    使用 `sensor_msgs/PointCloud2` 时，必须考虑获取点云数据的传感器的参考坐标系。点云最初在传感器的参考坐标系中捕获。您可以通过监听 `/tf_static` 话题来确定此参考坐标系。但是，根据您的特定应用需求，您可能需要将点云转换为另一个参考坐标系。此转换可以使用 `tf2_ros` 包实现，该包提供了管理坐标系和在它们之间转换数据的工具。

!!! tip "获取点云"

    点云可以使用各种传感器获取：

    1. **LIDAR（光探测和测距）**：使用激光脉冲测量到对象的距离并创建高[精度](https://www.ultralytics.com/glossary/precision) 3D 地图。
    2. **深度相机**：捕获每个像素的深度信息，允许场景的 3D 重建。
    3. **立体相机**：利用两个或多个相机通过三角测量获取深度信息。
    4. **结构光扫描仪**：将已知图案投射到表面上并测量变形以计算深度。

### 将 YOLO 与点云一起使用

要将 YOLO 与 `sensor_msgs/PointCloud2` 类型消息集成，我们可以采用与深度图类似的方法。通过利用点云中嵌入的颜色信息，我们可以提取 2D 图像，使用 YOLO 对此图像执行分割，然后将生成的掩码应用于三维点以隔离感兴趣的 3D 对象。

对于处理点云，我们建议使用 Open3D（`pip install open3d`），这是一个用户友好的 Python 库。Open3D 提供了强大的工具来管理点云数据结构、可视化它们并无缝执行复杂操作。此库可以显著简化流程并增强我们结合基于 YOLO 的分割来操作和分析点云的能力。

#### 点云分步使用

导入必要的库并实例化用于分割的 YOLO 模型。

```python
import time

import rospy

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)
segmentation_model = YOLO("yolo11m-seg.pt")
```

创建一个函数 `pointcloud2_to_array`，将 `sensor_msgs/PointCloud2` 消息转换为两个 numpy 数组。`sensor_msgs/PointCloud2` 消息根据获取图像的 `width` 和 `height` 包含 `n` 个点。例如，`480 x 640` 图像将有 `307,200` 个点。每个点包括三个空间坐标（`xyz`）和相应的 `RGB` 格式颜色。这些可以被视为两个独立的信息通道。

该函数以原始相机分辨率（`width x height`）的格式返回 `xyz` 坐标和 `RGB` 值。大多数传感器都有最大距离，称为裁剪距离，超过该距离的值表示为 inf（`np.inf`）。在处理之前，重要的是过滤掉这些空值并将它们赋值为 `0`。

```python
import numpy as np
import ros_numpy


def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
    """将 ROS PointCloud2 消息转换为 numpy 数组。

    Args:
        pointcloud2 (PointCloud2): PointCloud2 消息

    Returns:
        (tuple): 包含 (xyz, rgb) 的元组
    """
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    nan_rows = np.isnan(xyz).all(axis=2)
    xyz[nan_rows] = [0, 0, 0]
    rgb[nan_rows] = [0, 0, 0]
    return xyz, rgb
```

接下来，订阅 `/camera/depth/points` 话题以接收点云消息，并将 `sensor_msgs/PointCloud2` 消息转换为包含 XYZ 坐标和 RGB 值的 numpy 数组（使用 `pointcloud2_to_array` 函数）。使用 YOLO 模型处理 RGB 图像以提取分割对象。对于每个检测到的对象，提取分割掩码并将其应用于 RGB 图像和 XYZ 坐标，以在 3D 空间中隔离对象。

处理掩码很简单，因为它由二进制值组成，`1` 表示对象存在，`0` 表示不存在。要应用掩码，只需将原始通道乘以掩码。此操作有效地隔离了图像中感兴趣的对象。最后，创建一个 Open3D 点云对象，并在 3D 空间中可视化带有关联颜色的分割对象。

```python
import sys

import open3d as o3d

ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("未检测到对象")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

??? example "完整代码"

    ```python
    import sys
    import time

    import numpy as np
    import open3d as o3d
    import ros_numpy
    import rospy
    from sensor_msgs.msg import PointCloud2

    from ultralytics import YOLO

    rospy.init_node("ultralytics")
    time.sleep(1)
    segmentation_model = YOLO("yolo11m-seg.pt")


    def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
        """将 ROS PointCloud2 消息转换为 numpy 数组。

        Args:
            pointcloud2 (PointCloud2): PointCloud2 消息

        Returns:
            (tuple): 包含 (xyz, rgb) 的元组
        """
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
        split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
        rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
        xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
        xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
        nan_rows = np.isnan(xyz).all(axis=2)
        xyz[nan_rows] = [0, 0, 0]
        rgb[nan_rows] = [0, 0, 0]
        return xyz, rgb


    ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
    xyz, rgb = pointcloud2_to_array(ros_cloud)
    result = segmentation_model(rgb)

    if not len(result[0].boxes.cls):
        print("未检测到对象")
        sys.exit()

    classes = result[0].boxes.cls.cpu().numpy().astype(int)
    for index, class_id in enumerate(classes):
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        mask_expanded = np.stack([mask, mask, mask], axis=2)

        obj_rgb = rgb * mask_expanded
        obj_xyz = xyz * mask_expanded

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
        pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
        o3d.visualization.draw_geometries([pcd])
    ```

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/point-cloud-segmentation-ultralytics.avif" alt="使用 Ultralytics 进行点云分割">
</p>

## 常见问题

### 什么是机器人操作系统（ROS）？

[机器人操作系统（ROS）](https://www.ros.org/)是一个常用于机器人领域的开源框架，帮助开发者创建强大的机器人应用程序。它提供了一系列[库和工具](https://www.ros.org/blog/ecosystem/)用于构建和与机器人系统交互，使复杂应用程序的开发更加容易。ROS 支持使用话题或服务上的消息在节点之间进行通信。

### 如何将 Ultralytics YOLO 与 ROS 集成以进行实时目标检测？

将 Ultralytics YOLO 与 ROS 集成涉及设置 ROS 环境并使用 YOLO 处理传感器数据。首先安装所需的依赖项，如 `ros_numpy` 和 Ultralytics YOLO：

```bash
pip install ros_numpy ultralytics
```

接下来，创建一个 ROS 节点并订阅[图像话题](../tasks/detect.md)以处理传入的数据。以下是一个最小示例：

```python
import ros_numpy
import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO

detection_model = YOLO("yolo11m.pt")
rospy.init_node("ultralytics")
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)


def callback(data):
    array = ros_numpy.numpify(data)
    det_result = detection_model(array)
    det_annotated = det_result[0].plot(show=False)
    det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.spin()
```

### 什么是 ROS 话题，它们如何在 Ultralytics YOLO 中使用？

ROS 话题通过发布-订阅模型促进 ROS 网络中节点之间的通信。话题是节点用于异步发送和接收消息的命名通道。在 Ultralytics YOLO 的上下文中，您可以让节点订阅图像话题，使用 YOLO 处理图像以执行[检测](https://docs.ultralytics.com/tasks/detect/)或[分割](https://docs.ultralytics.com/tasks/segment/)等任务，并将结果发布到新话题。

例如，订阅相机话题并处理传入的图像进行检测：

```python
rospy.Subscriber("/camera/color/image_raw", Image, callback)
```

### 为什么在 ROS 中将深度图像与 Ultralytics YOLO 一起使用？

ROS 中的深度图像（由 `sensor_msgs/Image` 表示）提供对象与相机的距离，对于避障、3D 建图和定位等任务至关重要。通过[使用深度信息](https://en.wikipedia.org/wiki/Depth_map)与 RGB 图像结合，机器人可以更好地理解其 3D 环境。

使用 YOLO，您可以从 RGB 图像中提取[分割掩码](https://www.ultralytics.com/glossary/image-segmentation)并将这些掩码应用于深度图像以获取精确的 3D 对象信息，从而提高机器人导航和与周围环境交互的能力。

### 如何在 ROS 中使用 YOLO 可视化 3D 点云？

要在 ROS 中使用 YOLO 可视化 3D 点云：

1. 将 `sensor_msgs/PointCloud2` 消息转换为 numpy 数组。
2. 使用 YOLO 分割 RGB 图像。
3. 将分割掩码应用于点云。

以下是使用 [Open3D](https://www.open3d.org/) 进行可视化的示例：

```python
import sys

import open3d as o3d
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2

from ultralytics import YOLO

rospy.init_node("ultralytics")
segmentation_model = YOLO("yolo11m-seg.pt")


def pointcloud2_to_array(pointcloud2):
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    return xyz, rgb


ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("未检测到对象")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((-1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((-1, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

此方法提供分割对象的 3D 可视化，对于[机器人应用](https://docs.ultralytics.com/guides/steps-of-a-cv-project/)中的导航和操作等任务非常有用。
