---
comments: true
description: 使用 Ultralytics YOLO11 的实时目标检测增强您的安全性。减少误报并与现有系统无缝集成。
keywords: YOLO11, 安全报警系统, 实时目标检测, Ultralytics, 计算机视觉, 集成, 误报
---

# 使用 Ultralytics YOLO11 的安全报警系统项目

<img src="https://github.com/ultralytics/docs/releases/download/0/security-alarm-system-ultralytics-yolov8.avif" alt="安全报警系统">

使用 Ultralytics YOLO11 的安全报警系统项目集成了先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)功能来增强安全措施。由 Ultralytics 开发的 YOLO11 提供实时[目标检测](https://www.ultralytics.com/glossary/object-detection)，使系统能够及时识别和响应潜在的安全威胁。该项目提供多项优势：

- **实时检测**：YOLO11 的高效性使安全报警系统能够实时检测和响应安全事件，最大限度地减少响应时间。
- **[准确率](https://www.ultralytics.com/glossary/accuracy)**：YOLO11 以其目标检测的准确性而闻名，减少误报并增强安全报警系统的可靠性。
- **集成能力**：该项目可以与现有的安全基础设施无缝集成，提供升级的智能监控层。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/DTjtBnSK2fY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 + 解决方案的安全报警系统 <a href="https://www.ultralytics.com/glossary/object-detection">目标检测</a>
</p>

???+ note

    需要生成应用密码

- 导航到[应用密码生成器](https://myaccount.google.com/apppasswords)，指定应用名称如"security project"，并获取 16 位密码。复制此密码并粘贴到下面代码中指定的 `password` 字段。

!!! example "使用 Ultralytics YOLO 的安全报警系统"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        from_email = "abc@gmail.com"  # 发送者邮箱地址
        password = "---- ---- ---- ----"  # 通过 https://myaccount.google.com/apppasswords 生成的 16 位密码
        to_email = "xyz@gmail.com"  # 接收者邮箱地址

        # 初始化安全报警对象
        securityalarm = solutions.SecurityAlarm(
            show=True,  # 显示输出
            model="yolo11n.pt",  # 例如 yolo11s.pt, yolo11m.pt
            records=1,  # 发送邮件的总检测计数
        )

        securityalarm.authenticate(from_email, password, to_email)  # 验证邮件服务器

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或视频处理已成功完成。")
                break

            results = securityalarm(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

当您运行代码时，如果检测到任何目标，您将收到一封邮件通知。通知会立即发送，不会重复发送。您可以自定义代码以满足您的项目需求。

#### 收到的邮件示例

<img width="256" src="https://github.com/ultralytics/docs/releases/download/0/email-received-sample.avif" alt="收到的邮件示例">

### `SecurityAlarm` 参数

下表列出了 `SecurityAlarm` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "records"]) }}

`SecurityAlarm` 解决方案支持多种 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以使用以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 工作原理

安全报警系统使用[目标跟踪](https://docs.ultralytics.com/modes/track/)来监控视频流并检测潜在的安全威胁。当系统检测到超过指定阈值（由 `records` 参数设置）的目标时，它会自动发送带有显示检测到目标的图像附件的邮件通知。

该系统利用 [SecurityAlarm 类](https://docs.ultralytics.com/reference/solutions/security_alarm/)提供以下方法：

1. 处理帧并提取目标检测
2. 用边界框标注检测到的目标
3. 当检测阈值超过时发送邮件通知

此实现非常适合家庭安全、零售监控和其他需要立即通知检测到目标的监控应用。

## 常见问题

### Ultralytics YOLO11 如何提高安全报警系统的准确性？

Ultralytics YOLO11 通过提供高精度、实时的目标检测来增强安全报警系统。其先进的算法显著减少误报，确保系统仅对真正的威胁做出响应。这种增强的可靠性可以与现有的安全基础设施无缝集成，提升整体监控质量。

### 我可以将 Ultralytics YOLO11 与我现有的安全基础设施集成吗？

是的，Ultralytics YOLO11 可以与您现有的安全基础设施无缝集成。该系统支持各种模式并提供自定义灵活性，允许您使用先进的目标检测功能增强现有设置。有关在项目中集成 YOLO11 的详细说明，请访问[集成部分](https://docs.ultralytics.com/integrations/)。

### 运行 Ultralytics YOLO11 的存储要求是什么？

在标准设置上运行 Ultralytics YOLO11 通常需要约 5GB 的可用磁盘空间。这包括存储 YOLO11 模型和任何其他依赖项的空间。对于基于云的解决方案，[Ultralytics HUB](https://docs.ultralytics.com/hub/) 提供高效的项目管理和数据集处理，可以优化存储需求。了解更多关于 [Pro 计划](../hub/pro.md)的增强功能，包括扩展存储。

### Ultralytics YOLO11 与 Faster R-CNN 或 SSD 等其他目标检测模型有什么不同？

Ultralytics YOLO11 凭借其实时检测能力和更高的准确率，相比 Faster R-CNN 或 SSD 等模型具有优势。其独特的架构允许在不牺牲[精度](https://www.ultralytics.com/glossary/precision)的情况下更快地处理图像，使其非常适合安全报警系统等时间敏感的应用。有关目标检测模型的全面比较，您可以探索我们的[指南](https://docs.ultralytics.com/models/)。

### 如何使用 Ultralytics YOLO11 减少安全系统中的误报频率？

为了减少误报，请确保您的 Ultralytics YOLO11 模型使用多样化且标注良好的数据集进行充分训练。微调超参数并定期使用新数据更新模型可以显著提高检测准确性。详细的[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)技术可以在我们的[超参数调优指南](../guides/hyperparameter-tuning.md)中找到。
