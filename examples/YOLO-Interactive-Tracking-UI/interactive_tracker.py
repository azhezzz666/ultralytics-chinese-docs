# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import time

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

enable_gpu = False  # 如果使用 CUDA 运行，设置为 True
model_file = "yolo11s.pt"  # 模型文件路径
show_fps = True  # 如果为 True，在左上角显示当前 FPS
show_conf = False  # 显示或隐藏置信度分数
save_video = False  # 设置为 True 以保存输出视频
video_output_path = "interactive_tracker_output.avi"  # 输出视频文件名


conf = 0.3  # 目标检测的最小置信度（较低 = 更多检测，可能更多误检）
iou = 0.3  # NMS 的 IoU 阈值（较高 = 允许更少重叠）
max_det = 20  # 每张图像的最大目标数（对于拥挤场景可增加）

tracker = "bytetrack.yaml"  # 跟踪器配置: 'bytetrack.yaml', 'botsort.yaml' 等
track_args = {
    "persist": True,  # 保持帧历史作为流以进行连续跟踪
    "verbose": False,  # 打印跟踪器的调试信息
}

window_name = "Ultralytics YOLO 交互式跟踪"  # 输出窗口名称

LOGGER.info("🚀 正在初始化模型...")
if enable_gpu:
    LOGGER.info("使用 GPU...")
    model = YOLO(model_file)
    model.to("cuda")
else:
    LOGGER.info("使用 CPU...")
    model = YOLO(model_file, task="detect")

classes = model.names  # 存储模型类别名称

cap = cv2.VideoCapture(0)  # 如需要可替换为视频路径
if not cap.isOpened():
    raise SystemError("无法打开视频源。")

vw = None  # 在读取第一帧后延迟初始化

selected_object_id = None
selected_bbox = None
selected_center = None
latest_detections: list[list[float]] = []


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """计算边界框的中心点。

    Args:
        x1 (int): 左上角 X 坐标。
        y1 (int): 左上角 Y 坐标。
        x2 (int): 右下角 X 坐标。
        y2 (int): 右下角 Y 坐标。

    Returns:
        center_x (int): 中心点的 X 坐标。
        center_y (int): 中心点的 Y 坐标。
    """
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x: int, mid_y: int, direction: str, img_shape: tuple[int, int, int]) -> tuple[int, int]:
    """计算从中心向图像边缘延伸线的端点。

    Args:
        mid_x (int): 中点的 X 坐标。
        mid_y (int): 中点的 Y 坐标。
        direction (str): 延伸方向 ('left', 'right', 'up', 'down')。
        img_shape (tuple[int, int, int]): 图像形状 (高度, 宽度, 通道)。

    Returns:
        end_x (int): 端点的 X 坐标。
        end_y (int): 端点的 Y 坐标。
    """
    h, w = img_shape[:2]
    if direction == "down":
        return mid_x, h - 1
    elif direction == "left":
        return 0, mid_y
    elif direction == "right":
        return w - 1, mid_y
    elif direction == "up":
        return mid_x, 0
    else:
        return mid_x, mid_y


def draw_tracking_scope(im, bbox: tuple, color: tuple) -> None:
    """绘制从边界框延伸到图像边缘的跟踪范围线。

    Args:
        im (np.ndarray): 要绘制的图像数组。
        bbox (tuple): 边界框坐标 (x1, y1, x2, y2)。
        color (tuple): 绘制用的 BGR 格式颜色。
    """
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    """处理鼠标点击事件以选择要聚焦跟踪的目标。

    Args:
        event (int): OpenCV 鼠标事件类型。
        x (int): 鼠标事件的 X 坐标。
        y (int): 鼠标事件的 Y 坐标。
        flags (int): OpenCV 传递的任何相关标志。
        param (Any): 附加参数（未使用）。
    """
    global selected_object_id, latest_detections
    if event == cv2.EVENT_LBUTTONDOWN:
        if not latest_detections:
            return
        min_area = float("inf")
        best_match = None
        for track in latest_detections:
            if len(track) < 6:
                continue
            x1, y1, x2, y2 = map(int, track[:4])
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < min_area:
                    track_id = int(track[4]) if len(track) >= 7 else -1
                    class_id = int(track[6]) if len(track) >= 7 else int(track[5])
                    min_area = area
                    best_match = (track_id, classes.get(class_id, str(class_id)))
        if best_match:
            selected_object_id, label = best_match
            LOGGER.info(f"开始跟踪: {label} (ID {selected_object_id})")


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    success, im = cap.read()
    if not success:
        break

    results = model.track(im, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
    annotator = Annotator(im)
    detections = results[0].boxes.data if results[0].boxes is not None else []
    latest_detections = detections.cpu().tolist() if hasattr(detections, "cpu") else list(detections)  # type: ignore[arg-type]
    detected_objects: list[str] = []
    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = colors(track_id, True)
        txt_color = annotator.get_txt_color(color)
        conf_score = float(track[5]) if len(track) >= 7 else 0.0
        class_name = classes.get(class_id, str(class_id))
        label = f"{class_name} ID {track_id}" + (f" ({conf_score:.2f})" if show_conf else "")
        center = get_center(x1, y1, x2, y2)
        detected_objects.append(f"{class_name}#{track_id}@{center[0]},{center[1]}")
        if track_id == selected_object_id:
            draw_tracking_scope(im, (x1, y1, x2, y2), color)
            cv2.circle(im, center, 6, color, -1)

            # 用于吸引注意力的脉冲圆
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(im, center, pulse_radius, color, 2)

            annotator.box_label([x1, y1, x2, y2], label=f"激活: 跟踪 {track_id}", color=color)
        else:
            # 为其他目标绘制虚线框
            for i in range(x1, x2, 10):
                cv2.line(im, (i, y1), (i + 5, y1), color, 3)
                cv2.line(im, (i, y2), (i + 5, y2), color, 3)
            for i in range(y1, y2, 10):
                cv2.line(im, (x1, i), (x1, i + 5), color, 3)
                cv2.line(im, (x2, i), (x2, i + 5), color, 3)
            # 绘制带背景的标签文本
            (tw, th), bl = cv2.getTextSize(label, 0, 0.7, 2)
            cv2.rectangle(im, (x1 + 5 - 5, y1 + 20 - th - 5), (x1 + 5 + tw + 5, y1 + 20 + bl), color, -1)
            cv2.putText(im, label, (x1 + 5, y1 + 20), 0, 0.7, txt_color, 1, cv2.LINE_AA)

    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # 绘制带背景的 FPS 文本
        fps_text = f"FPS: {fps_display}"
        (tw, th), bl = cv2.getTextSize(fps_text, 0, 0.7, 2)
        cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
        cv2.putText(im, fps_text, (10, 25), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

    if save_video and vw is None:
        h, w = im.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = float(fps) if fps and fps > 0 else 30.0
        ext = video_output_path.lower()
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext.endswith(".avi") else "mp4v"))
        vw = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)
    # 终端日志记录
    LOGGER.info(
        f"检测到 {len(detections)} 个目标: {' | '.join(detected_objects)}"
        if detected_objects
        else f"检测到 {len(detections)} 个目标。"
    )

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        LOGGER.info("跟踪已重置。")
        selected_object_id = None

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()
