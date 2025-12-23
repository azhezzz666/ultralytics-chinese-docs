# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "Ultralytics YOLO 多边形区域",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 多边形顶点
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR 值
        "text_color": (255, 255, 255),  # 区域文本颜色
    },
    {
        "name": "Ultralytics YOLO 矩形区域",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 多边形顶点
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR 值
        "text_color": (0, 0, 0),  # 区域文本颜色
    },
]


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """处理视频帧中区域操作的鼠标事件。

    该函数为计数区域启用交互式区域选择和拖动功能。它响应鼠标按下、移动和释放事件，
    允许用户实时选择和重新定位计数区域。

    Args:
        event (int): 鼠标事件类型（例如 cv2.EVENT_LBUTTONDOWN、cv2.EVENT_MOUSEMOVE）。
        x (int): 鼠标指针的 x 坐标。
        y (int): 鼠标指针的 y 坐标。
        flags (int): OpenCV 传递的附加标志。
        param (Any): 传递给回调的附加参数。

    Examples:
        设置鼠标回调以进行交互式区域操作
        >>> cv2.setMouseCallback("window_name", mouse_callback)
    """
    global current_region

    # 鼠标左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # 鼠标左键释放事件
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights: str = "yolo11n.pt",
    source: str | None = None,
    device: str = "cpu",
    view_img: bool = False,
    save_img: bool = False,
    exist_ok: bool = False,
    classes: list[int] | None = None,
    line_thickness: int = 2,
    track_thickness: int = 2,
    region_thickness: int = 2,
) -> None:
    """使用 YOLO 和 ByteTrack 在指定区域内运行目标检测和计数。

    该函数在用户定义的多边形或矩形区域内执行实时目标检测、跟踪和计数。
    它支持交互式区域操作、多个计数区域，以及实时查看和视频保存功能。

    Args:
        weights (str): YOLO 模型权重文件的路径。
        source (str): 输入视频文件的路径。
        device (str): 处理设备规格（'cpu'、'0'、'1' 等）。
        view_img (bool): 在实时窗口中显示结果。
        save_img (bool): 将处理后的视频保存到文件。
        exist_ok (bool): 覆盖现有输出文件而不递增。
        classes (list[int], optional): 要检测和跟踪的特定类别 ID。
        line_thickness (int): 边界框线条粗细。
        track_thickness (int): 目标跟踪线条粗细。
        region_thickness (int): 计数区域边界粗细。

    Examples:
        使用默认设置运行区域计数
        >>> run(source="video.mp4", view_img=True)

        使用自定义模型和特定类别运行
        >>> run(weights="yolo11s.pt", source="traffic.mp4", classes=[0, 2, 3], device="0")
    """
    vid_frame_count = 0

    # 检查源路径
    if not Path(source).exists():
        raise FileNotFoundError(f"源路径 '{source}' 不存在。")

    # 设置模型
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # 提取类别名称
    names = model.names

    # 视频设置
    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 输出设置
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"), fourcc, fps, (frame_width, frame_height))

    # 遍历视频帧
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # 提取结果
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 边界框中心

                track = track_history[track_id]  # 跟踪线绘制
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # 检查检测是否在区域内
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # 绘制区域（多边形/矩形）
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coordinates], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLO 可移动区域计数器")
                cv2.setMouseCallback("Ultralytics YOLO 可移动区域计数器", mouse_callback)
            cv2.imshow("Ultralytics YOLO 可移动区域计数器", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # 为每个区域重新初始化计数
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    """解析区域计数应用程序的命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="初始权重路径")
    parser.add_argument("--device", default="", help="cuda 设备，例如 0 或 0,1,2,3 或 cpu")
    parser.add_argument("--source", type=str, required=True, help="视频文件路径")
    parser.add_argument("--view-img", action="store_true", help="显示结果")
    parser.add_argument("--save-img", action="store_true", help="保存结果")
    parser.add_argument("--exist-ok", action="store_true", help="现有项目/名称可用，不递增")
    parser.add_argument("--classes", nargs="+", type=int, help="按类别过滤：--classes 0，或 --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="边界框粗细")
    parser.add_argument("--track-thickness", type=int, default=2, help="跟踪线粗细")
    parser.add_argument("--region-thickness", type=int, default=4, help="区域粗细")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """使用提供的选项执行主要区域计数功能。"""
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
