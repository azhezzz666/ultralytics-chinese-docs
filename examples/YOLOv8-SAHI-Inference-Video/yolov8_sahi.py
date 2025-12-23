# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_model_weights

from ultralytics.utils.files import increment_path


class SAHIInference:
    """使用 Ultralytics YOLO11 和 SAHI 对视频进行目标检测，支持查看、保存和跟踪结果。

    该类将 SAHI（切片辅助超推理）与 YOLO11 模型集成，通过将大图像切片成较小的片段、
    对每个切片运行推理，然后合并结果来执行高效的目标检测。

    Attributes:
        detection_model (AutoDetectionModel): 使用 SAHI 功能包装的已加载 YOLO11 模型。

    Methods:
        load_model: 使用指定权重加载 YOLO11 模型，用于 SAHI 目标检测。
        inference: 使用 YOLO11 和 SAHI 对视频运行目标检测。
        parse_opt: 解析推理过程的命令行参数。

    Examples:
        初始化并对视频运行 SAHI 推理
        >>> sahi_inference = SAHIInference()
        >>> sahi_inference.inference(weights="yolo11n.pt", source="video.mp4", view_img=True)
    """

    def __init__(self):
        """初始化 SAHIInference 类，用于使用 SAHI 和 YOLO11 模型执行切片推理。"""
        self.detection_model = None

    def load_model(self, weights: str, device: str) -> None:
        """使用指定权重加载 YOLO11 模型，用于 SAHI 目标检测。

        Args:
            weights (str): 模型权重文件的路径。
            device (str): CUDA 设备，例如 '0' 或 '0,1,2,3' 或 'cpu'。
        """
        from ultralytics.utils.torch_utils import select_device

        if weights and os.path.exists(weights):
            yolo11_model_path = weights
        else:
            yolo11_model_path = f"models/{weights}"
            download_model_weights(yolo11_model_path)  # 如果不存在则下载模型
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device=select_device(device)
        )

    def inference(
        self,
        weights: str = "yolo11n.pt",
        source: str = "test.mp4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
        device: str = "",
        hide_conf: bool = False,
        slice_width: int = 512,
        slice_height: int = 512,
    ) -> None:
        """使用 YOLO11 和 SAHI 对视频运行目标检测。

        该函数处理视频的每一帧，使用 SAHI 应用切片推理，并可选择显示和/或保存带有边界框和标签的结果。

        Args:
            weights (str): 模型权重路径。
            source (str): 视频文件路径。
            view_img (bool): 是否在窗口中显示结果。
            save_img (bool): 是否将结果保存到视频文件。
            exist_ok (bool): 是否覆盖现有输出文件。
            device (str, optional): CUDA 设备，例如 '0' 或 '0,1,2,3' 或 'cpu'。
            hide_conf (bool, optional): 在输出中显示或隐藏置信度的标志。
            slice_width (int, optional): 推理的切片宽度。
            slice_height (int, optional): 推理的切片高度。
        """
        # 视频设置
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频源：'{source}'")

        save_dir = None
        if save_img:
            save_dir = increment_path("runs/detect/predict", exist_ok)
            save_dir.mkdir(parents=True, exist_ok=True)

        # 加载模型
        self.load_model(weights, device)
        idx = 0  # 图像帧写入索引
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 使用 SAHI 执行切片预测
            results = get_sliced_prediction(
                frame[..., ::-1],  # 将 BGR 转换为 RGB
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
            )

            # 如果请求则显示结果
            if view_img:
                cv2.imshow("Ultralytics YOLO 推理", frame)

            # 如果请求则保存结果
            if save_img and save_dir is not None:
                idx += 1
                results.export_visuals(export_dir=save_dir, file_name=f"img_{idx}", hide_conf=hide_conf)

            # 如果按下 'q' 则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 清理资源
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def parse_opt() -> argparse.Namespace:
        """解析推理过程的命令行参数。

        Returns:
            (argparse.Namespace): 解析后的命令行参数。
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolo11n.pt", help="初始权重路径")
        parser.add_argument("--source", type=str, required=True, help="视频文件路径")
        parser.add_argument("--view-img", action="store_true", help="显示结果")
        parser.add_argument("--save-img", action="store_true", help="保存结果")
        parser.add_argument("--exist-ok", action="store_true", help="现有项目/名称可用，不递增")
        parser.add_argument("--device", default="", help="cuda 设备，例如 0 或 0,1,2,3 或 cpu")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="显示或隐藏置信度")
        parser.add_argument("--slice-width", default=512, type=int, help="推理的切片宽度")
        parser.add_argument("--slice-height", default=512, type=int, help="推理的切片高度")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
