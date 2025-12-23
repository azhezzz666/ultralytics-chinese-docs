# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


class TorchVisionVideoClassifier:
    """使用预训练 TorchVision 模型进行动作识别的视频分类器。

    该类提供了使用 TorchVision 视频模型集合中各种预训练模型进行视频分类的接口，
    支持 S3D、R3D、Swin3D 和 MViT 等架构。

    属性:
        model (torch.nn.Module): 加载的用于视频分类的 TorchVision 模型。
        weights (torchvision.models.video.Weights): 模型使用的权重。
        device (torch.device): 模型加载的设备。

    方法:
        available_model_names: 返回可用模型名称列表。
        preprocess_crops_for_video_cls: 预处理用于视频分类的裁剪图像。
        __call__: 对给定序列执行推理。
        postprocess: 后处理模型输出。

    示例:
        >>> classifier = TorchVisionVideoClassifier("s3d", device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)

    参考:
        https://pytorch.org/vision/stable/
    """

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str | torch.device = ""):
        """使用指定的模型名称和设备初始化视频分类器。

        参数:
            model_name (str): 要使用的模型名称，必须是可用模型之一。
            device (str | torch.device): 运行模型的设备。
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> list[str]:
        """获取可用模型名称列表。

        返回:
            (list[str]): 可与此分类器一起使用的可用模型名称列表。
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """预处理用于视频分类的裁剪图像列表。

        参数:
            crops (list[np.ndarray]): 要预处理的裁剪图像列表，每个裁剪图像的维度应为 (H, W, C)。
            input_size (list[int], optional): 模型的目标输入尺寸。

        返回:
            (torch.Tensor): 预处理后的裁剪图像张量，维度为 (1, T, C, H, W)。
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """对给定序列执行推理。

        参数:
            sequences (torch.Tensor): 模型的输入序列，批量视频帧的维度为 (B, T, C, H, W)，
                单个视频帧的维度为 (T, C, H, W)。

        返回:
            (torch.Tensor): 模型的输出 logits。
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[str], list[float]]:
        """后处理模型的批量输出。

        参数:
            outputs (torch.Tensor): 模型的输出 logits。

        返回:
            pred_labels (list[str]): 预测的标签。
            pred_confs (list[float]): 预测的置信度。
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs


class HuggingFaceVideoClassifier:
    """使用 Hugging Face transformer 模型的零样本视频分类器。

    该类提供了使用 Hugging Face 模型进行零样本视频分类的接口，
    支持自定义标签集和各种用于视频理解的 transformer 架构。

    属性:
        fp16 (bool): 是否使用 FP16 进行推理。
        labels (list[str]): 用于零样本分类的标签列表。
        device (torch.device): 模型加载的设备。
        processor (transformers.AutoProcessor): 模型的处理器。
        model (transformers.AutoModel): 加载的 Hugging Face 模型。

    方法:
        preprocess_crops_for_video_cls: 预处理用于视频分类的裁剪图像。
        __call__: 对给定序列执行推理。
        postprocess: 后处理模型输出。

    示例:
        >>> labels = ["walking", "running", "dancing"]
        >>> classifier = HuggingFaceVideoClassifier(labels, device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)
    """

    def __init__(
        self,
        labels: list[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str | torch.device = "",
        fp16: bool = False,
    ):
        """使用指定的模型名称初始化 HuggingFaceVideoClassifier。

        参数:
            labels (list[str]): 用于零样本分类的标签列表。
            model_name (str): 要使用的模型名称。
            device (str | torch.device): 运行模型的设备。
            fp16 (bool): 是否使用 FP16 进行推理。
        """
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """预处理用于视频分类的裁剪图像列表。

        参数:
            crops (list[np.ndarray]): 要预处理的裁剪图像列表，每个裁剪图像的维度应为 (H, W, C)。
            input_size (list[int], optional): 模型的目标输入尺寸。

        返回:
            (torch.Tensor): 预处理后的裁剪图像张量，维度为 (1, T, C, H, W)。
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if self.fp16:
            output = output.half()
        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """对给定序列执行推理。

        参数:
            sequences (torch.Tensor): 批量输入视频帧，形状为 (B, T, H, W, C)。

        返回:
            (torch.Tensor): 模型的输出 logits。
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[list[str]], list[list[float]]]:
        """后处理模型的批量输出。

        参数:
            outputs (torch.Tensor): 模型的输出 logits。

        返回:
            pred_labels (list[list[str]]): 每个样本预测的 top2 标签。
            pred_confs (list[list[float]]): 每个样本预测的 top2 置信度。
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs  # Assuming outputs is already the logits tensor
            probs = logits_per_video.softmax(dim=-1)  # Use softmax to convert logits to probabilities

        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs


def crop_and_pad(frame: np.ndarray, box: list[float], margin_percent: int) -> np.ndarray:
    """带边距裁剪边界框并从帧中获取正方形裁剪图像。

    参数:
        frame (np.ndarray): 要裁剪的输入帧。
        box (list[float]): 边界框坐标 [x1, y1, x2, y2]。
        margin_percent (int): 边界框周围添加的边距百分比。

    返回:
        (np.ndarray): 裁剪并调整大小后的正方形图像。
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # 添加边距
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # 从帧中获取正方形裁剪
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "yolo11n.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: str | None = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: list[str] | None = None,
) -> None:
    """使用 YOLO 进行目标检测和视频分类器在视频源上运行动作识别。

    参数:
        weights (str): YOLO 模型权重路径。
        device (str): 运行模型的设备。使用 'cuda' 表示 NVIDIA GPU，'mps' 表示 Apple Silicon，或 'cpu'。
        source (str): mp4 视频文件路径或 YouTube URL。
        output_path (str, optional): 保存输出视频的路径。
        crop_margin_percentage (int): 检测目标周围添加的边距百分比。
        num_video_sequence_samples (int): 用于分类的视频帧数量。
        skip_frame (int): 检测之间跳过的帧数。
        video_cls_overlap_ratio (float): 视频序列之间的重叠比率。
        fp16 (bool): 是否使用半精度浮点数。
        video_classifier_model (str): 视频分类器模型的名称或路径。
        labels (list[str], optional): 用于零样本分类的标签列表。
    """
    if labels is None:
        labels = [
            "walking",
            "running",
            "brushing teeth",
            "looking into phone",
            "weight lifting",
            "cooking",
            "sitting",
        ]
    # 初始化模型和设备
    device = select_device(device)
    yolo_model = YOLO(weights).to(device)
    if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
        print("'fp16' is not supported for TorchVisionVideoClassifier. Setting fp16 to False.")
        print(
            "'labels' is not used for TorchVisionVideoClassifier. Ignoring the provided labels and using Kinetics-400 labels."
        )
        video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)
    else:
        video_classifier = HuggingFaceVideoClassifier(
            labels, model_name=video_classifier_model, device=device, fp16=fp16
        )

    # 初始化视频捕获
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        source = get_best_youtube_url(source)
    elif not source.endswith(".mp4"):
        raise ValueError("Invalid source. Supported sources are YouTube URLs and MP4 files.")
    cap = cv2.VideoCapture(source)

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 初始化跟踪历史
    track_history = defaultdict(list)
    frame_counter = 0

    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # 运行 YOLO 跟踪
        results = yolo_model.track(frame, persist=True, classes=[0])  # 仅跟踪人员类别

        if results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # 可视化预测结果
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if frame_counter % skip_frame == 0:
                crops_to_infer = []
                track_ids_to_infer = []

            for box, track_id in zip(boxes, track_ids):
                if frame_counter % skip_frame == 0:
                    crop = crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)

                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)

                if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                    start_time = time.time()
                    crops = video_classifier.preprocess_crops_for_video_cls(track_history[track_id])
                    end_time = time.time()
                    preprocess_time = end_time - start_time
                    print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
                    crops_to_infer.append(crops)
                    track_ids_to_infer.append(track_id)

            if crops_to_infer and (
                not pred_labels
                or frame_counter % int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0
            ):
                crops_batch = torch.cat(crops_to_infer, dim=0)

                start_inference_time = time.time()
                output_batch = video_classifier(crops_batch)
                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                print(f"video cls inference time: {inference_time:.4f} seconds")

                pred_labels, pred_confs = video_classifier.postprocess(output_batch)

            if track_ids_to_infer and crops_to_infer:
                for box, track_id, pred_label, pred_conf in zip(boxes, track_ids_to_infer, pred_labels, pred_confs):
                    top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                    label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                    annotator.box_label(box, label_text, color=(0, 0, 255))

        # 将标注帧写入输出视频
        if output_path is not None:
            out.write(frame)

        # 显示标注帧
        cv2.imshow("YOLOv8 Tracking with S3D Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    """解析动作识别流水线的命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="ultralytics detector model path")
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="video file path or youtube URL",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="number of video frames to use for classification"
    )
    parser.add_argument("--skip-frame", type=int, default=2, help="number of frames to skip between detections")
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="overlap ratio between video sequences"
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="video classifier model name"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=["dancing", "singing a song"],
        help="labels for zero-shot video classification",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    """使用解析的命令行参数运行动作识别流水线。"""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
