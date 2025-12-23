{% macro param_table(params=None) -%}
| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---- | ------- | ----------- |
{% set default_params = {
    "model": ["str", "None", "Ultralytics YOLO 模型文件的路径。"],
    "region": ["list", "'[(20, 400), (1260, 400)]'", "定义计数区域的点列表。"],
    "show_in": ["bool", "True", "控制是否在视频流上显示进入计数的标志。"],
    "show_out": ["bool", "True", "控制是否在视频流上显示离开计数的标志。"],
    "analytics_type": ["str", "'line'", "图表类型，即 `line`、`bar`、`area` 或 `pie`。"],
    "colormap": ["int", "cv2.COLORMAP_JET", "用于热力图的颜色映射。"],
    "json_file": ["str", "None", "包含所有停车坐标数据的 JSON 文件路径。"],
    "up_angle": ["float", "145.0", "'向上'姿势的角度阈值。"],
    "kpts": ["list[int]", "'[6, 8, 10]'", "用于监控锻炼的三个关键点索引列表。这些关键点对应身体关节或部位，如肩膀、肘部和手腕，用于俯卧撑、引体向上、深蹲、腹部锻炼等。"],
    "down_angle": ["float", "90.0", "'向下'姿势的角度阈值。"],
    "blur_ratio": ["float", "0.5", "调整模糊强度的百分比，值范围为 `0.1 - 1.0`。"],
    "crop_dir": ["str", "'cropped-detections'", "存储裁剪检测的目录名称。"],
    "records": ["int", "5", "触发安全警报系统发送邮件的总检测计数。"],
    "vision_point": ["tuple[int, int]", "(20, 20)", "VisionEye 解决方案跟踪对象和绘制路径的点。"],
    "source": ["str", "None", "输入源的路径（视频、RTSP 等）。仅可与解决方案命令行界面（CLI）一起使用。"],
    "figsize": ["tuple[int, int]", "(12.8, 7.2)", "分析图表（如热力图或图形）的图形尺寸。"],
    "fps": ["float", "30.0", "用于速度计算的每秒帧数。"],
    "max_hist": ["int", "5", "每个对象跟踪的最大历史点数，用于速度/方向计算。"],
    "meter_per_pixel": ["float", "0.05", "用于将像素距离转换为真实世界单位的缩放因子。"],
    "max_speed": ["int", "120", "视觉叠加中的最大速度限制（用于警报）。"],
    "data": ["str", "'images'", "用于相似性搜索的图像目录路径。"],
} %}
{% if not params %}
{% for param, details in default_params.items() %}
| `{{ param }}` | `{{ details[0] }}` | `{{ details[1] }}` | {{ details[2] }} |
{% endfor %}
{% else %}
{% for param in params %}
{% if param in default_params %}
| `{{ param }}` | `{{ default_params[param][0] }}` | `{{ default_params[param][1] }}` | {{ default_params[param][2] }} |
{% endif %}
{% endfor %}
{% endif %}
{%- endmacro -%}
