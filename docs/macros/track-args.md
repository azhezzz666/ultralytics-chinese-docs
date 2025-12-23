{% macro param_table(params=None) -%}
| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---- | ------- | ----------- |
{% set default_params = {
    "source": ["str", "None", "指定图像或视频的源目录。支持文件路径、URL 和视频流。"],
    "persist": ["bool", "False", "启用帧间对象的持久跟踪，在视频序列中保持 ID。"],
    "stream": ["bool", "False", "将输入源视为连续视频流进行实时处理。"],
    "tracker": ["str", "'botsort.yaml'", "指定要使用的跟踪算法，例如 `bytetrack.yaml` 或 `botsort.yaml`。"],
    "conf": ["float", "0.3", "设置检测的置信度阈值；较低的值允许跟踪更多对象，但可能包含误检。"],
    "iou": ["float", "0.5", "设置[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（IoU）阈值，用于过滤重叠检测。"],
    "classes": ["list", "None", "按类别索引过滤结果。例如，`classes=[0, 2, 3]` 只跟踪指定的类别。"],
    "verbose": ["bool", "True", "控制跟踪结果的显示，提供跟踪对象的可视化输出。"],
    "device": ["str", "None", "指定推理设备（例如 `cpu`、`cuda:0` 或 `0`）。允许用户在 CPU、特定 GPU 或其他计算设备之间选择执行模型。"],
    "show": ["bool", "False", "如果为 `True`，在窗口中显示标注的图像或视频以获得即时视觉反馈。"],
    "line_width": ["None 或 int", "None", "指定边界框的线宽。如果为 `None`，线宽根据图像大小自动调整。"]
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
