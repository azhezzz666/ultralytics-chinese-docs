{% macro param_table(params=None) -%}
| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---- | ------- | ----------- |
{%- set default_params = {
    "show": ["bool", "False", "如果为 `True`，在窗口中显示标注的图像或视频。用于开发或测试期间的即时视觉反馈。"],
    "save": ["bool", "False 或 True", "启用将标注的图像或视频保存到文件。用于文档、进一步分析或分享结果。使用 CLI 时默认为 True，使用 Python 时默认为 False。"],
    "save_frames": ["bool", "False", "处理视频时，将单独的帧保存为图像。用于提取特定帧或进行详细的逐帧分析。"],
    "save_txt": ["bool", "False", "将检测结果保存到文本文件，格式为 `[class] [x_center] [y_center] [width] [height] [confidence]`。用于与其他分析工具集成。"],
    "save_conf": ["bool", "False", "在保存的文本文件中包含置信度分数。增强后处理和分析的可用细节。"],
    "save_crop": ["bool", "False", "保存检测的裁剪图像。用于数据集增强、分析或为特定对象创建专注数据集。"],
    "show_labels": ["bool", "True", "在可视化输出中显示每个检测的标签。提供对检测对象的即时理解。"],
    "show_conf": ["bool", "True", "在标签旁边显示每个检测的置信度分数。提供模型对每个检测确定性的洞察。"],
    "show_boxes": ["bool", "True", "在检测到的对象周围绘制边界框。对于图像或视频帧中对象的视觉识别和定位至关重要。"],
    "line_width": ["None 或 int", "None", "指定边界框的线宽。如果为 `None`，线宽根据图像大小自动调整。提供清晰度的视觉自定义。"],
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
