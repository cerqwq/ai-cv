# 👁️ AI CV

AI计算机视觉工具，支持图像识别、目标检测、图像生成。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenAI-API-green?logo=openai" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

## ✨ 特性

- 🔍 CV管道设计
- 🎯 YOLO配置生成
- 📝 图像描述生成
- 🎨 分割代码生成
- 📊 数据增强建议
- 📈 模型评估

## 🚀 快速开始

```bash
pip install openai

python tools.py
```

## 📖 使用

```python
from ai_cv import create_tools

tools = create_tools()

# CV管道
pipeline = tools.design_cv_pipeline("人脸检测", "实时，高精度")

# YOLO配置
yolo = tools.generate_yolo_config("COCO", "目标检测")

# 图像描述
caption = tools.generate_image_captioning("一张猫的图片")

# 分割代码
segmentation = tools.generate_segmentation("医学影像", "U-Net")

# 数据增强
augmentation = tools.suggest_augmentation(dataset_info, "分类")

# 模型评估
evaluation = tools.evaluate_model("ResNet50", metrics)
```

## 📁 项目结构

```
ai-cv/
├── tools.py       # CV工具核心
└── README.md
```

## 📄 许可证

MIT License
