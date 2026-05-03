# Детекция персонажей Minecraft: FCOS vs YOLOv8

Проект по дообучению и сравнению моделей объектной детекции **FCOS** и **YOLOv8s** на датасете мобов из Minecraft (17 классов).

## Классы

`bee`, `chicken`, `cow`, `creeper`, `enderman`, `fox`, `frog`, `ghast`, `goat`, `llama`, `pig`, `sheep`, `skeleton`, `spider`, `turtle`, `wolf`, `zombie`

## Структура проекта

```
mmdetection/
├── configs/fcos/fcos_minecraft.py    — конфигурация FCOS
├── datasets/minecraft/
│   ├── train/ val/ test/             — изображения + annotations.json
│   ├── annotations/                  — аннотации в формате COCO
│   ├── data.yaml                     — конфигурация для YOLO
│   └── video.mp4                     — видео для инференса
├── checkpoints/                      — предобученные веса
├── artifacts/
│   ├── fcos/                         — логи и веса FCOS
│   ├── yolo/                         — логи и веса YOLO
│   ├── inference/fcos/               — результаты инференса FCOS
│   ├── inference/yolo/               — результаты инференса YOLO
│   ├── videos/                       — видео с детекцией
│   ├── metrics/metrics_comparison.csv
│   └── report.pdf
├── notebook.ipynb                    — основной ноутбук
└── README.md
```

## Запуск

1. Установите зависимости:
   ```bash
   pip install -U openmim
   mim install mmengine "mmcv>=2.0.0" mmdet
   pip install ultralytics
   ```
2. Скачайте датасет и разместите в `datasets/minecraft/`.
3. Откройте `notebook.ipynb` и выполняйте ячейки по порядку.

## Модели

| Модель | Архитектура | Фреймворк |
|--------|-------------|-----------|
| FCOS | ResNet-50 + FPN, anchor-free | MMDetection |
| YOLOv8s | CSPDarknet, one-stage | Ultralytics |

## Результаты

| Модель | mAP | mAP@50 | FPS |
|--------|-----|--------|-----|
| FCOS | — | — | — |
| YOLOv8s | — | — | — |

> Заполните таблицу после обучения и оценки моделей.
