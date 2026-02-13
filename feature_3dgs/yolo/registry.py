from typing import Tuple

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import YOLOExtractor
from .decoder import YOLODecoder


# https://github.com/ultralytics/assets/releases
YOLOVERSIONS = [
    # YOLO26 (Ultralytics, Jan 14 2026) :contentReference[oaicite:0]{index=0}
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",

    # YOLO12 (Ultralytics) :contentReference[oaicite:1]{index=1}
    "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",

    # YOLO11 (Ultralytics, Sep 10 2024) :contentReference[oaicite:2]{index=2}
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",

    # YOLOv10 (THU-MIG / Ultralytics integration; includes special "b" balanced) :contentReference[oaicite:3]{index=3}
    "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",

    # YOLOv9 :contentReference[oaicite:4]{index=4}
    "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",

    # YOLOv8 :contentReference[oaicite:5]{index=5}
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",

    # YOLOv5u (Ultralytics modernized variants) :contentReference[oaicite:6]{index=6}
    "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu",
    "yolov5n6u", "yolov5s6u", "yolov5m6u", "yolov5l6u", "yolov5x6u",

    # YOLOv3u (Ultralytics variants) :contentReference[oaicite:7]{index=7}
    "yolov3-tinyu", "yolov3u", "yolov3-sppu",

    # YOLO-NAS (not YOLOv* lineage, but Ultralytics-supported) :contentReference[oaicite:8]{index=8}
    "yolo_nas_s", "yolo_nas_m", "yolo_nas_l",
]

# TODO
