import torch
from .abc import AbstractFeatureExtractor
import torch.nn.functional as F
from ultralytics import YOLO

class MLPExtractor(AbstractFeatureExtractor):
    def __init__(self, version: str = "yolov8n.pt"):
        self.model = YOLO(version)

    def extract(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(image)
            feature_resized = F.interpolate(
                outputs,
                image.shape[-2 : ],
                mode = "bicubic",
                align_corners = True
            )
            return feature_resized