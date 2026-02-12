import torch
import torch.nn.functional as F
from ultralytics import YOLO
from feature_3dgs.dataset.abc import AbstractFeatureExtractor


class YOLOExtractor(AbstractFeatureExtractor):
    def __init__(self, version: str):
        self.model = YOLO(version)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # TODO: https://github.com/orgs/ultralytics/discussions/5906
        with torch.no_grad():
            outputs = self.model(image)
            feature_resized = F.interpolate(
                outputs,
                image.shape[-2:],
                mode="bicubic",
                align_corners=True
            )
            return feature_resized

    def to(self, device) -> 'YOLOExtractor':
        self.model.to(device)
        return self
