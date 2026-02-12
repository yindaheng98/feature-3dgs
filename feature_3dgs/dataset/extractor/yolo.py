import torch
import torch.nn.functional as F
from ultralytics import YOLO
from gaussian_splatting.dataset import CameraDataset
from feature_3dgs.dataset import AbstractFeatureExtractor, FeatureCameraDataset


class YOLOExtractor(AbstractFeatureExtractor):
    def __init__(self, version: str = "yolov8n.pt"):
        self.model = YOLO(version)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
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


def YOLOFeatureCameraDataset(cameras: CameraDataset, version: str = "yolov8n.pt") -> FeatureCameraDataset:
    return FeatureCameraDataset(cameras, feature_extractor=YOLOExtractor(version))
