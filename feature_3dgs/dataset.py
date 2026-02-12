from abc import ABC, abstractmethod
from typing import NamedTuple
import torch
from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset


class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureExtractor':
        return self


# Dynamically create FeatureCamera as a proper NamedTuple
# with all Camera fields + feature_map
FeatureCamera = NamedTuple('FeatureCamera', [
    (name, Camera.__annotations__[name]) for name in Camera._fields
] + [('feature_map', torch.Tensor)])
FeatureCamera.__new__.__defaults__ = (
    *Camera._field_defaults.values(),
    None,  # feature_map default
)


class FeatureCameraDataset(CameraDataset):

    def __init__(self, cameras: CameraDataset, feature_extractor: AbstractFeatureExtractor):
        self.cameras = cameras
        self.feature_extractor = feature_extractor
        self.feature_map_cache = [None] * len(cameras)

    def to(self, device) -> 'FeatureCameraDataset':
        self.cameras.to(device)
        self.feature_extractor.to(device)
        self.feature_map_cache = [(feature_map.to(device) if feature_map is not None else None) for feature_map in self.feature_map_cache]
        return self

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx) -> FeatureCamera:
        camera = self.cameras[idx]
        feature_map = None
        if camera.ground_truth_image is not None:
            if self.feature_map_cache[idx] is None:
                self.feature_map_cache[idx] = self.feature_extractor(camera.ground_truth_image)
            feature_map = self.feature_map_cache[idx]
        return FeatureCamera(*camera, feature_map=feature_map)
