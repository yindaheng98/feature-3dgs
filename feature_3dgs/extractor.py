from abc import ABC, abstractmethod
import torch
from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset


class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureExtractor':
        return self


class FeatureCameraDataset(CameraDataset):

    def __init__(self, cameras: CameraDataset, extractor: AbstractFeatureExtractor, cache_device=None):
        self.cameras = cameras
        self.extractor = extractor
        self.feature_map_cache = [None] * len(cameras)
        self.cache_device = cache_device

    def to(self, device) -> 'FeatureCameraDataset':
        self.cameras.to(device)
        self.extractor.to(device)
        return self

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:
        camera = self.cameras[idx]
        feature_map = None
        if camera.ground_truth_image is not None:
            if self.feature_map_cache[idx] is None:
                feature_map = self.extractor(camera.ground_truth_image)
                if self.cache_device is not None:
                    feature_map = feature_map.to(self.cache_device)
                self.feature_map_cache[idx] = feature_map
            feature_map = self.feature_map_cache[idx].to(camera.ground_truth_image.device)
        return camera._replace(custom_data={**camera.custom_data, 'feature_map': feature_map})

    def save_cameras(self, path):
        return self.cameras.save_cameras(path)

    def scene_extent(self):
        return self.cameras.scene_extent()

    @property
    def embed_dim(self) -> int:
        return self[0].custom_data['feature_map'].shape[0]


class TrainableFeatureCameraDataset(FeatureCameraDataset):

    def __init__(self, cameras: TrainableCameraDataset, extractor: AbstractFeatureExtractor, cache_device=None):
        super().__init__(cameras=cameras, extractor=extractor, cache_device=cache_device)
        self.quaternions = cameras.quaternions
        self.Ts = cameras.Ts
        self.exposures = cameras.exposures
