from gaussian_splatting.camera import Camera, camera2dict, dict2camera
from gaussian_splatting.dataset import CameraDataset
import torch

class FeatureDataset(CameraDataset):
    def __init__(self, basedataset: CameraDataset, semantic_feature_path: dict | None = None):
        self.base = basedataset
        assert semantic_feature_path is not None
        self.semantic_feature = torch.load(semantic_feature_path) # TODO: check how the semantic feature is designed

    def to(self, device) -> 'CameraDataset':
        return self.base.to(device)

    def __len__(self) -> int:
        return self.base.__len__()

    def __getitem__(self, idx) -> Camera:
        camera = self.base.__getitem__(idx)
        # TODO: add feature 
        feature = None
        camera.custom_data["feature"] = self.semantic_feature
        return camera
