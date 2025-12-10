from gaussian_splatting.camera import Camera, camera2dict, dict2camera
from gaussian_splatting.dataset import CameraDataset

class FeatureDataset(CameraDataset):
    def __init__(self, basedataset: CameraDataset):
        self.base = basedataset

    def to(self, device) -> 'CameraDataset':
        return self.base.to(device)

    def __len__(self) -> int:
        return self.base.__len__()

    def __getitem__(self, idx) -> Camera:
        camera = self.base.__getitem__(idx)
        # TODO: add feature 
        feature = None
        camera.custom_data["feature"] = feature
        return camera
