from gaussian_splatting.dataset import CameraDataset
from feature_3dgs.dataset.abc import FeatureCameraDataset
from .extractor import DINOv3Extractor, MODELS, MODEL_DINOV3_VITS


def DINOv3FeatureCameraDataset(cameras: CameraDataset, version: str = MODEL_DINOV3_VITS) -> FeatureCameraDataset:
    return FeatureCameraDataset(cameras, feature_extractor=DINOv3Extractor(version))


available_datasets = {
    version: (lambda cameras, v=version: DINOv3FeatureCameraDataset(cameras, version=v)) for version in MODELS
}
