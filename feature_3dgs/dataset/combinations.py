from gaussian_splatting.dataset import CameraDataset
from .abc import FeatureCameraDataset
from .yolo import available_datasets as available_yolo_datasets
from .dinov3 import available_datasets as available_dinov3_datasets


available_datasets = {
    **available_yolo_datasets,
    **available_dinov3_datasets,
}


def build_dataset(name: str, cameras: CameraDataset, *args, **kwargs) -> FeatureCameraDataset:
    return available_datasets[name](cameras, *args, **kwargs)
