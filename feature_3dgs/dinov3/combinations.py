from gaussian_splatting.dataset import CameraDataset
from feature_3dgs import FeatureCameraDataset
from .vit import DINOv3ViTExtractor, MODELS as VIT_MODELS, MODEL_DINOV3_VITS
from .convnext import DINOv3ConvNextExtractor, MODELS as CONVNEXT_MODELS, MODEL_DINOV3_CONVNEXTB


def DINOv3ViTFeatureCameraDataset(cameras: CameraDataset, version: str = MODEL_DINOV3_VITS) -> FeatureCameraDataset:
    return FeatureCameraDataset(cameras, extractor=DINOv3ViTExtractor(version))


def DINOv3ConvNextFeatureCameraDataset(cameras: CameraDataset, version: str = MODEL_DINOV3_CONVNEXTB) -> FeatureCameraDataset:
    return FeatureCameraDataset(cameras, extractor=DINOv3ConvNextExtractor(version))


available_datasets = {
    **{version: (lambda cameras, v=version: DINOv3ViTFeatureCameraDataset(cameras, version=v)) for version in VIT_MODELS},
    **{version: (lambda cameras, v=version: DINOv3ConvNextFeatureCameraDataset(cameras, version=v)) for version in CONVNEXT_MODELS},
}
