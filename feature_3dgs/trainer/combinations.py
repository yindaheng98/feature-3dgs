from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from gaussian_splatting.trainer.depth import DepthTrainerWrapper
from gaussian_splatting.trainer.opacity_reset import OpacityResetTrainerWrapper

from .densifier import BaseSemanticDensificationTrainer as BaseDensificationTrainer
from .base import BaseCameraTrainer, BaseDepthTrainer

# Copy from https://github.com/yindaheng98/gaussian-splatting/blob/a26955eb92af3094e49e0fbe1678ea6c5c86d5ed/gaussian_splatting/trainer/combinations.py#L10-L39

# Camera trainer


def DepthCameraTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, scene_extent, dataset, *args, **kwargs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, scene_extent, *args, **kwargs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, scene_extent, *args, **kwargs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


# Aliases for default trainers
Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
