from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from gaussian_splatting.trainer.depth import DepthTrainerWrapper
from gaussian_splatting.trainer.opacity_reset import OpacityResetTrainerWrapper

from .base import BaseCameraTrainer, BaseDepthTrainer, BaseDensificationTrainer

# Copy from https://github.com/yindaheng98/gaussian-splatting/blob/a26955eb92af3094e49e0fbe1678ea6c5c86d5ed/gaussian_splatting/trainer/combinations.py#L10-L39

# Camera trainer


def DepthCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: BaseOpacityResetDensificationTrainer(model, dataset, **configs),
        model, dataset, **configs
    )


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: DepthOpacityResetDensificationTrainer(model, dataset, **configs),
        model, dataset, **configs
    )


# Aliases for default trainers
Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
