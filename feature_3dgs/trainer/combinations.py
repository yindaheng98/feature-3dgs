from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from gaussian_splatting.trainer.depth import DepthTrainerWrapper
from gaussian_splatting.trainer.opacity_reset import OpacityResetTrainerWrapper
from feature_3dgs import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
from feature_3dgs import FeatureCameraDataset, TrainableFeatureCameraDataset

from .base import BaseCameraTrainer, BaseDepthTrainer, BaseDensificationTrainer

# Copy from https://github.com/yindaheng98/gaussian-splatting/blob/a26955eb92af3094e49e0fbe1678ea6c5c86d5ed/gaussian_splatting/trainer/combinations.py#L10-L39

# Camera trainer


def DepthCameraTrainer(model: SemanticGaussianModel, dataset: TrainableFeatureCameraDataset, **configs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableSemanticGaussianModel, dataset: TrainableFeatureCameraDataset, **configs):
    return CameraTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableSemanticGaussianModel, dataset: TrainableFeatureCameraDataset, **configs):
    return CameraTrainerWrapper(DepthOpacityResetDensificationTrainer, model, dataset, **configs)


# Aliases for default trainers
Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
