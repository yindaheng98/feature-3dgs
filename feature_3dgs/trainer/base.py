from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper
from gaussian_splatting.trainer.depth import DepthTrainer, DepthTrainerWrapper
from feature_3dgs import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
from feature_3dgs import FeatureCameraDataset, TrainableFeatureCameraDataset

from .trainer import BaseSemanticTrainer as BaseTrainer, SemanticTrainerWrapper
from .densifier import BaseSemanticDensificationTrainer


def BaseCameraTrainer(model: CameraTrainableSemanticGaussianModel, dataset: TrainableFeatureCameraDataset, **configs):
    return CameraTrainerWrapper(BaseTrainer, model, dataset, **configs)


def BaseDepthTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, dataset, **configs)


def BaseDensificationTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs):
    return SemanticTrainerWrapper(BaseSemanticDensificationTrainer, model, dataset, **configs)
