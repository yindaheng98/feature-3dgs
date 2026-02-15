from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper
from gaussian_splatting.trainer.depth import DepthTrainer, DepthTrainerWrapper
from feature_3dgs import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
from feature_3dgs import FeatureCameraDataset, TrainableFeatureCameraDataset

from .trainer import BaseSemanticTrainer as BaseTrainer, SemanticTrainerWrapper
from .densifier import BaseSemanticDensificationTrainer


def BaseCameraTrainer(
        model: CameraTrainableSemanticGaussianModel,
        dataset: TrainableFeatureCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: BaseTrainer(model, dataset, **configs),
        model, dataset, **configs
    )


def BaseDepthTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, dataset, **configs)


def BaseDensificationTrainer(
        model: SemanticGaussianModel,
        dataset: FeatureCameraDataset,
        semantic_lr: float = 0.001,
        semantic_decoder_lr: float = 0.0001,
        semantic_loss_weight: float = 1,
        **configs):
    return SemanticTrainerWrapper(
        lambda model, dataset, **configs: BaseSemanticDensificationTrainer(model, dataset, **configs),
        model, dataset, semantic_lr=semantic_lr, semantic_decoder_lr=semantic_decoder_lr, semantic_loss_weight=semantic_loss_weight,
        **configs
    )
