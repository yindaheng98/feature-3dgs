from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper
from gaussian_splatting.trainer.depth import DepthTrainer, DepthTrainerWrapper

from .trainer import BaseSemanticTrainer as BaseTrainer, SemanticTrainerWrapper
from .densifier import BaseSemanticDensificationTrainer


def BaseCameraTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def BaseDepthTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)


def BaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        semantic_lr: float = 0.001,
        semantic_decoder_lr: float = 0.0001,
        semantic_loss_weight: float = 1,
        *args, **kwargs) -> BaseSemanticDensificationTrainer:
    return SemanticTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseSemanticDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, semantic_lr=semantic_lr, semantic_decoder_lr=semantic_decoder_lr, semantic_loss_weight=semantic_loss_weight,
    )
