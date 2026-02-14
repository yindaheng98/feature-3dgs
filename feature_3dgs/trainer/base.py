from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer.camera_trainable import CameraTrainerWrapper
from gaussian_splatting.trainer.depth import DepthTrainer, DepthTrainerWrapper

from .trainer import BaseSemanticTrainer as BaseTrainer


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
