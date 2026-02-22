from typing import Tuple

from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.prepare import prepare_dataset
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from .extractor import FeatureCameraDataset, TrainableFeatureCameraDataset
from .decoder import AbstractTrainableFeatureDecoder
from .registry import build_extractor_decoder
from .gaussian_model import SemanticGaussianModel
from .trainer import (
    Trainer,
    OpacityResetDensificationTrainer,
    CameraTrainer,
    OpacityResetDensificationCameraTrainer,
)


def prepare_dataset_and_decoder(
        name: str, source: str, embed_dim: int, device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_camera: str = None, load_mask=True, load_depth=True,
        configs={},
) -> Tuple[FeatureCameraDataset, AbstractTrainableFeatureDecoder]:
    """Prepare a FeatureCameraDataset and its corresponding decoder.

    This is a convenience function that chains together camera loading,
    extractor/decoder construction, and dataset creation.
    """
    cameras = prepare_dataset(
        source=source, device=device,
        trainable_camera=trainable_camera, load_camera=load_camera,
        load_mask=load_mask, load_depth=load_depth,
    )
    extractor, decoder = build_extractor_decoder(
        name=name, embed_dim=embed_dim, **configs
    )
    dataset = (FeatureCameraDataset if not trainable_camera else TrainableFeatureCameraDataset)(cameras=cameras, extractor=extractor, cache_device=dataset_cache_device).to(device)
    return dataset, decoder


def prepare_gaussians(
        decoder: AbstractTrainableFeatureDecoder, sh_degree: int,
        source: str, dataset: FeatureCameraDataset, device: str,
        trainable_camera: bool = False, load_ply: str = None, load_semantic: bool = True,
) -> SemanticGaussianModel:
    from .gaussian_model import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
    gaussians = (SemanticGaussianModel if not trainable_camera else CameraTrainableSemanticGaussianModel)(sh_degree, decoder=decoder).to(device)
    gaussians.load_ply(load_ply, load_semantic=load_semantic) if load_ply else colmap_init(gaussians, source)
    if not load_ply or not load_semantic:
        decoder.init_semantic(gaussians, dataset)
    return gaussians


modes = {
    "base": Trainer,
    "densify": OpacityResetDensificationTrainer,
    "camera": CameraTrainer,
    "camera-densify": OpacityResetDensificationCameraTrainer,
}


def prepare_trainer(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset, mode: str, trainable_camera: bool = False, with_scale_reg=False, configs={}) -> AbstractTrainer:
    # Copy from https://github.com/yindaheng98/gaussian-splatting/blob/master/gaussian_splatting/prepare.py#L74-L90
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda model, dataset, **configs: ScaleRegularizeTrainerWrapper(modes[mode], model, dataset, **configs)
    trainer = constructor(
        gaussians,
        dataset=dataset,
        **configs
    )
    return trainer
