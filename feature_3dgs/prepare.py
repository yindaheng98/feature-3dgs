from typing import Tuple

from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.prepare import prepare_dataset
from .extractor import FeatureCameraDataset
from .decoder import AbstractFeatureDecoder
from .registry import build_extractor_decoder
from .gaussian_model import SemanticGaussianModel


def prepare_dataset_and_decoder(
        name: str, source: str, device: str, embed_dim: int,
        trainable_camera: bool = False, load_camera: str = None, load_mask=True, load_depth=True,
        **kwargs
) -> Tuple[FeatureCameraDataset, AbstractFeatureDecoder]:
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
        name=name, embed_dim=embed_dim, **kwargs
    )
    dataset = FeatureCameraDataset(cameras, extractor=extractor).to(device)
    return dataset, decoder


def prepare_gaussians(decoder: AbstractFeatureDecoder, sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None) -> SemanticGaussianModel:
    from .gaussian_model import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
    gaussians = (SemanticGaussianModel if not trainable_camera else CameraTrainableSemanticGaussianModel)(sh_degree, decoder=decoder).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians
