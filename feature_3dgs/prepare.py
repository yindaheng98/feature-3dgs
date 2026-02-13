from typing import Tuple

from gaussian_splatting.prepare import prepare_dataset
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.decoder import AbstractFeatureDecoder
from feature_3dgs.registry import build_extractor_decoder


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
