import os
from typing import Tuple

import torch
from vggt.models.vggt import VGGT

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import VGGTExtractor, FEAT_SIZE, PATCH_SIZE
from .track import VGGTrackExtractor, FEAT_SIZE as TRACK_FEAT_SIZE, PATCH_SIZE as TRACK_PATCH_SIZE
from .decoder import VGGTLinearAvgDecoder

FEATURE_DIM = 2048              # 2 * embed_dim (1024)
TRACK_FEATURE_DIM = 128         # TrackHead DPT features

MODEL_VGGT = "vggt"
MODEL_VGGTRACK = "vggtrack"


def load_vggt(checkpoint: str = "checkpoints/vggt_1B_commercial.pt") -> VGGT:
    if os.path.isfile(checkpoint):
        model = VGGT()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    return model


def VGGTFeatureExtractor(checkpoint: str = "checkpoints/vggt_1B_commercial.pt", img_load_resolution: int = 1024) -> VGGTExtractor:
    model = load_vggt(checkpoint)
    return VGGTExtractor(model=model, img_load_resolution=img_load_resolution)


def VGGTrackFeatureExtractor(checkpoint: str = "checkpoints/vggt_1B_commercial.pt", img_load_resolution: int = 1024) -> VGGTrackExtractor:
    model = load_vggt(checkpoint)
    return VGGTrackExtractor(model=model, img_load_resolution=img_load_resolution)


def build_factory():
    def factory(embed_dim: int, checkpoint="checkpoints/vggt_1B_commercial.pt", img_load_resolution: int = 1024, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = VGGTFeatureExtractor(checkpoint, img_load_resolution=img_load_resolution)
        decoder = VGGTLinearAvgDecoder(
            in_channels=embed_dim,
            out_channels=FEATURE_DIM,
            feat_size=FEAT_SIZE,
            kernel_size=PATCH_SIZE,
            **configs,
        )
        return extractor, decoder
    return factory


def build_track_factory():
    def factory(embed_dim: int, checkpoint="checkpoints/vggt_1B_commercial.pt", img_load_resolution: int = 1024, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = VGGTrackFeatureExtractor(checkpoint, img_load_resolution=img_load_resolution)
        decoder = VGGTLinearAvgDecoder(
            in_channels=embed_dim,
            out_channels=TRACK_FEATURE_DIM,
            feat_size=TRACK_FEAT_SIZE,
            kernel_size=TRACK_PATCH_SIZE,
            **configs,
        )
        return extractor, decoder
    return factory


register_extractor_decoder(MODEL_VGGT, build_factory())
register_extractor_decoder(MODEL_VGGTRACK, build_track_factory())
