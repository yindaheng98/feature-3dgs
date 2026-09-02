from typing import Tuple

from vggttt.nets.vggt.models.vggt import VGGT

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder
from feature_3dgs.vggt.vggt import (
    FEATURE_DIM,
    FEAT_SIZE,
    PATCH_SIZE,
    TRACK_FEATURE_DIM,
    TRACK_FEAT_SIZE,
    TRACK_PATCH_SIZE,
    load_vggt,
)

from .extractor import VGGTTTExtractor
from .track import VGGTTTTrackExtractor
from .decoder import VGGTTTLinearAvgDecoder

MODEL_VGGTTT = "vggttt"
MODEL_VGGTTTTRACK = "vggttttrack"

DEFAULT_CHECKPOINT = "nvidia/vgg-ttt"
DEFAULT_TRACK_CHECKPOINT = "checkpoints/vggt_1B_commercial.pt"


def load_vggttt(checkpoint: str = DEFAULT_CHECKPOINT, track_checkpoint: str | None = None) -> VGGT:
    model = VGGT.from_pretrained(checkpoint)
    if track_checkpoint is not None and model.track_head is None:
        track_model = load_vggt(track_checkpoint)
        model.track_head = track_model.track_head
    return model


def VGGTTTFeatureExtractor(checkpoint: str = DEFAULT_CHECKPOINT, img_load_resolution: int = 1024) -> VGGTTTExtractor:
    model = load_vggttt(checkpoint)
    return VGGTTTExtractor(
        model=model,
        feature_dim=FEATURE_DIM,
        img_load_resolution=img_load_resolution,
    )


def VGGTTTTrackFeatureExtractor(checkpoint: str = DEFAULT_CHECKPOINT, track_checkpoint: str = DEFAULT_TRACK_CHECKPOINT, img_load_resolution: int = 1024) -> VGGTTTTrackExtractor:
    model = load_vggttt(checkpoint, track_checkpoint=track_checkpoint)
    return VGGTTTTrackExtractor(
        model=model,
        feature_dim=TRACK_FEATURE_DIM,
        img_load_resolution=img_load_resolution,
    )


def build_factory():
    def factory(encoded_dim: int, checkpoint=DEFAULT_CHECKPOINT, img_load_resolution: int = 1024, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = VGGTTTFeatureExtractor(checkpoint, img_load_resolution=img_load_resolution)
        decoder = VGGTTTLinearAvgDecoder(
            in_channels=encoded_dim,
            out_channels=extractor.feature_dim,
            feat_size=FEAT_SIZE,
            kernel_size=PATCH_SIZE,
            **configs,
        )
        return extractor, decoder
    return factory


def build_track_factory():
    def factory(encoded_dim: int, checkpoint=DEFAULT_CHECKPOINT, track_checkpoint=DEFAULT_TRACK_CHECKPOINT, img_load_resolution: int = 1024, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = VGGTTTTrackFeatureExtractor(checkpoint, track_checkpoint=track_checkpoint, img_load_resolution=img_load_resolution)
        decoder = VGGTTTLinearAvgDecoder(
            in_channels=encoded_dim,
            out_channels=extractor.feature_dim,
            feat_size=TRACK_FEAT_SIZE,
            kernel_size=TRACK_PATCH_SIZE,
            **configs,
        )
        return extractor, decoder
    return factory


register_extractor_decoder(MODEL_VGGTTT, build_factory())
register_extractor_decoder(MODEL_VGGTTTTRACK, build_track_factory())
