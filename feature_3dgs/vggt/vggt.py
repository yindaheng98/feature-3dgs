import os
from typing import Tuple

import torch
from vggt.models.vggt import VGGT

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import VGGTExtractor
from .decoder import VGGTLinearAvgDecoder

PATCH_SIZE = 16
VGGT_PATCH_SIZE = 14
FEATURE_DIM = 2048  # 2 * embed_dim (1024)

MODEL_VGGT = "vggt"


def VGGTFeatureExtractor(checkpoint: str = "checkpoints/vggt_1B_commercial.pt") -> VGGTExtractor:
    model = VGGT()
    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    return VGGTExtractor(model=model, patch_size=PATCH_SIZE, vggt_patch_size=VGGT_PATCH_SIZE)


def build_factory():
    def factory(embed_dim: int, checkpoint="checkpoints/vggt_1B_commercial.pt", **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = VGGTFeatureExtractor(checkpoint)
        decoder = VGGTLinearAvgDecoder(
            in_channels=embed_dim,
            out_channels=FEATURE_DIM,
            patch_size=PATCH_SIZE,
            **configs,
        )
        return extractor, decoder
    return factory


register_extractor_decoder(MODEL_VGGT, build_factory())
