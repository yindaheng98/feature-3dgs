import os
from typing import Tuple

import torch

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .models.blocks import BasicEncoder
from .extractor import MVTAPExtractor, FEATURE_DIM, STRIDE, MODEL_HEIGHT, MODEL_WIDTH
from .decoder import MVTAPLinearAvgDecoder

MODEL_MVTAP = "mvtap"
DEFAULT_CHECKPOINT = "checkpoints/mvtap.ckpt"
# Lightning checkpoint: state_dict["model.fnet.<BasicEncoder param>"]
FNET_PREFIX = "model.fnet."


def load_basic_encoder(checkpoint: str = DEFAULT_CHECKPOINT) -> BasicEncoder:
    model = BasicEncoder(input_dim=3, output_dim=FEATURE_DIM, stride=STRIDE)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"MVTAP checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fnet_state = {
        k.removeprefix(FNET_PREFIX): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(FNET_PREFIX)
    }
    model.load_state_dict(fnet_state)
    return model


def MVTAPFeatureExtractor(
    checkpoint: str = DEFAULT_CHECKPOINT,
    input_height: int = MODEL_HEIGHT,
    input_width: int = MODEL_WIDTH,
) -> MVTAPExtractor:
    model = load_basic_encoder(checkpoint)
    return MVTAPExtractor(
        model=model,
        input_height=input_height,
        input_width=input_width,
    )


def build_factory():
    def factory(
        encoded_dim: int,
        checkpoint: str = DEFAULT_CHECKPOINT,
        input_height: int = MODEL_HEIGHT,
        input_width: int = MODEL_WIDTH,
        **configs,
    ) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = MVTAPFeatureExtractor(
            checkpoint,
            input_height=input_height,
            input_width=input_width,
        )
        decoder = MVTAPLinearAvgDecoder(
            in_channels=encoded_dim,
            out_channels=extractor.feature_dim,
            input_height=input_height,
            input_width=input_width,
            **configs,
        )
        return extractor, decoder
    return factory


register_extractor_decoder(MODEL_MVTAP, build_factory())
