from __future__ import annotations

from pathlib import Path
from typing import Tuple

from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.registry import register_extractor_decoder

from .decoder import TTT3RLinearDecoder
from .extractor import TTT3RExtractor
from ._impl.model import ARCroco3DStereo

MODEL_TTT3R = "ttt3r"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def load_ttt3r(
    checkpoint: str = "checkpoints/cut3r_512_dpt_4_64.pth",
    model_update_type: str = "ttt3r",
):
    model = ARCroco3DStereo.from_pretrained(str(_resolve_path(checkpoint)))
    model.config.model_update_type = model_update_type
    model.eval()
    return model


def TTT3RFeatureExtractor(
    checkpoint: str = "checkpoints/cut3r_512_dpt_4_64.pth",
    resize: int = 512,
    reset_interval: int = 1_000_000,
    model_update_type: str = "ttt3r",
    square_ok: bool = False,
) -> TTT3RExtractor:
    model = load_ttt3r(checkpoint=checkpoint, model_update_type=model_update_type)
    return TTT3RExtractor(
        model=model,
        resize=resize,
        reset_interval=reset_interval,
        square_ok=square_ok,
    )


def build_factory():
    def factory(
        embed_dim: int,
        checkpoint: str = "checkpoints/cut3r_512_dpt_4_64.pth",
        resize: int = 512,
        reset_interval: int = 1_000_000,
        model_update_type: str = "ttt3r",
        square_ok: bool = False,
        **configs,
    ) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = TTT3RFeatureExtractor(
            checkpoint=checkpoint,
            resize=resize,
            reset_interval=reset_interval,
            model_update_type=model_update_type,
            square_ok=square_ok,
        )
        decoder = TTT3RLinearDecoder(
            in_channels=embed_dim,
            out_channels=extractor.feature_dim,
            resize=resize,
            patch_size=extractor.patch_size,
            square_ok=square_ok,
            **configs,
        )
        return extractor, decoder

    return factory


register_extractor_decoder(MODEL_TTT3R, build_factory())
