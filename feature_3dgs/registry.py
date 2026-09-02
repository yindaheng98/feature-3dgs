from typing import Tuple, Protocol

from .extractor import AbstractFeatureExtractor
from .decoder import AbstractTrainableDecoder


class ExtractorDecoderFactory(Protocol):
    def __call__(self, encoded_dim: int, *args: object, **kwargs: object) -> tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]: ...


REGISTRY: dict[str, ExtractorDecoderFactory] = {}


def register_extractor_decoder(name: str, factory: ExtractorDecoderFactory) -> None:
    """Register an (Extractor, Decoder) factory under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Extractor-Decoder combination '{name}' is already registered.")
    REGISTRY[name] = factory


def get_available_extractor_decoders() -> list[str]:
    """Return the names of all registered extractor-decoder combinations."""
    return list(REGISTRY.keys())


def build_extractor_decoder(name: str, encoded_dim: int, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
    """Build an (Extractor, Decoder) pair by name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Extractor-Decoder combination '{name}' not found. "
            f"Available: {get_available_extractor_decoders()}"
        )
    extractor, decoder = REGISTRY[name](encoded_dim, **configs)
    if extractor.feature_dim != decoder.semantic_dim:
        raise ValueError(
            f"Extractor feature_dim ({extractor.feature_dim}) does not match "
            f"decoder semantic_dim ({decoder.semantic_dim})."
        )
    return extractor, decoder
