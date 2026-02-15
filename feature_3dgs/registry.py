from typing import Tuple, Protocol

from .extractor import AbstractFeatureExtractor
from .decoder import AbstractFeatureDecoder


class ExtractorDecoderFactory(Protocol):
    def __call__(self, embed_dim: int, *args: object, **kwargs: object) -> tuple[AbstractFeatureExtractor, AbstractFeatureDecoder]: ...


REGISTRY: dict[str, ExtractorDecoderFactory] = {}


def register_extractor_decoder(name: str, factory: ExtractorDecoderFactory) -> None:
    """Register an (Extractor, Decoder) factory under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Extractor-Decoder combination '{name}' is already registered.")
    REGISTRY[name] = factory


def get_available_extractor_decoders() -> list[str]:
    """Return the names of all registered extractor-decoder combinations."""
    return list(REGISTRY.keys())


def build_extractor_decoder(name: str, embed_dim: int, *args, **configs) -> Tuple[AbstractFeatureExtractor, AbstractFeatureDecoder]:
    """Build an (Extractor, Decoder) pair by name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Extractor-Decoder combination '{name}' not found. "
            f"Available: {get_available_extractor_decoders()}"
        )
    return REGISTRY[name](embed_dim, *args, **configs)
