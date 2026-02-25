import torch
from .trainable import AbstractTrainableFeatureDecoder


class NoopFeatureDecoder(AbstractTrainableFeatureDecoder):
    def __init__(self, embed_dim: int):
        self._embed_dim = embed_dim

    def to(self, device) -> 'AbstractTrainableFeatureDecoder':
        return self

    def load(self, path: str) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def parameters(self):
        return []

    @property
    def embed_dim(self) -> int:
        return self._embed_dim
