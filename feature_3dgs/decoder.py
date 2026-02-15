from abc import ABC, abstractmethod
import torch
from .extractor import FeatureCameraDataset


class AbstractFeatureDecoder(ABC):

    @abstractmethod
    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureDecoder':
        return self

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def parameters(self):
        return []

    @abstractmethod
    def init(self, dataset: FeatureCameraDataset):
        """Initialise the decoder with a dataset, if necessary. Called once before training starts."""
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        return 0


class NoopFeatureDecoder(AbstractFeatureDecoder):
    def __init__(self, embed_dim: int):
        self._embed_dim = embed_dim

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map

    def to(self, device) -> 'AbstractFeatureDecoder':
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
