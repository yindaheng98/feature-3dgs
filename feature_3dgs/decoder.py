from abc import ABC, abstractmethod
import torch


class AbstractDecoder(ABC):

    @abstractmethod
    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map

    @abstractmethod
    def to(self, device) -> 'AbstractDecoder':
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

    @property
    @abstractmethod
    def input_dim(self) -> int:
        return 0


class NoopDecoder(AbstractDecoder):
    def __init__(self, input_dim: int):
        self._input_dim = input_dim

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map

    def to(self, device) -> 'AbstractDecoder':
        return self

    def load(self, path: str) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def parameters(self):
        return []

    @property
    def input_dim(self) -> int:
        return self._input_dim
