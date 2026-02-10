from abc import ABC, abstractmethod
import torch


class AbstractDecoder(ABC):

    @abstractmethod
    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device) -> 'AbstractDecoder':
        return self

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
