from abc import ABC, abstractmethod
import torch


class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureExtractor':
        return self
