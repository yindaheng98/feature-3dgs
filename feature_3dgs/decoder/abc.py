from abc import ABC, abstractmethod
import torch

class AbstractDecoder(ABC):

    @abstractmethod
    def load_checkpoint(path: str) -> None:
        pass

    @abstractmethod
    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        pass

    def save(self, save_path: str):
        pass
