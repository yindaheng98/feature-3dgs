from abc import ABC, abstractmethod
from typing import Callable, Dict
import torch
from gaussian_splatting import Camera
from feature_3dgs import FeatureGaussian

class AbstractFeatureExtractor(ABC):

    @abstractmethod
    def extract(image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def load(path: str) -> None:
        pass

    @abstractmethod
    def save(path: str) -> None:
        pass

    @property
    @abstractmethod
    def paramters() -> torch.nn.parameter.Parameter:
        pass