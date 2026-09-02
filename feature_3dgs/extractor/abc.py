from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import torch


class AbstractFeatureExtractor(ABC):

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Channel dimension of the extracted semantic features."""
        return 0

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        for image in images:
            yield self(image)

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureExtractor':
        return self
