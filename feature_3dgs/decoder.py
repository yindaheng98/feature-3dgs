from abc import abstractmethod
import torch
from .extractor import FeatureCameraDataset
from .gaussian_model import AbstractFeatureDecoder, SemanticGaussianModel


class AbstractTrainableFeatureDecoder(AbstractFeatureDecoder):
    """Interface for trainable feature decoders that map from extractor feature space to a custom
    feature space.  Provides two more operations:

    - ``init_semantic``: initialise the decoder (e.g. via PCA on extractor features).
    - ``parameters``: return trainable parameters to be optimised by the trainer.
    """

    @staticmethod
    def init_semantic(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset):
        """Build the feature mapping from data (e.g. PCA). Called before training."""
        pass

    @abstractmethod
    def parameters(self):
        return []


class NoopFeatureDecoder(AbstractTrainableFeatureDecoder):
    def __init__(self, embed_dim: int):
        self._embed_dim = embed_dim

    def transform_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map  # Do nothing to the feature map (identity mapping)

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
