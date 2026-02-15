from abc import ABC, abstractmethod
import torch
from .extractor import FeatureCameraDataset


class AbstractFeatureDecoder(ABC):

    @abstractmethod
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pointwise feature transformation (one feature in, one feature out).

        Maps each feature vector independently:
          (N, C_in) -> (N, C_out)

        This is trainable and can be used both on rendered feature maps
        (via __call__) and directly on per-Gaussian features in
        SemanticGaussianModel.
        """
        return features

    @abstractmethod
    def postprocess(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Spatial post-processing on a transformed feature map.

        Handles resolution mismatch between rendered output and extractor
        output (e.g. downscaling to patch-level resolution).

        Args:
            feature_map: (C, H, W) tensor after transform_feature.

        Returns:
            (C, H', W') tensor matching extractor output spatial size.
        """
        return feature_map

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Full decode pipeline on a feature map (C_in, H, W).

        1. Reshape (C_in, H, W) -> (H*W, C_in), apply transform_feature,
           then reshape back to (C_out, H, W).
        2. Apply postprocess for spatial downsampling.
        """
        C, H, W = feature_map.shape
        # (C_in, H, W) -> (H*W, C_in)
        x = feature_map.permute(1, 2, 0).reshape(-1, C)
        # (H*W, C_in) -> (H*W, C_out)
        x = self.transform_features(x)
        # (H*W, C_out) -> (C_out, H, W)
        x = x.reshape(H, W, -1).permute(2, 0, 1)
        return self.postprocess(x)

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

    def transform_features(self, feature: torch.Tensor) -> torch.Tensor:
        return feature

    def postprocess(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map

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

    def init(self, dataset: FeatureCameraDataset):
        pass

    @property
    def embed_dim(self) -> int:
        return self._embed_dim
