from abc import ABC, abstractmethod
import torch
from .extractor import FeatureCameraDataset


class AbstractFeatureDecoder(ABC):
    """Feature decoder that bridges encoded Gaussian semantics and extractor
    feature space.  Provides three core operations:

    - ``init``: build the feature mapping from a dataset (e.g. PCA on
      extractor outputs).  Doubles as a quick visualisation mapping.
    - ``transform_features``: per-point mapping ``(N, C_in) -> (N, C_out)``,
      applicable directly to per-Gaussian encoded semantics.
    - ``transform_feature_map``: convert a full rendered feature map
      ``(C_in, H, W)`` into extractor output format ``(C_out, H', W')``,
      matching both channel dimension and spatial resolution.

    ``transform_feature_map`` has a default implementation that applies
    ``transform_features`` per pixel (no spatial change).  Subclasses may
    override it with fused, memory-efficient implementations — e.g. a single
    Conv2d whose weights are derived from the linear layer, combining the
    per-point mapping and spatial downsampling without materialising a large
    intermediate tensor.
    """

    def init(self, dataset: FeatureCameraDataset):
        """Build the feature mapping from data (e.g. PCA). Called before training."""
        pass

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Per-point feature transform: (N, C_in) -> (N, C_out)."""
        return features

    def transform_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Convert rendered feature map to extractor output format.

        Default: apply transform_features per pixel (no spatial change).
        Subclasses may override for fused spatial downsampling.

        Args:
            feature_map: (C_in, H, W)

        Returns:
            (C_out, H', W') matching extractor output.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_in)
        x = self.transform_features(x)                    # (H*W, C_out)
        return x.reshape(H, W, -1).permute(2, 0, 1)       # (C_out, H, W)

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.transform_feature_map(feature_map)

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

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        return 0


class NoopFeatureDecoder(AbstractFeatureDecoder):
    def __init__(self, embed_dim: int):
        self._embed_dim = embed_dim

    def transform_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map  # Do nothing to the feature map (identity mapping)

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
