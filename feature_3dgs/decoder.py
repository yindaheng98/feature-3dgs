from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from .extractor import FeatureCameraDataset


class AbstractFeatureDecoder(ABC):
    """Feature decoder that bridges encoded Gaussian semantics and extractor
    feature space.  Provides three core operations:

    - ``transform_features``: per-point mapping ``(N, C_in) -> (N, C_out)``,
      applicable directly to per-Gaussian encoded semantics.
    - ``transform_feature_map``: convert a full rendered feature map
      ``(C_in, H, W)`` into extractor output format ``(C_out, H', W')``,
      matching both channel dimension and spatial resolution.  Subclasses
      may override it with reparameterized, memory-efficient implementations
      — e.g. a single Conv2d that fuses per-point mapping and spatial
      downsampling without materialising a large intermediate tensor.
    - ``project_feature_map``: per-pixel projection that applies
      ``transform_features`` and optionally a custom linear layer
      (``weight`` / ``bias``), always preserving the original spatial
      resolution.  Useful for producing full-resolution feature maps with
      arbitrary output dimensions — e.g. a 3-channel PCA visualisation.
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

    def project_feature_map(self, feature_map: torch.Tensor,
                            weight: torch.Tensor = None,
                            bias: torch.Tensor = None) -> torch.Tensor:
        """Per-pixel feature projection (spatial resolution preserved).

        Applies ``transform_features`` to every pixel, then optionally a
        custom linear layer ``F.linear(x, weight, bias)``.

        Args:
            feature_map: (C_in, H, W)
            weight: (C_proj, C_out) or None.  Skipped when None.
            bias:   (C_proj,) or None.

        Returns:
            (C_out, H, W) when weight is None,
            (C_proj, H, W) when weight is given.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_in)
        x = self.transform_features(x)                    # (H*W, C_out)
        if weight is not None:
            x = F.linear(x, weight, bias)                  # (H*W, C_proj)
        return x.reshape(H, W, -1).permute(2, 0, 1)

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
