from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from gaussian_splatting import Camera


class AbstractSemanticDecoder(ABC):
    """Feature decoder that bridges encoded Gaussian semantics and extractor
    feature space.  Provides three core operations:

    Terminology
    -----------
    - **C_enc** — channel dimension of the *encoded* (compact) representation
      stored per Gaussian and produced by rasterisation.
    - **C_feat** — channel dimension of the *decoded* feature, which matches
      the ground-truth extractor output (e.g. DINOv2 feature dim).

    Operations
    ----------
    - ``decode_features``: per-point decoding ``(N, C_enc) -> (N, C_feat)``,
      applicable directly to per-Gaussian encoded semantics.
    - ``decode_feature_map``: convert a full rendered feature map
      ``(C_enc, H, W)`` into extractor output format ``(C_feat, H', W')``,
      matching both channel dimension and spatial resolution.  Subclasses
      may override it with reparameterized, memory-efficient implementations
      — e.g. a single Conv2d that fuses per-point mapping and spatial
      downsampling without materialising a large intermediate tensor.
    - ``decode_feature_pixels``: per-pixel projection that applies
      ``decode_features`` and optionally a custom linear layer
      (``weight`` / ``bias``), always preserving the original spatial
      resolution.  Useful for producing full-resolution feature maps with
      arbitrary output dimensions — e.g. a 3-channel PCA visualisation.
    - ``encode_features``: per-point encoding ``(N, C_feat) -> (N, C_enc)``,
      the inverse of ``decode_features``.
    - ``encode_feature_map``: convert an extractor feature map
      ``(C_feat, H', W')`` back to encoded format ``(C_enc, H, W)``,
      the inverse of ``decode_feature_map``.
    - ``encode_feature_pixels``: per-pixel encoding that applies
      ``encode_features`` at full spatial resolution, the inverse of
      ``decode_feature_pixels`` (without weight/bias).
    """

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Per-point decoding: (N, C_enc) -> (N, C_feat)."""
        return features

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Per-point encoding: (N, C_feat) -> (N, C_enc).

        Inverse of ``decode_features``.
        """
        return features

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Convert rendered feature map to extractor output format.

        Default: apply decode_features per pixel (no spatial change).
        Subclasses may override for fused spatial downsampling.

        Args:
            feature_map: (C_enc, H, W) — encoded (rasterised) feature map.

        Returns:
            (C_feat, H', W') matching extractor ground-truth output.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_enc)
        x = self.decode_features(x)                       # (H*W, C_feat)
        return x.reshape(H, W, -1).permute(2, 0, 1)       # (C_feat, H, W), override to change spatial resolution as well

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Convert extractor feature map back to encoded format.

        Default: apply encode_features per pixel (no spatial change).
        Subclasses may override to add spatial upsampling.

        Args:
            feature_map: (C_feat, H', W') — extractor feature map.
            camera: Camera with target spatial dimensions.

        Returns:
            (C_enc, H, W) in the encoded space.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_feat)
        x = self.encode_features(x)                       # (H*W, C_enc)
        return x.reshape(H, W, -1).permute(2, 0, 1)       # (C_enc, H, W)

    def decode_feature_pixels(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Per-pixel feature projection (spatial resolution preserved).

        Applies ``decode_features`` to **every pixel**, then optionally a
        custom linear layer ``F.linear(x, weight, bias)``.

        Args:
            feature_map: (C_enc, H, W) — encoded (rasterised) feature map.
            weight: (C_proj, C_feat) or None.  Skipped when None.
            bias:   (C_proj,) or None.

        Returns:
            (C_feat, H, W) when weight is None,
            (C_proj, H, W) when weight is given.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_enc)
        x = self.decode_features(x)                       # (H*W, C_feat)
        if weight is not None:
            x = F.linear(x, weight, bias)                  # (H*W, C_proj)
        return x.reshape(H, W, -1).permute(2, 0, 1)

    def encode_feature_pixels(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Per-pixel encoding (spatial resolution preserved).

        Applies ``encode_features`` to **every pixel**.

        Args:
            feature_map: (C_feat, H, W) — decoded feature map.

        Returns:
            (C_enc, H, W) in the encoded space.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_feat)
        x = self.encode_features(x)                       # (H*W, C_enc)
        return x.reshape(H, W, -1).permute(2, 0, 1)

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.decode_feature_map(feature_map)

    @abstractmethod
    def to(self, device) -> 'AbstractSemanticDecoder':
        return self

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        return 0
