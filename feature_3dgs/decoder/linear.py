from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainable import AbstractTrainableFeatureDecoder
from feature_3dgs.utils import pca_inverse_transform_params_to_transform_params
from feature_3dgs.utils.featurefusion import feature_fusion_alpha_avg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature_3dgs.extractor import FeatureCameraDataset
    from feature_3dgs.gaussian_model import SemanticGaussianModel


class LinearDecoder(AbstractTrainableFeatureDecoder):
    """Trainable linear decoder backed by a single ``nn.Linear(C_enc, C_feat)``.

    Provides per-point and per-pixel encode/decode operations, PCA-based
    initialisation, and persistence.  ``decode_feature_map`` and
    ``encode_feature_map`` default to their per-pixel counterparts (no
    spatial resolution change); subclasses may override them to add
    downsampling / upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.linear = nn.Linear(in_channels, out_channels)

    # ------------------------------------------------------------------
    # Per-point operations
    # ------------------------------------------------------------------

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pointwise decoding: (N, C_enc) -> (N, C_feat)."""
        return self.linear(features)

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pointwise encoding via pseudo-inverse: (N, C_feat) -> (N, C_enc)."""
        W_pinv = torch.linalg.pinv(self.linear.weight)     # (C_enc, C_feat)
        return F.linear(features - self.linear.bias, W_pinv)

    # ------------------------------------------------------------------
    # Per-pixel operations (spatial resolution preserved)
    # ------------------------------------------------------------------

    def decode_feature_pixels(
            self, feature_map: torch.Tensor,
            weight: torch.Tensor = None,
            bias: torch.Tensor = None) -> torch.Tensor:
        """Reparameterized per-pixel projection via 1x1 Conv2d.

        When *weight* is given, fuses ``self.linear`` and the custom linear
        into one:  weight_c = weight @ W1,  bias_c = weight @ b1 + bias,
        avoiding the ``(H*W, C_feat)`` intermediate.
        When *weight* is None, applies ``self.linear`` per pixel directly.
        """
        combined_weight = self.linear.weight
        if weight is not None:
            combined_weight = weight @ self.linear.weight         # (C_proj, C_enc)
        combined_bias = F.linear(self.linear.bias, weight, bias)  # (C_proj,)
        return F.conv2d(feature_map.unsqueeze(0), combined_weight[:, :, None, None], combined_bias).squeeze(0)

    def encode_feature_pixels(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Reparameterized per-pixel encoding via 1x1 Conv2d.

        Equivalent to applying ``encode_features`` per pixel but avoids
        permute/reshape overhead by using ``F.conv2d`` with a 1x1 kernel
        derived from the pseudo-inverse of ``self.linear``.
        """
        W_pinv = torch.linalg.pinv(self.linear.weight)       # (C_enc, C_feat)
        b_pinv = -(W_pinv @ self.linear.bias)                 # (C_enc,)
        return F.conv2d(feature_map.unsqueeze(0), W_pinv[:, :, None, None], b_pinv).squeeze(0)

    # ------------------------------------------------------------------
    # Feature-map operations (no spatial resolution change by default)
    # ------------------------------------------------------------------

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Per-pixel decoding: (C_enc, H, W) -> (C_feat, H, W)."""
        return self.decode_feature_pixels(feature_map)

    def encode_feature_map(self, feature_map: torch.Tensor, camera=None) -> torch.Tensor:
        """Per-pixel encoding: (C_feat, H, W) -> (C_enc, H, W)."""
        return self.encode_feature_pixels(feature_map)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def init_semantic(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset):
        """Initialise linear layer weights via PCA on the extractor features.

        Collects all feature vectors from the dataset, computes PCA, and
        sets ``self.linear`` so that it initially performs PCA reconstruction:
          - weight = top-k principal components  (out_channels, in_channels)
          - bias   = feature mean                (out_channels,)
        """
        self: LinearDecoder = gaussians.get_decoder
        weight, bias = dataset.pca_inverse_transform_params(
            n_components=self.linear.in_features, whiten=False)
        with torch.no_grad():
            device = self.linear.weight.device
            self.linear.weight.copy_(weight.to(device))
            self.linear.bias.copy_(bias.to(device))
        weight, bias = pca_inverse_transform_params_to_transform_params(weight, bias)
        fused, _ = feature_fusion_alpha_avg(gaussians, dataset, weight.to(device), bias.to(device))
        # fused, _ = feature_fusion_alpha_max(gaussians, dataset, weight.to(device), bias.to(device))  # worse than avg
        gaussians._encoded_semantics = nn.Parameter(fused.requires_grad_(True))

    # ------------------------------------------------------------------
    # Persistence & utilities
    # ------------------------------------------------------------------

    def to(self, device) -> LinearDecoder:
        self.linear = self.linear.to(device)
        return self

    def load(self, path: str) -> None:
        state_dict = torch.load(path, weights_only=True)
        self.linear.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(self.linear.state_dict(), path)

    def parameters(self):
        return self.linear.parameters()

    @property
    def embed_dim(self) -> int:
        return self.linear.in_features
