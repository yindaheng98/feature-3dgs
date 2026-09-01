from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainable import AbstractTrainableDecoder
from feature_3dgs.utils import pca_inverse_transform_params
from feature_3dgs.utils.featurefusion import feature_fusion_alpha_avg, feature_fusion_alpha_max
from feature_3dgs.utils.featurepickup import feature_pickup_alpha_max

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature_3dgs.extractor import FeatureCameraDataset
    from feature_3dgs.gaussian_model import SemanticGaussianModel


class LinearDecoder(AbstractTrainableDecoder):
    """Trainable linear decoder backed by a single ``nn.Linear(C_enc, C_feat)``.

    Provides per-point and per-pixel encode/decode operations, PCA-based
    initialisation, and persistence.  ``decode_feature_map`` and
    ``encode_feature_map`` default to their per-pixel counterparts (no
    spatial resolution change); subclasses may override them to add
    downsampling / upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, init_method="fusion avg"):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        assert init_method in ["pickup max", "fusion avg", "fusion max"], f"Unsupported init method {init_method}"
        self.init_method = init_method

    # ------------------------------------------------------------------
    # Per-point operations
    # ------------------------------------------------------------------

    def decode_features(self, features: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Pointwise decoding: (N, C_enc) -> (N, C_feat).

        When *weight* is given, fuses ``self.linear`` and the custom linear
        into one:  weight_c = weight @ W1,  bias_c = weight @ b1 + bias.
        """
        if weight is None:
            return self.linear(features)
        combined_weight = weight @ self.linear.weight         # (C_proj, C_enc)
        combined_bias = F.linear(self.linear.bias, weight, bias)
        return F.linear(features, combined_weight, combined_bias)

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
        combined_bias = self.linear.bias
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

    def similarity_encoded(self, encoded_query: torch.Tensor, encoded_features: torch.Tensor) -> torch.Tensor:
        """``similarity(decode(encoded_query), decode(encoded_features))`` via the Gram matrix.

        Equivalent to ``cos(Wq + b, Ws + b)`` without materialising ``C_feat``.
        Leading dims broadcast.
        """
        W, b = self.linear.weight, self.linear.bias
        G = W.T @ W
        c = W.T @ b
        bb = b @ b
        qG = encoded_query @ G
        qc = encoded_query @ c
        sc = encoded_features @ c
        num = (qG * encoded_features).sum(-1) + qc + sc + bb
        den_q = ((qG * encoded_query).sum(-1) + 2 * qc + bb).clamp_min(0).sqrt().clamp_min(1e-8)
        den_s = ((encoded_features @ G * encoded_features).sum(-1) + 2 * sc + bb).clamp_min(0).sqrt().clamp_min(1e-8)
        return num / (den_q * den_s)

    def similarity_encoded_query(self, encoded_query: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """``similarity(decode(encoded_query), features)`` without materialising decoded query.

        Leading dims broadcast.  ``(Wq + b) · f = q · (W^T f) + b · f``.
        """
        W, b = self.linear.weight, self.linear.bias
        G = W.T @ W
        c = W.T @ b
        bb = b @ b
        num = (encoded_query * F.linear(features, W.T)).sum(-1) + (features * b).sum(-1)
        den_q = ((encoded_query @ G * encoded_query).sum(-1) + 2 * (encoded_query @ c) + bb).clamp_min(0).sqrt().clamp_min(1e-8)
        den_f = features.norm(dim=-1).clamp_min(1e-8)
        return num / (den_q * den_f)

    def similarity_encoded_features(self, query: torch.Tensor, encoded_features: torch.Tensor) -> torch.Tensor:
        """``similarity(query, decode(encoded_features))`` without materialising decoded features.

        Leading dims broadcast.  ``(Ws + b) · q = s · (W^T q) + b · q``.
        """
        W, b = self.linear.weight, self.linear.bias
        G = W.T @ W
        c = W.T @ b
        bb = b @ b
        num = (encoded_features * F.linear(query, W.T)).sum(-1) + (query * b).sum(-1)
        den_q = query.norm(dim=-1).clamp_min(1e-8)
        den_s = ((encoded_features @ G * encoded_features).sum(-1) + 2 * (encoded_features @ c) + bb).clamp_min(0).sqrt().clamp_min(1e-8)
        return num / (den_q * den_s)

    # ------------------------------------------------------------------
    # Feature-map operations (no spatial resolution change by default)
    # ------------------------------------------------------------------

    def decode_feature_map(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Per-pixel decoding: (C_enc, H, W) -> (C_feat, H, W)."""
        return self.decode_feature_pixels(feature_map, weight=weight, bias=bias)

    def encode_feature_map(self, feature_map: torch.Tensor, camera=None) -> torch.Tensor:
        """Per-pixel encoding: (C_feat, H, W) -> (C_enc, H, W)."""
        return self.encode_feature_pixels(feature_map)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def init_semantic(
            gaussians: SemanticGaussianModel,
            dataset: FeatureCameraDataset,
            decoder: LinearDecoder | None = None):
        """Initialise semantics from PCA or a preloaded linear decoder.

        When *decoder* is None, collects all feature vectors from the
        dataset, computes PCA, and sets ``self.linear`` so that it initially
        performs PCA reconstruction:
          - weight = top-k principal components  (out_channels, in_channels)
          - bias   = feature mean                (out_channels,)

        When *decoder* is provided, copies its linear weights before
        computing the fused encoded semantics.
        """
        self: LinearDecoder = gaussians.get_decoder
        if decoder is None:
            weight, bias = pca_inverse_transform_params(
                dataset, n_components=self.linear.in_features, whiten=False,
                cache_device=dataset.cache_device)
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)
        else:
            if not isinstance(decoder, LinearDecoder):
                raise TypeError(f"Expected LinearDecoder, got {type(decoder)!r}")
            if decoder is not self:
                with torch.no_grad():
                    self.linear.weight.copy_(decoder.linear.weight)
                    self.linear.bias.copy_(decoder.linear.bias)
        if self.init_method == "pickup max":
            fused, _ = feature_pickup_alpha_max(gaussians, dataset, self.encode_feature_pixels)
        elif self.init_method == "fusion avg":
            fused, _ = feature_fusion_alpha_avg(gaussians, dataset, self.encode_feature_map)
        elif self.init_method == "fusion max":
            fused, _ = feature_fusion_alpha_max(gaussians, dataset, self.encode_feature_map)  # worse than avg
        gaussians._encoded_semantics = nn.Parameter(fused.requires_grad_(True))

    @property
    def embed_dim(self) -> int:
        return self.linear.in_features
