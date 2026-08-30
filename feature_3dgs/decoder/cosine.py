from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import LinearDecoder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature_3dgs.extractor import FeatureCameraDataset
    from feature_3dgs.gaussian_model import SemanticGaussianModel


class CosineLinearDecoder(LinearDecoder):
    """``LinearDecoder`` adapted to cosine-similarity training.

    Differences from :class:`LinearDecoder`:

    - **PCA init**: L2-normalise features, uncentered SVD, ``bias = 0``.
      The mean is not a reconstruction offset under cosine and would dominate
      the angle.
    - **bias**: kept at zero and frozen so decode stays a linear subspace
      through the origin (``decode(s) = W s``).
    - **encode**: unit-normalise then apply the adjoint ``W^T``, not
      ``pinv(W)``.  With ``bias = 0``,
      ``⟨decode(s), f̂⟩ = ⟨s, encode(f)⟩`` where ``f̂ = f / ‖f‖``.
      At cosine-PCA init ``W`` is column-orthonormal, so ``W^T = pinv(W)``
      and ``cos(decode(s), f) = ⟨s, encode(f)⟩ / ‖s‖``.
    """

    def __init__(self, in_channels: int, out_channels: int, init_method="fusion avg"):
        super().__init__(in_channels, out_channels, init_method)
        nn.init.zeros_(self.linear.bias)
        self.linear.bias.requires_grad_(False)

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pointwise adjoint encoding: (N, C_feat) -> (N, C_enc).

        Unit-normalises *features* then applies ``W^T`` so that
        ``decode(s) · f == s · encode(f)`` when ``‖f‖ = 1`` and ``bias = 0``.
        """
        return F.linear(F.normalize(features, dim=-1), self.linear.weight.T)

    def encode_feature_pixels(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Per-pixel adjoint encoding via 1x1 Conv2d: (C_feat, H, W) -> (C_enc, H, W)."""
        W_adj = self.linear.weight.T  # (C_enc, C_feat)
        feature_map = F.normalize(feature_map, dim=0)
        return F.conv2d(feature_map.unsqueeze(0), W_adj[:, :, None, None]).squeeze(0)

    @staticmethod
    def init_semantic(
            gaussians: SemanticGaussianModel,
            dataset: FeatureCameraDataset,
            decoder: LinearDecoder | None = None):
        """Initialise semantics from cosine PCA or a preloaded linear decoder.

        When *decoder* is None, sets ``self.linear`` to the uncentered SVD of
        unit-normalised extractor features (``bias`` stays zero).

        When *decoder* is provided, copies its linear **weight** only; bias is
        always reset to zero so a reconstruction decoder cannot reintroduce
        the feature mean.
        """
        self: CosineLinearDecoder = gaussians.get_decoder
        if decoder is None:
            weight, bias = dataset.cosine_pca_inverse_transform_params(
                n_components=self.linear.in_features)
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)
        else:
            if not isinstance(decoder, LinearDecoder):
                raise TypeError(f"Expected LinearDecoder, got {type(decoder)!r}")
            if decoder is not self:
                with torch.no_grad():
                    self.linear.weight.copy_(decoder.linear.weight)
            with torch.no_grad():
                self.linear.bias.zero_()
        # Reuse fusion / pickup; pass self so reconstruction PCA and bias copy are skipped.
        LinearDecoder.init_semantic(gaussians, dataset, decoder=self)
