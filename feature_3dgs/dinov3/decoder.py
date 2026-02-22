import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.decoder import NoopFeatureDecoder
from feature_3dgs.extractor import FeatureCameraDataset

from .extractor import padding


class DINOv3LinearAvgDecoder(NoopFeatureDecoder):
    """Decoder that aligns Gaussian features with DINOv3 extractor output.

    Three operations:
      1. **init** - PCA on extractor features to initialise the linear layer.
      2. **transform_features** - ``nn.Linear(C_in, C_out)`` per point.
      3. **transform_feature_map** - reparameterizes the linear mapping
         with patch-level average pooling into a single ``F.conv2d`` call.
         The Conv2d kernel is derived on-the-fly from ``self.linear``
         weights (uniform spatial values = weight / P²), so only the
         linear layer's parameters are trained.  This avoids materialising
         a large ``(C_out, H, W)`` intermediate tensor.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        """
        Args:
            in_channels:  Per-point semantic embedding dimension rendered by
                          the Gaussian rasteriser.
            out_channels: Feature dimension D produced by DINOv3Extractor.
            patch_size:   Patch size used by the paired DINOv3Extractor.
        """
        super().__init__(embed_dim=in_channels)
        self.patch_size = patch_size
        # Step 1: trainable linear mapping  (C_in -> C_out per point)
        self.linear = nn.Linear(in_channels, out_channels)

    # ------------------------------------------------------------------
    # Three core operations
    # ------------------------------------------------------------------

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pointwise linear mapping: (N, C_in) -> (N, C_out)."""
        return self.linear(features)

    def transform_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Fused linear + avg-pool via a single Conv2d.

        Equivalent to (but avoids the large (C_out, H, W) intermediate):

            x = padding(feature_map, P)
            C, H, W = x.shape
            x = x.permute(1, 2, 0).reshape(-1, C)        # (H*W, C_in)
            x = self.linear(x)                             # (H*W, C_out)
            x = x.reshape(H, W, -1).permute(2, 0, 1)      # (C_out, H, W)
            x = F.avg_pool2d(x, kernel_size=P, stride=P)   # (C_out, H_p, W_p)

        Because avg_pool (mean over P² elements) and the linear layer are
        both linear operations, they fuse into one Conv2d with kernel
        ``weight[:, :, None, None] / P²`` and stride P.
        """
        P = self.patch_size
        x = padding(feature_map, P)                        # (C_in, H', W')
        # Derive conv kernel from linear weights
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, P, P) / (P * P)
        return F.conv2d(x.unsqueeze(0), weight, self.linear.bias, stride=P).squeeze(0)

    def transform_feature_map_linear(self, feature_map: torch.Tensor,
                                     weight: torch.Tensor,
                                     bias: torch.Tensor = None) -> torch.Tensor:
        """Reparameterized: merges ``self.linear`` and the custom linear into one.

        Combined linear:  weight_c = weight @ W1,  bias_c = weight @ b1 + bias
        This avoids materialising the ``(H*W, C_out)`` intermediate.
        """
        combined_weight = weight @ self.linear.weight          # (C_custom, C_in)
        combined_bias = F.linear(self.linear.bias, weight, bias)  # (C_custom,)

        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)       # (H*W, C_in)
        x = F.linear(x, combined_weight, combined_bias)        # (H*W, C_custom)
        return x.reshape(H, W, -1).permute(2, 0, 1)            # (C_custom, H, W)

    def init(self, dataset: FeatureCameraDataset):
        """Initialise linear layer weights via PCA on the extractor features.

        Collects all feature vectors from the dataset, computes PCA, and
        sets ``self.linear`` so that it initially performs PCA reconstruction:
          - weight = top-k principal components  (out_channels, in_channels)
          - bias   = feature mean                (out_channels,)
        """
        weight, bias = dataset.pca_inverse_transform_params(
            n_components=self.linear.in_features, whiten=False)
        with torch.no_grad():
            device = self.linear.weight.device
            self.linear.weight.copy_(weight.to(device))
            self.linear.bias.copy_(bias.to(device))

    # ------------------------------------------------------------------
    # Persistence & utilities
    # ------------------------------------------------------------------

    def to(self, device) -> 'DINOv3LinearAvgDecoder':
        self.linear = self.linear.to(device)
        return self

    def load(self, path: str) -> None:
        state_dict = torch.load(path, weights_only=True)
        self.linear.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(self.linear.state_dict(), path)

    def parameters(self):
        return self.linear.parameters()
