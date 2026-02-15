import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.decoder import NoopFeatureDecoder

from .extractor import padding


class DINOv3LinearAvgDecoder(NoopFeatureDecoder):
    """Decoder that aligns Gaussian features with DINOv3 extractor output.

    Two-stage pipeline:
      1. **transform_features** - a learnable per-point linear mapping
         ``(N, C_in) -> (N, C_out)`` that converts each feature vector
         from the Gaussian embedding space to the DINOv3 feature space.
      2. **postprocess** - spatial downsampling that matches DINOv3's
         patch-level resolution: pad to patch-size multiples, then average
         each non-overlapping patch into a single pixel.
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
    # Two-stage interface
    # ------------------------------------------------------------------

    def transform_features(self, feature: torch.Tensor) -> torch.Tensor:
        """Pointwise linear mapping from Gaussian space to DINOv3 space.

        Args:
            feature: (N, C_in) batch of feature vectors.

        Returns:
            (N, C_out) transformed feature vectors.
        """
        return self.linear(feature)

    def postprocess(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pad to patch-size multiples then average-pool each patch.

        Args:
            feature_map: (C, H, W) tensor (already in DINOv3 feature space).

        Returns:
            (C, H_patches, W_patches) tensor matching DINOv3Extractor's
            spatial output exactly.
        """
        x = padding(feature_map, self.patch_size)          # (C, H', W')
        x = F.avg_pool2d(
            x.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(0)                                        # (C, H_p, W_p)
        return x

    # ------------------------------------------------------------------
    # AbstractDecoder interface
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
