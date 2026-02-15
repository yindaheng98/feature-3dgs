import torch
import torch.nn as nn

from feature_3dgs.decoder import NoopFeatureDecoder

from .extractor import padding


class DINOv3CNNDecoder(NoopFeatureDecoder):
    """CNN-based decoder that mirrors DINOv3Extractor's spatial transformation.

    Takes a feature map with the same spatial size as the input image to
    DINOv3Extractor, and produces output with matching spatial dimensions
    and feature channels as DINOv3Extractor's output.

    The decoder performs:
      1. Padding to patch-size multiples (identical to DINOv3Extractor).
      2. A learnable Conv2d with kernel_size=patch_size and stride=patch_size,
         which maps each non-overlapping patch to a single output pixel —
         the learnable analog of ViT patch embedding.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        """
        Args:
            in_channels:  Number of input feature channels (customisable,
                          equals the per-point semantic embedding dimension
                          rendered by the Gaussian rasteriser).
            out_channels: Number of output feature channels.  Must equal the
                          feature dimension D produced by DINOv3Extractor so
                          that the two can be compared directly.
            patch_size:   Patch size used by the paired DINOv3Extractor.
        """
        super().__init__(embed_dim=in_channels)
        self.patch_size = patch_size
        # Core learnable layer: map each (in_channels, patch_size, patch_size)
        # patch to a single (out_channels,) pixel.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=patch_size, stride=patch_size),
        )

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Decode a rasterised feature map to match DINOv3Extractor output.

        Args:
            feature_map: (C_in, H, W) tensor — same spatial size as the
                         image fed to DINOv3Extractor.

        Returns:
            (D, H_patches, W_patches) tensor whose spatial size and channel
            count match DINOv3Extractor's output exactly.
        """
        x = feature_map  # (C_in, H, W)

        # 1. Pad to patch-size multiples (reuse DINOv3Extractor's padding)
        x = padding(x, self.patch_size)

        # 2. Learnable patch-to-pixel mapping
        x = self.net(x.unsqueeze(0)).squeeze(0)  # (D, H_patches, W_patches)

        return x

    # ------------------------------------------------------------------
    # AbstractDecoder interface
    # ------------------------------------------------------------------

    def to(self, device) -> 'DINOv3CNNDecoder':
        self.net = self.net.to(device)
        return self

    def load(self, path: str) -> None:
        state_dict = torch.load(path, weights_only=True)
        self.net.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def parameters(self):
        return self.net.parameters()
