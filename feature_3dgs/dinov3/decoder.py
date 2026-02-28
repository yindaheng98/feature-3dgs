import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import LinearDecoder

from .extractor import padding


class DINOv3LinearAvgDecoder(LinearDecoder):
    """Decoder that aligns Gaussian features with DINOv3 extractor output.

    Extends ``LinearDecoder`` with patch-level average pooling
    (``decode_feature_map``) and bilinear upsampling (``encode_feature_map``)
    to match the spatial resolution of the DINOv3 patch-based extractor.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        """
        Args:
            in_channels:  Per-point semantic embedding dimension rendered by
                          the Gaussian rasteriser.
            out_channels: Feature dimension D produced by DINOv3Extractor.
            patch_size:   Patch size used by the paired DINOv3Extractor.
        """
        super().__init__(in_channels, out_channels)
        self.patch_size = patch_size

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Fused linear + avg-pool via a single Conv2d.

        Equivalent to (but avoids the large (C_feat, H, W) intermediate):

            x = padding(feature_map, P)
            C, H, W = x.shape
            x = x.permute(1, 2, 0).reshape(-1, C)        # (H*W, C_enc)
            x = self.linear(x)                             # (H*W, C_feat)
            x = x.reshape(H, W, -1).permute(2, 0, 1)      # (C_feat, H, W)
            x = F.avg_pool2d(x, kernel_size=P, stride=P)   # (C_feat, H_p, W_p)

        Because avg_pool (mean over P² elements) and the linear layer are
        both linear operations, they fuse into one Conv2d with kernel
        ``weight[:, :, None, None] / P²`` and stride P.
        """
        P = self.patch_size
        x = padding(feature_map, P)                        # (C_enc, H', W')
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, P, P) / (P * P)
        return F.conv2d(x.unsqueeze(0), weight, self.linear.bias, stride=P).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of decode_feature_map: (C_feat, H_p, W_p) -> (C_enc, H, W).

        Applies ``encode_feature_pixels`` then bilinear upsampling to restore
        full spatial resolution.
        """
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)
