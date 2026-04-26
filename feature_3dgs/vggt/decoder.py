import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import LinearDecoder

from .extractor import compute_square_padding, compute_square_valid_region


class VGGTLinearAvgDecoder(LinearDecoder):
    """Decoder that aligns Gaussian features with VGGTExtractor output.

    ``decode_feature_map``: center-pad -> interpolate to the square model
    resolution -> fused avg-pool + linear projection -> crop valid region.

    ``encode_feature_map``: linear inverse projection (channel down), then
    bilinear upsample to full image resolution.

    Args:
        feat_size: spatial size of the extractor's square feature grid.
            37 for ``VGGTExtractor`` (patch tokens), 259 for
            ``VGGTrackExtractor`` (DPT feature map with down_ratio=2).
        kernel_size: square downsampling kernel. 14 for ``VGGTExtractor``,
            2 for ``VGGTrackExtractor``.
    """

    def __init__(self, in_channels: int, out_channels: int, feat_size: int, kernel_size: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.feat_size = feat_size
        self.kernel_size = kernel_size

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pad, interpolate, fused downsample+linear projection, then crop.

        This mirrors the extractor geometry so each output pixel corresponds
        to the same square-coordinate input region as the paired extractor.
        """
        _, H, W = feature_map.shape
        # 1. Center-pad to the same square coordinate system as the extractor
        pad_left, pad_right, pad_top, pad_bottom = compute_square_padding(H, W)
        square = F.pad(feature_map, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        # 2. Interpolate to the square model resolution before downsampling
        square_resolution = self.feat_size * self.kernel_size
        square = F.interpolate(
            square.unsqueeze(0),
            size=(square_resolution, square_resolution),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        # 3. Fused avg-pool downsampling + learned linear projection
        k = self.kernel_size
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, k, k) / (k * k)
        square = F.conv2d(square.unsqueeze(0), weight, self.linear.bias, stride=k).squeeze(0)

        # 4. Crop the valid region
        top, left, h, w = compute_square_valid_region(H, W, square_size=self.feat_size)
        return square[:, top: top + h, left: left + w].contiguous()

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of decode_feature_map: (C_feat, H_p, W_p) -> (C_enc, H, W).

        Applies ``encode_feature_pixels`` then bilinear upsampling to restore
        full spatial resolution.
        """
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)
