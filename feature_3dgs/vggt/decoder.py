import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import LinearDecoder

from .extractor import compute_patch_grid_size


class VGGTLinearAvgDecoder(LinearDecoder):
    """Decoder that aligns Gaussian features with VGGTExtractor output.

    ``decode_feature_map``: adaptive average-pool to match the extractor's
    feature grid, then linear projection (channel up).

    ``encode_feature_map``: linear inverse projection (channel down), then
    bilinear upsample to full image resolution.

    Args:
        feat_size: spatial size of the extractor's square feature grid.
            37 for ``VGGTExtractor`` (patch tokens), 259 for
            ``VGGTrackExtractor`` (DPT feature map with down_ratio=2).
    """

    def __init__(self, in_channels: int, out_channels: int, feat_size: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.feat_size = feat_size

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Fused avg-pool + linear via a single ``F.conv2d``.

        Pads (H, W) to exact multiples of (h_p, w_p), then applies one
        strided Conv2d whose kernel averages each patch and projects
        channels simultaneously:  ``weight / (kh*kw)`` with stride ``(kh, kw)``.
        """
        _, H, W = feature_map.shape
        # 1. Pad feature_map to exact multiples of (h_p, w_p)
        h_p, w_p = compute_patch_grid_size(H, W, self.feat_size)
        pad_h = (h_p - H % h_p) % h_p
        pad_w = (w_p - W % w_p) % w_p
        if pad_h or pad_w:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            feature_map = F.pad(feature_map, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        # 2. Apply Conv2d (Avg-pool + Linear)
        kh = feature_map.shape[1] // h_p
        kw = feature_map.shape[2] // w_p
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, kh, kw) / (kh * kw)
        return F.conv2d(feature_map.unsqueeze(0), weight, self.linear.bias, stride=(kh, kw)).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of decode_feature_map: (C_feat, H_p, W_p) -> (C_enc, H, W).

        Applies ``encode_feature_pixels`` then bilinear upsampling to restore
        full spatial resolution.
        """
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)
