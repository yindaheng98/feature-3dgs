import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import CosineLinearDecoder

from .extractor import padding


class MVTAPLinearAvgDecoder(CosineLinearDecoder):
    """Decoder that aligns Gaussian features with MVTAPExtractor output.

    Channel projection is the trainable linear layer.  Spatial resampling
    follows MV-TAP ``BasicEncoder``, which bilinearly resizes multi-scale CNN
    maps to ``(H // stride, W // stride)`` with ``align_corners=True`` — a
    dense 1/4 feature field, not disjoint patch averages.

    Source (``_bilinear_intepolate``):
        https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/blocks.py#L273-L279
    Output grid ``H4, W4 = H // stride``:
        https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/mvtap.py#L391
    """

    def __init__(self, *args, stride: int, **configs):
        """
        Args:
            in_channels:  Per-point semantic embedding dimension rendered by
                          the Gaussian rasteriser.
            out_channels: Feature dimension D produced by MVTAPExtractor.
            stride:       Downsample stride used by the paired MVTAPExtractor.
        """
        super().__init__(*args, **configs)
        self.stride = stride

    def decode_feature_map(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Linear projection, then bilinear downsample like ``BasicEncoder``.

        Equivalent to:

            x = padding(feature_map, S)                   # (C_enc, H', W')
            x = self.decode_feature_pixels(x, weight, bias)
            x = F.interpolate(
                x.unsqueeze(0),
                (H' // S, W' // S),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        The interpolate is the same operator as MV-TAP
        ``models/blocks.py`` ``BasicEncoder.forward`` L273-L279:

            def _bilinear_intepolate(x):
                return F.interpolate(
                    x,
                    (H // self.stride, W // self.stride),
                    mode="bilinear",
                    align_corners=True,
                )

        Linear (per-channel mix) and bilinear interpolate (per-channel
        spatial mix) commute, so an optional extra linear is fused inside
        ``decode_feature_pixels`` before the resize.
        """
        S = self.stride
        x = padding(feature_map, S)
        x = self.decode_feature_pixels(x, weight=weight, bias=bias)
        _, H, W = x.shape
        # models/blocks.py L273-L279: bilinear to (H // stride, W // stride)
        return F.interpolate(
            x.unsqueeze(0),
            (H // S, W // S),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of decode_feature_map: (C_feat, H_s, W_s) -> (C_enc, H, W).

        Applies ``encode_feature_pixels`` then the inverse of
        ``_bilinear_intepolate`` (``models/blocks.py`` L273-L279): bilinear
        upsample with ``align_corners=True`` to the original image size.
        """
        x = self.encode_feature_pixels(feature_map)
        return F.interpolate(
            x.unsqueeze(0),
            size=(camera.image_height, camera.image_width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
