import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import CosineLinearDecoder

from .extractor import MODEL_HEIGHT, MODEL_WIDTH, STRIDE, input_size, padding


class MVTAPLinearAvgDecoder(CosineLinearDecoder):
    """Decoder that aligns Gaussian features with MVTAPExtractor output.

    Channel projection is the trainable linear layer.  Spatial resampling
    repeats the extractor: resize under ``input_height`` / ``input_width``,
    reflect-pad so H/W are multiples of ``STRIDE``, then average-pool to the
    1/4 field ``(H // stride, W // stride)``.

    ``input_height`` / ``input_width`` match ``MVTAPExtractor``.  Both set
    (default 384x512) stretches to that size.  One ``None`` keeps aspect
    ratio.  Both ``None`` keep the native image size.

    Output grid ``H4, W4 = H // stride``:
        https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/mvtap.py#L391
    Official chart ``model_resolution`` 384x512:
        https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/mvtap.py#L43
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int = MODEL_HEIGHT,
        input_width: int = MODEL_WIDTH,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.input_height = input_height
        self.input_width = input_width

    def decode_feature_map(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Resize and pad like ``MVTAPExtractor``, then fused linear + avg-pool.

        Equivalent to (but avoids the large (C_feat, H, W) intermediate):

            _, H, W = feature_map.shape
            th, tw = input_size(H, W, input_height, input_width)
            x = F.interpolate(feature_map, (th, tw), mode="bilinear", align_corners=False)
            x = padding(x)                                 # (C_enc, H', W')
            C, H, W = x.shape
            x = x.permute(1, 2, 0).reshape(-1, C)          # (H*W, C_enc)
            x = self.linear(x)                              # (H*W, C_feat)
            x = x.reshape(H, W, -1).permute(2, 0, 1)       # (C_feat, H, W)
            x = F.avg_pool2d(x, kernel_size=STRIDE, stride=STRIDE)

        The interpolate and ``padding`` match the extractor.  Avg-pool (mean
        over ``STRIDE²`` elements) and the linear layer are both linear, so
        they fuse into one Conv2d with kernel ``W[:, :, None, None] / STRIDE²``
        and stride ``STRIDE``.  An optional extra linear (``weight`` / ``bias``)
        is fused the same way.
        """
        _, height, width = feature_map.shape
        th, tw = input_size(height, width, self.input_height, self.input_width)
        if (height, width) != (th, tw):
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
        x = padding(feature_map)
        lin_weight, lin_bias = self.linear.weight, self.linear.bias
        if weight is not None:
            lin_weight = weight @ lin_weight
            lin_bias = F.linear(lin_bias, weight, bias)
        kernel = lin_weight[:, :, None, None].expand(-1, -1, STRIDE, STRIDE) / (STRIDE * STRIDE)
        return F.conv2d(x.unsqueeze(0), kernel, lin_bias, stride=STRIDE).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of decode_feature_map: (C_feat, H_s, W_s) -> (C_enc, H, W).

        Applies ``encode_feature_pixels`` then bilinear upsampling to restore
        full spatial resolution.
        """
        x = self.encode_feature_pixels(feature_map)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode="bilinear", align_corners=True).squeeze(0)
