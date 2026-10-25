import torch
import torch.nn.functional as F
from gaussian_splatting import Camera
from feature_3dgs.decoder import CosineLinearDecoder

from .extractor import select_inference_hw


class UniCeptionLinearAvgDecoder(CosineLinearDecoder):
    """Decoder that aligns Gaussian features with ``UniCeptionExtractor`` output.

    The extractor stretches each image onto UFM's trained canvas, then the
    ViT keeps one token per ``patch_size`` pixels. This decoder resizes the
    rendered map onto that same canvas and average-pools each patch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        resolutions: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.patch_size = patch_size
        self.resolutions = resolutions

    def decode_feature_map(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Resize onto the UFM canvas, then fused linear + patch average-pool.

        Equivalent to interpolating like ``ResizeToFixedManipulation``, applying
        the linear layer per pixel, and ``avg_pool2d`` with kernel ``patch_size``.
        The pool and the linear layer fuse into one strided convolution.
        """
        _, height, width = feature_map.shape
        target_h, target_w = select_inference_hw(height, width, self.resolutions)
        canvas = F.interpolate(
            feature_map.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        patch = self.patch_size
        lin_weight, lin_bias = self.linear.weight, self.linear.bias
        if weight is not None:
            lin_weight = weight @ lin_weight
            lin_bias = F.linear(lin_bias, weight, bias)
        kernel = lin_weight[:, :, None, None].expand(-1, -1, patch, patch) / (patch * patch)
        return F.conv2d(canvas, kernel, lin_bias, stride=patch).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse channel map, then bilinear upsample to the camera image."""
        encoded = self.encode_feature_pixels(feature_map)
        return F.interpolate(
            encoded.unsqueeze(0),
            size=(camera.image_height, camera.image_width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
