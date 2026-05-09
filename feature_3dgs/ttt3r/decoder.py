from __future__ import annotations

import torch
import torch.nn.functional as F
from gaussian_splatting import Camera

from feature_3dgs.decoder import LinearDecoder

from .preprocess import PATCH_SIZE, compute_crop_window


class TTT3RLinearDecoder(LinearDecoder):
    """Decoder aligned with TTT3R's resized, patch-token feature grid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resize: int = 512,
        patch_size: int = PATCH_SIZE,
        square_ok: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, **kwargs)
        self.resize = resize
        self.patch_size = patch_size
        self.square_ok = square_ok

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        _, height, width = feature_map.shape
        resized_height, resized_width, top, left, crop_height, crop_width = compute_crop_window(
            height,
            width,
            self.resize,
            square_ok=self.square_ok,
            patch_size=self.patch_size,
        )
        resized = F.interpolate(
            feature_map.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        cropped = resized[:, top : top + crop_height, left : left + crop_width]
        kernel = self.patch_size
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, kernel, kernel)
        weight = weight / (kernel * kernel)
        decoded = F.conv2d(
            cropped.unsqueeze(0), weight, self.linear.bias, stride=kernel
        ).squeeze(0)
        return decoded.contiguous()

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        encoded = self.encode_feature_pixels(feature_map)
        (
            resized_height,
            resized_width,
            top,
            left,
            crop_height,
            crop_width,
        ) = compute_crop_window(
            camera.image_height,
            camera.image_width,
            self.resize,
            square_ok=self.square_ok,
            patch_size=self.patch_size,
        )
        encoded_crop = F.interpolate(
            encoded.unsqueeze(0),
            size=(crop_height, crop_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        canvas = encoded_crop.new_zeros(encoded_crop.shape[0], resized_height, resized_width)
        canvas[:, top : top + crop_height, left : left + crop_width] = encoded_crop
        return F.interpolate(
            canvas.unsqueeze(0),
            size=(camera.image_height, camera.image_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
