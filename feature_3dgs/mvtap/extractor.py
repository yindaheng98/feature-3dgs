import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

# Official MV-TAP defaults: stride / latent_dim (models/mvtap.py L37, L42),
# model_resolution 384x512 (L43-L44), inference resize_H/resize_W.
# https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/mvtap.py#L37
STRIDE = 4
FEATURE_DIM = 128
MODEL_HEIGHT = 384
MODEL_WIDTH = 512


def input_size(height: int, width: int, target_height: int = None, target_width: int = None) -> tuple[int, int]:
    """Encoder input size before stride padding.

    Both dimensions set: stretch to that size. One ``None``: scale the
    other side so aspect ratio is preserved. Both ``None``: keep ``(height, width)``.
    """
    if target_height is None and target_width is None:
        return height, width
    if target_height is None:
        return round(height * target_width / width), target_width
    if target_width is None:
        return target_height, round(width * target_height / height)
    return target_height, target_width


def padding(image: torch.Tensor) -> torch.Tensor:
    """Pad image so that H and W are multiples of ``STRIDE``."""
    _, h, w = image.shape  # (C, H, W)
    pad_h = (STRIDE - h % STRIDE) % STRIDE
    pad_w = (STRIDE - w % STRIDE) % STRIDE
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image


class MVTAPExtractor(AbstractFeatureExtractor):
    """Feature extractor based on MVTAP ``BasicEncoder``.

    Resizes RGB, maps ``[0, 1]`` to ``[-1, 1]``, pads so H/W are multiples
    of ``STRIDE``, runs the frozen CNN, then L2-normalises.

    ``input_height`` / ``input_width`` default to 384x512. One ``None``
    keeps aspect ratio. Both ``None`` skip the resize.
    """

    def __init__(
        self,
        model: nn.Module,
        input_height: int = MODEL_HEIGHT,
        input_width: int = MODEL_WIDTH,
    ):
        self.model = model
        self.model.eval()
        self.input_height = input_height
        self.input_width = input_width

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Extract MVTAP ``fnet`` features from an image tensor.

        Args:
            image: (C, H, W) tensor in [0, 1] range.

        Returns:
            Feature map of shape (D, H_s, W_s) on the 1/STRIDE chart.
        """
        _, h, w = image.shape
        th, tw = input_size(h, w, self.input_height, self.input_width)
        if (h, w) != (th, tw):
            image = F.interpolate(image.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
        x = 2 * image - 1.0
        x = padding(x)
        feature_map = self.model(x.unsqueeze(0))  # (1, D, H_s, W_s)
        feature_map = F.normalize(feature_map, dim=1, eps=1e-12, p=2)
        return feature_map.squeeze(0)  # (D, H_s, W_s)

    def to(self, device) -> "MVTAPExtractor":
        self.model.to(device)
        return self
