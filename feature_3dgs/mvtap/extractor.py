import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

# Official MV-TAP defaults: stride (models/mvtap.py L37), latent_dim (L42)
# https://github.com/cvlab-kaist/MV-TAP/blob/b248aea43dd04c79679563abb44bb6bd914e1224/models/mvtap.py#L37
STRIDE = 4
FEATURE_DIM = 128


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

    Matches the original MVTAP ``fnet`` path: map ``[0, 1]`` images to
    ``[-1, 1]``, pad so H/W are multiples of ``STRIDE``, run the CNN, then
    L2-normalise.  Output is a dense map of shape ``(FEATURE_DIM, H/STRIDE, W/STRIDE)``.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Extract MVTAP ``fnet`` features from an image tensor.

        Args:
            image: (C, H, W) tensor in [0, 1] range.

        Returns:
            Feature map of shape (D, H', W') at 1/STRIDE resolution.
        """
        x = 2 * image - 1.0
        x = padding(x)
        feature_map = self.model(x.unsqueeze(0))  # (1, D, H_s, W_s)
        feature_map = F.normalize(feature_map, dim=1, eps=1e-12, p=2)
        return feature_map.squeeze(0)  # (D, H_s, W_s)

    def to(self, device) -> "MVTAPExtractor":
        self.model.to(device)
        return self
