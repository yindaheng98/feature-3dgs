import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from feature_3dgs import AbstractFeatureExtractor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def padding(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad image so that H and W are multiples of patch_size."""
    _, h, w = image.shape  # (C, H, W)
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image


class DINOv3Extractor(AbstractFeatureExtractor):
    """Feature extractor based on DINOv3 models.

    Extracts dense patch-level features from the last transformer layer,
    then bilinearly interpolates them to the original image resolution.
    """

    def __init__(self, model: nn.Module, n_layers: int, patch_size: int):
        self.model = model
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3 features from an image tensor.

        Args:
            image: (C, H, W) tensor in [0, 1] range.

        Returns:
            Feature map of shape (D, H, W) interpolated to original spatial size.
        """
        x = image

        # Normalize with ImageNet statistics
        x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Pad to patch-size multiple
        x = padding(x, self.patch_size)

        # Forward pass: extract features from all layers, take the last one
        # get_intermediate_layers returns a tuple of (1, D, H_p, W_p) tensors
        feats = self.model.get_intermediate_layers(
            x.unsqueeze(0),  # add batch dim
            n=range(self.n_layers),
            reshape=True,
            norm=True,
        )
        # Last layer features: (1, D, H_patches, W_patches)
        feature_map = feats[-1].squeeze(0)  # (D, H_patches, W_patches)

        # # Interpolate back to original spatial resolution
        # _, orig_h, orig_w = image.shape
        # feature_map = F.interpolate(
        #     feature_map.unsqueeze(0),
        #     size=(orig_h, orig_w),
        #     mode="bilinear",
        #     align_corners=True,
        # ).squeeze(0)  # (D, H, W)

        return feature_map

    def to(self, device) -> 'DINOv3Extractor':
        self.model.to(device)
        return self
