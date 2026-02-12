import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from feature_3dgs.dataset.abc import AbstractFeatureExtractor

from dinov3.hub.backbones import (
    dinov3_vits16,
    dinov3_vits16plus,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vith16plus,
    dinov3_vit7b16,
)

# Copy from https://github.com/facebookresearch/dinov3/blob/54694f7627fd815f62a5dcc82944ffa6153bbb76/notebooks/pca.ipynb
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

# Model name -> factory function
MODEL_TO_FACTORY = {
    MODEL_DINOV3_VITS: dinov3_vits16,
    MODEL_DINOV3_VITSP: dinov3_vits16plus,
    MODEL_DINOV3_VITB: dinov3_vitb16,
    MODEL_DINOV3_VITL: dinov3_vitl16,
    MODEL_DINOV3_VITHP: dinov3_vith16plus,
    MODEL_DINOV3_VIT7B: dinov3_vit7b16,
}
MODELS = [
    MODEL_DINOV3_VITS,
    MODEL_DINOV3_VITSP,
    MODEL_DINOV3_VITB,
    MODEL_DINOV3_VITL,
    MODEL_DINOV3_VITHP,
    MODEL_DINOV3_VIT7B,
]


def _make_model_filename(
    compact_arch_name: str,
    hash: str,
    patch_size: int = PATCH_SIZE,
    weights_name: str = "lvd1689m",
    version: str | None = None,
) -> str:
    """Construct the local checkpoint file path for a DINOv3 model.

    Follows the same naming convention as ``dinov3.hub.backbones``:
        ``{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth``
    """
    model_name = "dinov3"
    # Replicate _make_dinov3_vit_model_arch logic
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    version_suffix = f"_{version}" if version else ""
    hash_suffix = f"-{hash}" if hash else ""
    filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return filename


# Model name -> local checkpoint filename
MODEL_TO_FILENAME = {
    MODEL_DINOV3_VITS: _make_model_filename(compact_arch_name="vits", hash="08c60483"),
    MODEL_DINOV3_VITSP: _make_model_filename(compact_arch_name="vitsplus", hash="4057cbaa"),
    MODEL_DINOV3_VITB: _make_model_filename(compact_arch_name="vitb", hash="73cec8be"),
    MODEL_DINOV3_VITL: _make_model_filename(compact_arch_name="vitl", hash="8aa4cbdd"),
    MODEL_DINOV3_VITHP: _make_model_filename(compact_arch_name="vithplus", hash="7c1da9a5"),
    MODEL_DINOV3_VIT7B: _make_model_filename(compact_arch_name="vit7b", hash="a955f4ea"),
}


def padding(image: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Pad image so that H and W are multiples of patch_size."""
    _, h, w = image.shape  # (C, H, W)
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image


class DINOv3Extractor(AbstractFeatureExtractor):
    """Feature extractor based on DINOv3 ViT models.

    Extracts dense patch-level features from the last transformer layer,
    then bilinearly interpolates them to the original image resolution.
    """

    def __init__(self, version: str = "dinov3_vitl16", checkpoint_dir: str = "checkpoints"):
        assert version in MODELS, f"DINOv3 version '{version}' not supported. Choose from: {MODELS}"
        self.n_layers = MODEL_TO_NUM_LAYERS[version]
        local_path = os.path.join(checkpoint_dir, MODEL_TO_FILENAME[version])
        if os.path.isfile(local_path):
            self.model = MODEL_TO_FACTORY[version](pretrained=True, weights=local_path)
        else:
            self.model = MODEL_TO_FACTORY[version](pretrained=True)
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
        x = padding(x)

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
