import os
from typing import Tuple

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import DINOv3Extractor
from .decoder import DINOv3CNNDecoder

from dinov3.hub.backbones import (
    dinov3_vits16,
    dinov3_vits16plus,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vith16plus,
    dinov3_vit7b16,
)

# Copy from https://github.com/facebookresearch/dinov3/blob/54694f7627fd815f62a5dcc82944ffa6153bbb76/notebooks/pca.ipynb
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
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


def DINOv3ViTExtractor(version: str = "dinov3_vitl16", checkpoint_dir: str = "checkpoints") -> DINOv3Extractor:
    assert version in MODELS, f"DINOv3 version '{version}' not supported. Choose from: {MODELS}"
    n_layers = MODEL_TO_NUM_LAYERS[version]
    local_path = os.path.join(checkpoint_dir, MODEL_TO_FILENAME[version])
    if os.path.isfile(local_path):
        model = MODEL_TO_FACTORY[version](pretrained=True, weights=local_path)
    else:
        model = MODEL_TO_FACTORY[version](pretrained=True)
    return DINOv3Extractor(model=model, n_layers=n_layers, patch_size=PATCH_SIZE)


# Feature dimensions (D) for each backbone
FEATURE_DIMS = {
    MODEL_DINOV3_VITS:   384,
    MODEL_DINOV3_VITSP:  384,
    MODEL_DINOV3_VITB:   768,
    MODEL_DINOV3_VITL:  1024,
    MODEL_DINOV3_VITHP: 1280,
    MODEL_DINOV3_VIT7B: 4096,
}


def build_factory(version: str):
    def factory(embed_dim: int, checkpoint_dir="checkpoints") -> Tuple[AbstractFeatureExtractor, AbstractDecoder]:
        extractor = DINOv3ViTExtractor(version, checkpoint_dir=checkpoint_dir)
        decoder = DINOv3CNNDecoder(
            in_channels=embed_dim,
            out_channels=FEATURE_DIMS[version],
            patch_size=PATCH_SIZE,
        )
        return extractor, decoder
    return factory


for version in MODELS:
    register_extractor_decoder(version, build_factory(version))
