import os

from .extractor import DINOv3Extractor

from dinov3.hub.backbones import (
    dinov3_convnext_tiny,
    dinov3_convnext_small,
    dinov3_convnext_base,
    dinov3_convnext_large,
)


MODEL_DINOV3_CONVNEXTT = "dinov3_convnext_tiny"
MODEL_DINOV3_CONVNEXTS = "dinov3_convnext_small"
MODEL_DINOV3_CONVNEXTB = "dinov3_convnext_base"
MODEL_DINOV3_CONVNEXTL = "dinov3_convnext_large"
NUM_STAGES = 4  # ConvNeXt always has 4 stages
# ConvNeXt: stem stride=4, then 3 downsample layers stride=2 each => total stride = 4*2*2*2 = 32
INPUT_PAD_SIZE = 32

# Model name -> factory function
MODEL_TO_FACTORY = {
    MODEL_DINOV3_CONVNEXTT: dinov3_convnext_tiny,
    MODEL_DINOV3_CONVNEXTS: dinov3_convnext_small,
    MODEL_DINOV3_CONVNEXTB: dinov3_convnext_base,
    MODEL_DINOV3_CONVNEXTL: dinov3_convnext_large,
}
MODELS = [
    MODEL_DINOV3_CONVNEXTT,
    MODEL_DINOV3_CONVNEXTS,
    MODEL_DINOV3_CONVNEXTB,
    MODEL_DINOV3_CONVNEXTL,
]


def _make_model_filename(
    compact_arch_name: str,
    hash: str,
    weights_name: str = "lvd1689m",
) -> str:
    """Construct the local checkpoint filename for a DINOv3 ConvNeXt model.

    Follows the naming convention in ``dinov3.hub.backbones``:
        ``dinov3_{arch}_pretrain_{weights}{hash}.pth``
    """
    model_name = "dinov3"
    hash_suffix = f"-{hash}" if hash else ""
    return f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"


# Model name -> local checkpoint filename
MODEL_TO_FILENAME = {
    MODEL_DINOV3_CONVNEXTT: _make_model_filename(compact_arch_name="convnext_tiny", hash="21b726bb"),
    MODEL_DINOV3_CONVNEXTS: _make_model_filename(compact_arch_name="convnext_small", hash="296db49d"),
    MODEL_DINOV3_CONVNEXTB: _make_model_filename(compact_arch_name="convnext_base", hash="801f2ba9"),
    MODEL_DINOV3_CONVNEXTL: _make_model_filename(compact_arch_name="convnext_large", hash="61fa432d"),
}


class DINOv3ConvNextExtractor(DINOv3Extractor):
    """Feature extractor based on DINOv3 ConvNeXt models.

    ConvNeXt has 4 stages (stem + 3 downsampling layers).
    By default we extract the last-stage features via ``get_intermediate_layers``.
    """

    def __init__(self, version: str = MODEL_DINOV3_CONVNEXTB, checkpoint_dir: str = "checkpoints"):
        assert version in MODELS, f"DINOv3 ConvNeXt version '{version}' not supported. Choose from: {MODELS}"
        local_path = os.path.join(checkpoint_dir, MODEL_TO_FILENAME[version])
        if os.path.isfile(local_path):
            model = MODEL_TO_FACTORY[version](pretrained=True, weights=local_path)
        else:
            model = MODEL_TO_FACTORY[version](pretrained=True)
        super().__init__(model=model, n_layers=NUM_STAGES, patch_size=INPUT_PAD_SIZE)
