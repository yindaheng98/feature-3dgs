import json
from typing import Tuple

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from uniception.models.encoders import feature_returner_encoder_factory

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import UniCeptionExtractor
from .decoder import UniCeptionLinearAvgDecoder

# track_4dgs/ufm/tracker.py MODEL_REPOS
VARIANTS = {
    "ufm-base": "infinity1096/UFM-Base",
    "ufm-refine": "infinity1096/UFM-Refine",
    "ufm-base-980": "infinity1096/UFM-Base-980",
    "ufm-refine-980": "infinity1096/UFM-Refine-980",
    "ufm-base-dinov2l": "infinity1096/UFM-Base-DINOv2L-init",
    "ufm-base-dinov2g": "infinity1096/UFM-Base-DINOv2G-init",
}


def parse_resolutions(inference_resolution) -> tuple[tuple[int, int], ...]:
    """Normalize UFM ``inference_resolution`` to ``((width, height), ...)``."""
    if inference_resolution is None:
        inference_resolution = [(560, 420)]
    if isinstance(inference_resolution[0], int):
        inference_resolution = [inference_resolution]
    return tuple((int(width), int(height)) for width, height in inference_resolution)


def load_encoder_tensors(repo: str) -> dict[str, torch.Tensor]:
    """Load ``encoder.*`` tensors the same way ``PyTorchModelHubMixin`` finds weights."""
    prefix = "encoder."
    try:
        weights = hf_hub_download(repo_id=repo, filename="model.safetensors")
    except EntryNotFoundError:
        weights = hf_hub_download(repo_id=repo, filename="pytorch_model.bin")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    selected = {}
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                selected[key[len(prefix):]] = handle.get_tensor(key)
    return selected


def load_ufm_encoder(variant: str = "ufm-base") -> tuple[torch.nn.Module, tuple[tuple[int, int], ...]]:
    """Build the UniCeption encoder inside a UFM checkpoint and load its weights.

    Construction is ``feature_returner_encoder_factory(encoder_str, **encoder_kwargs)``,
    the same call as ``UniFlowMatch.__init__``. Hub DINOv2 weights are loaded
    first and then overwritten by the checkpoint, and ``load_state_dict`` uses
    ``strict=False``, matching ``PyTorchModelHubMixin._from_pretrained``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown UFM variant '{variant}'. Choose from {list(VARIANTS)}")
    repo = VARIANTS[variant]
    config_path = hf_hub_download(repo_id=repo, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    encoder = feature_returner_encoder_factory(config["encoder_str"], **config["encoder_kwargs"])
    selected = load_encoder_tensors(repo)
    if not selected:
        raise RuntimeError(f"Checkpoint {repo} has no encoder weights")
    own = encoder.state_dict()
    # DINOv2 drops mask_token before UFM saves. Any other unknown key would be
    # loaded into the original module and must not be skipped here.
    unexpected = [key for key in selected if key not in own and not key.endswith("mask_token")]
    if unexpected:
        raise RuntimeError(f"Checkpoint {repo} has encoder keys this UniCeption build does not: {unexpected}")
    selected = {key: value for key, value in selected.items() if key in own}
    # strict=False matches PyTorchModelHubMixin._from_pretrained: keys absent
    # from the checkpoint stay at the torch.hub DINOv2 initialization.
    encoder.load_state_dict(selected, strict=False)
    encoder.eval()
    return encoder, parse_resolutions(config.get("inference_resolution"))


def UniCeptionFeatureExtractor(variant: str = "ufm-base") -> UniCeptionExtractor:
    encoder, resolutions = load_ufm_encoder(variant)
    return UniCeptionExtractor(encoder=encoder, resolutions=resolutions)


def build_factory(variant: str):
    def factory(encoded_dim: int, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        del configs
        extractor = UniCeptionFeatureExtractor(variant)
        decoder = UniCeptionLinearAvgDecoder(
            in_channels=encoded_dim,
            out_channels=extractor.feature_dim,
            patch_size=extractor.patch_size,
            resolutions=extractor.resolutions,
        )
        return extractor, decoder
    return factory


for variant in VARIANTS:
    register_extractor_decoder(variant, build_factory(variant))
