import contextlib

import torch
import torch.nn.functional as F
from uniception.models.encoders import ViTEncoderInput
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from feature_3dgs.extractor import AbstractFeatureExtractor


def select_inference_hw(height: int, width: int, resolutions: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Pick UFM's fixed canvas for one image.

    ``resolutions`` entries are ``(width, height)``, the same order as
    ``inference_resolution`` in the UFM config. With one trained resolution
    this is that canvas. With several, it is the one whose aspect is closest
    to the image, matching ``AutomaticShapeSelection`` when both frames share
    that aspect.

    Returns ``(height, width)``.
    """
    if not resolutions:
        raise ValueError("UFM inference_resolution is empty")

    def aspect_error(resolution: tuple[int, int]) -> float:
        target_w, target_h = resolution
        return abs(target_h / target_w - height / width)

    target_w, target_h = min(resolutions, key=aspect_error)
    return int(target_h), int(target_w)


class UniCeptionExtractor(AbstractFeatureExtractor):
    """UFM image encoder, the UniCeption DINOv2 intermediate-feature returner.

    Preprocessing follows ``UniFlowMatchPointTracker.track_batch`` and
    ``UniFlowMatchModelsBase.predict_correspondences_batched``: quantize
    ``[0, 1]`` RGB to uint8 by truncation, apply the encoder's normalization,
    then bilinear-resize with ``antialias=True`` onto the trained canvas.
    The forward is that canvas through ``model.encoder``, which is what UFM
    reads back as ``feat_list[-1].features``. On CUDA the encoder runs under
    the same bfloat16 autocast as ``predict_correspondences_batched``.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        resolutions: tuple[tuple[int, int], ...],
    ):
        self.encoder = encoder
        self.encoder.eval()
        self.resolutions = resolutions
        self.patch_size = int(encoder.patch_size)
        self._feature_dim = int(encoder.enc_embed_dim)
        for target_w, target_h in resolutions:
            if target_h % self.patch_size or target_w % self.patch_size:
                raise ValueError(
                    f"inference resolution {(target_w, target_h)} is not divisible by patch size {self.patch_size}"
                )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Return the ``(1, 3, H_t, W_t)`` tensor UFM feeds its encoder."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("UniCeptionExtractor expects an RGB image with shape [3, H, W]")
        height, width = int(image.shape[1]), int(image.shape[2])
        target_h, target_w = select_inference_hw(height, width, self.resolutions)
        # track_4dgs/ufm/tracker.py track_batch: clamp, scale to 255, truncate.
        quantized = image.detach().clamp(0, 1).mul(255).byte()
        # uniflowmatch/models/base.py uint8 branch.
        norm = IMAGE_NORMALIZATION_DICT[self.encoder.data_norm_type]
        mean = norm.mean.view(1, 3, 1, 1).to(device=quantized.device, non_blocking=True)
        std = norm.std.view(1, 3, 1, 1).to(device=quantized.device, non_blocking=True)
        normalized = (quantized.unsqueeze(0).float() / 255.0 - mean) / std
        # ResizeToFixedManipulation: bilinear, align_corners=False, antialias=True.
        return F.interpolate(
            normalized,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Extract the last encoder feature map.

        Args:
            image: ``(3, H, W)`` RGB in ``[0, 1]``.

        Returns:
            ``(D, H_t / patch, W_t / patch)`` float32 feature map. ``D`` is
            1024 for ViT-L checkpoints and 1536 for ViT-G.
        """
        prepared = self.prepare(image)
        if prepared.is_cuda:
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context = contextlib.nullcontext()
        with context:
            layers = self.encoder(ViTEncoderInput(
                image=prepared,
                data_norm_type=self.encoder.data_norm_type,
            ))
        return layers[-1].features[0].float()

    def to(self, device) -> "UniCeptionExtractor":
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        return self
