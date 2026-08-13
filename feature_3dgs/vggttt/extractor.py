from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from vggttt.nets.ttt import TTTOperator

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.vggt.extractor import (
    FEAT_SIZE,
    PATCH_SIZE,
    RESOLUTION,
    compute_square_valid_region,
    padding_square,
)


class VGGTTTExtractor(AbstractFeatureExtractor):
    """Feature extractor based on VGG-T3 aggregator.

    Preprocessing matches ``VGGTExtractor``: center-pad to square (black) ->
    bicubic resize to *img_load_resolution* -> bilinear resize to 518x518 ->
    TTT aggregator -> crop valid patch tokens -> (D, h_p, w_p) feature map.

    VGG-T3 requires multiple images (multi-view aggregation). Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model, img_load_resolution: int = 1024):
        self.model = model
        self.model.eval()
        self.img_load_resolution = img_load_resolution

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "VGG-T3 requires multiple images. Use extract_all() instead."
        )

    def run_aggregator(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        if batch.device.type != "cuda":
            raise RuntimeError("VGG-T3 feature extraction requires a CUDA device")
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(batch.device)[0] >= 8
            else torch.float16
        )
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx, _ = self.model.aggregator(
                batch,
                attn_kwargs={"info": {"ttt_op_order": [
                    TTTOperator(start=0, end=None, compute_grad=True, update=True, apply=False),
                    TTTOperator(start=0, end=None, compute_grad=False, update=False, apply=True),
                ]}},
            )
        return aggregated_tokens_list, ps_idx

    @torch.no_grad()
    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Extract VGG-T3 features from a sequence of images.

        Args:
            images: Iterable of (C, H, W) tensors in [0, 1] range.

        Yields:
            Per-image feature map of shape (D, h_p, w_p), with padded
            tokens cropped so only the original image content is kept.
        """
        # 1. Preprocess each image: center-pad + bicubic to img_load_resolution
        frames = []
        orig_sizes = []
        for img in images:
            frames.append(padding_square(img, self.img_load_resolution))
            orig_sizes.append(img.shape[1:])

        # 2. Bilinear down to 518, then feed to TTT aggregator
        batch = torch.stack(frames)
        if batch.shape[-2:] != (RESOLUTION, RESOLUTION):
            batch = F.interpolate(batch, size=(RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
        batch = batch.unsqueeze(0)
        aggregated_tokens_list, ps_idx = self.run_aggregator(batch)

        # 3. Extract per-image features from last-layer patch tokens
        tokens = aggregated_tokens_list[-1]          # (1, S, P_total, D)
        patch_tokens = tokens[0, :, ps_idx:, :]      # (S, 37*37, D)
        D = patch_tokens.shape[-1]

        # 4. Crop valid tokens for each image
        for i, (H, W) in enumerate(orig_sizes):
            grid = patch_tokens[i].view(FEAT_SIZE, FEAT_SIZE, D)
            top_p, left_p, h_p, w_p = compute_square_valid_region(H, W, square_size=FEAT_SIZE)
            feat = grid[top_p: top_p + h_p, left_p: left_p + w_p, :]
            yield feat.permute(2, 0, 1).contiguous()  # (D, h_p, w_p)

    def to(self, device) -> "VGGTTTExtractor":
        self.model.to(device)
        return self
