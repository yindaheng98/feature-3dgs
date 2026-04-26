from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

RESOLUTION = 518
PATCH_SIZE = 14
N_PATCHES = RESOLUTION // PATCH_SIZE  # 37


def compute_patch_grid_size(H: int, W: int) -> tuple[int, int]:
    """Compute valid (non-padded) patch grid size after center-pad-to-square + resize to 518.

    Returns (h_p, w_p) — the number of patches along each axis that
    correspond to original image content in the 37x37 token grid.
    """
    max_dim = max(H, W)
    h_p = max(round(H / max_dim * N_PATCHES), 1)
    w_p = max(round(W / max_dim * N_PATCHES), 1)
    return h_p, w_p


class VGGTExtractor(AbstractFeatureExtractor):
    """Feature extractor based on VGGT aggregator.

    Follows the official VGGT preprocessing:
    center-pad to square (black) -> bilinear resize to 518x518 -> aggregator
    -> crop valid patch tokens -> (D, h_p, w_p) feature map.

    VGGT requires multiple images (multi-view aggregation).  Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "VGGT requires multiple images. Use extract_all() instead."
        )

    @torch.no_grad()
    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Extract VGGT features from a sequence of images.

        Args:
            images: Iterable of (C, H, W) tensors in [0, 1] range.

        Yields:
            Per-image feature map of shape (D, h_p, w_p), with padded
            tokens cropped so only the original image content is kept.
        """
        # 1. Center-pad to square and resize to 518x518
        frames = []
        orig_sizes = []
        for img in images:
            _, H, W = img.shape
            max_dim = max(H, W)
            pad_top = (max_dim - H) // 2
            pad_left = (max_dim - W) // 2
            pad_bottom = max_dim - H - pad_top
            pad_right = max_dim - W - pad_left
            square = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            resized = F.interpolate(
                square.unsqueeze(0),
                size=(RESOLUTION, RESOLUTION),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            frames.append(resized)
            orig_sizes.append((H, W))

        # 2. Stack and feed to aggregator  [B=1, S=N, 3, 518, 518]
        batch = torch.stack(frames).unsqueeze(0)
        device = batch.device
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(batch)

        # 3. Extract per-image features from last-layer patch tokens
        tokens = aggregated_tokens_list[-1]          # (1, S, P_total, D)
        patch_tokens = tokens[0, :, ps_idx:, :]      # (S, 37*37, D)
        D = patch_tokens.shape[-1]

        # 4. Crop valid tokens for each image
        for i, (H, W) in enumerate(orig_sizes):
            grid = patch_tokens[i].view(N_PATCHES, N_PATCHES, D)
            h_p, w_p = compute_patch_grid_size(H, W)
            top_p = (N_PATCHES - h_p) // 2
            left_p = (N_PATCHES - w_p) // 2
            feat = grid[top_p: top_p + h_p, left_p: left_p + w_p, :]
            yield feat.permute(2, 0, 1).contiguous()  # (D, h_p, w_p)

    def to(self, device) -> "VGGTExtractor":
        self.model.to(device)
        return self
