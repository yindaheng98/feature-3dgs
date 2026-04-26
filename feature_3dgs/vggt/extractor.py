from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

RESOLUTION = 518
PATCH_SIZE = 14


def compute_patch_grid_size(H: int, W: int, feat_size: int) -> tuple[int, int]:
    """Compute valid (non-padded) grid size after center-pad-to-square.

    Works for any square feature map produced from a center-padded image:
    patch token grids (37x37), DPT feature maps (259x259), etc.

    Args:
        H, W: original image spatial dimensions (before padding).
        feat_size: spatial size of the square feature map (default ``N_PATCHES``).

    Returns:
        (h_f, w_f) — elements along each axis corresponding to original content.
    """
    max_dim = max(H, W)
    h_f = max(round(H / max_dim * feat_size), 1)
    w_f = max(round(W / max_dim * feat_size), 1)
    return h_f, w_f


def padding_square(img: torch.Tensor, target_resolution: int = 1024) -> torch.Tensor:
    """Center-pad to square + bicubic resize, matching the official VGGT
    ``load_and_preprocess_images_square``.

    Args:
        img: (C, H, W) tensor in [0, 1].
        target_resolution: output square size (default 1024).

    Returns:
        (C, target_resolution, target_resolution) tensor.
    """
    _, H, W = img.shape
    max_dim = max(H, W)
    pad_top = (max_dim - H) // 2
    pad_left = (max_dim - W) // 2
    pad_bottom = max_dim - H - pad_top
    pad_right = max_dim - W - pad_left
    square = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return F.interpolate(
        square.unsqueeze(0),
        size=(target_resolution, target_resolution),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    ).clamp_(0, 1).squeeze(0)


class VGGTExtractor(AbstractFeatureExtractor):
    """Feature extractor based on VGGT aggregator.

    Preprocessing matches the official VGGT ``load_and_preprocess_images_square``
    + ``run_VGGT`` chain: center-pad to square (black) -> bicubic resize to
    *img_load_resolution* -> bilinear resize to 518x518 -> aggregator -> crop
    valid patch tokens -> (D, h_p, w_p) feature map.

    VGGT requires multiple images (multi-view aggregation).  Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model, img_load_resolution: int = 1024):
        self.model = model
        self.model.eval()
        self.img_load_resolution = img_load_resolution

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
        # 1. Preprocess each image: center-pad + bicubic to img_load_resolution
        frames = []
        orig_sizes = []
        for img in images:
            frames.append(padding_square(img, self.img_load_resolution))
            orig_sizes.append(img.shape[1:])

        # 2. Bilinear down to 518 (matches run_VGGT), then feed to aggregator
        batch = torch.stack(frames)
        if batch.shape[-2:] != (RESOLUTION, RESOLUTION):
            batch = F.interpolate(batch, size=(RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
        batch = batch.unsqueeze(0)
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
        N_PATCHES = RESOLUTION // PATCH_SIZE  # 37
        for i, (H, W) in enumerate(orig_sizes):
            grid = patch_tokens[i].view(N_PATCHES, N_PATCHES, D)
            h_p, w_p = compute_patch_grid_size(H, W, feat_size=N_PATCHES)
            top_p = (N_PATCHES - h_p) // 2
            left_p = (N_PATCHES - w_p) // 2
            feat = grid[top_p: top_p + h_p, left_p: left_p + w_p, :]
            yield feat.permute(2, 0, 1).contiguous()  # (D, h_p, w_p)

    def to(self, device) -> "VGGTExtractor":
        self.model.to(device)
        return self
