from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

RESOLUTION = 518
PATCH_SIZE = 14
FEAT_SIZE = RESOLUTION // PATCH_SIZE  # 37


def compute_square_valid_region(H: int, W: int, square_size: int) -> tuple[int, int, int, int]:
    """Compute the valid region inside a square feature map.

    Works for any square feature map produced from a center-padded image:
    patch token grids (37x37), DPT feature maps (259x259), etc.

    Args:
        H, W: original image spatial dimensions (before padding).
        square_size: side length of the square feature map.

    Returns:
        (top, left, h, w) of the valid region corresponding to original content.
    """
    max_dim = max(H, W)
    h = max(round(H / max_dim * square_size), 1)
    w = max(round(W / max_dim * square_size), 1)
    top = (square_size - h) // 2
    left = (square_size - w) // 2
    return top, left, h, w


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
        for i, (H, W) in enumerate(orig_sizes):
            grid = patch_tokens[i].view(FEAT_SIZE, FEAT_SIZE, D)
            top_p, left_p, h_p, w_p = compute_square_valid_region(H, W, square_size=FEAT_SIZE)
            feat = grid[top_p: top_p + h_p, left_p: left_p + w_p, :]
            yield feat.permute(2, 0, 1).contiguous()  # (D, h_p, w_p)

    def to(self, device) -> "VGGTExtractor":
        self.model.to(device)
        return self
