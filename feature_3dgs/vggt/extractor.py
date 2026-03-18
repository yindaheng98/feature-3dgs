from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor


def padding(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad image so that H and W are multiples of patch_size."""
    _, h, w = image.shape  # (C, H, W)
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return image


class VGGTExtractor(AbstractFeatureExtractor):
    """Feature extractor based on VGGT aggregator.

    Preprocesses images so that the output patch grid matches an effective
    ``PATCH_SIZE=16`` (same as DINOv3), despite VGGT's native 14-pixel patches.

    Pipeline: reflect-pad to 16-multiples -> bicubic scale by 14/16
    -> center-pad to square (black, patch-aligned) -> aggregator
    -> crop valid tokens -> (D, h_p, w_p) feature map.

    VGGT requires multiple images (multi-view aggregation).  Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model, patch_size: int, vggt_patch_size: int):
        self.model = model
        self.patch_size = patch_size
        self.vggt_patch_size = vggt_patch_size
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
            Per-image feature map of shape (D, h_p, w_p) where
            h_p = ceil(H / patch_size), w_p = ceil(W / patch_size).
        """
        P = self.patch_size
        VP = self.vggt_patch_size

        # 1. Reflect-pad each image to 16-multiples and record patch grid sizes
        padded_images = []
        patch_sizes = []
        for image in images:
            x = padding(image, P)
            _, h_pad, w_pad = x.shape
            padded_images.append(x)
            patch_sizes.append((h_pad // P, w_pad // P))

        # 2. Global square size determined by all images
        sq_patches = max(max(h_p, w_p) for h_p, w_p in patch_sizes)
        sq_size = sq_patches * VP

        # 3. Bicubic resize and center-pad each image to the global square
        frames = []
        offsets = []
        for x, (h_p, w_p) in zip(padded_images, patch_sizes):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(h_p * VP, w_p * VP),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            top_patches = (sq_patches - h_p) // 2
            left_patches = (sq_patches - w_p) // 2
            top = top_patches * VP
            left = left_patches * VP

            frame = x.new_zeros(3, sq_size, sq_size)
            frame[:, top: top + h_p * VP, left: left + w_p * VP] = x
            frames.append(frame)
            offsets.append((top_patches, left_patches))

        # 4. Stack and feed to aggregator [B=1, S=N, 3, sq_size, sq_size]
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

        # 5. Extract per-image features from last-layer patch tokens
        #    aggregator output shape: (B, S, P, 2*C) where P includes special tokens
        tokens = aggregated_tokens_list[-1]          # (1, S, P, 2*C)
        patch_tokens = tokens[0, :, ps_idx:, :]      # (S, sq_patches^2, D)
        D = patch_tokens.shape[-1]

        for i, ((h_p, w_p), (top_p, left_p)) in enumerate(zip(patch_sizes, offsets)):
            grid = patch_tokens[i].view(sq_patches, sq_patches, D)
            feat = grid[top_p: top_p + h_p, left_p: left_p + w_p, :]
            yield feat.permute(2, 0, 1).contiguous()  # (D, h_p, w_p)

    def to(self, device) -> "VGGTExtractor":
        self.model.to(device)
        return self
