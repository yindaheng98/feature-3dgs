from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F

from .extractor import VGGTExtractor, RESOLUTION, padding_square, compute_patch_grid_size


class VGGTrackExtractor(VGGTExtractor):
    """Feature extractor based on VGGT aggregator + TrackHead DPT feature extractor.

    Preprocessing is identical to ``VGGTExtractor``: center-pad to square
    (black) -> bicubic resize to *img_load_resolution* -> bilinear resize to
    518x518 -> aggregator.  Then the TrackHead's DPT feature extractor turns
    aggregated tokens into per-image feature maps at half resolution, which
    are cropped to the valid (non-padded) region -> (C, h_f, w_f).

    VGGT requires multiple images (multi-view aggregation).  Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model, img_load_resolution: int = 1024):
        super().__init__(model=model, img_load_resolution=img_load_resolution)
        self.feature_extractor = model.track_head.feature_extractor

    @torch.no_grad()
    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Extract DPT track features from a sequence of images.

        Args:
            images: Iterable of (C, H, W) tensors in [0, 1] range.

        Yields:
            Per-image feature map of shape (C, h_f, w_f), with padded
            regions cropped so only the original image content is kept.
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

        # 3. Extract per-image feature maps via TrackHead's DPT feature extractor
        with torch.cuda.amp.autocast(dtype=dtype):
            feature_maps = self.feature_extractor(aggregated_tokens_list, batch, ps_idx)
        # feature_maps: (1, S, C, H_feat, W_feat)  e.g. (1, S, 128, 259, 259)
        feature_maps = feature_maps[0]              # (S, C, H_feat, W_feat)
        H_feat, W_feat = feature_maps.shape[-2:]

        # 4. Crop valid region for each image
        for i, (H, W) in enumerate(orig_sizes):
            h_f, w_f = compute_patch_grid_size(H, W, feat_size=H_feat)
            top_f = (H_feat - h_f) // 2
            left_f = (W_feat - w_f) // 2
            feat = feature_maps[i, :, top_f: top_f + h_f, left_f: left_f + w_f]
            yield feat.contiguous()                 # (C, h_f, w_f)
