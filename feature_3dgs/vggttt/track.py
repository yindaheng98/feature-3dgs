from feature_3dgs.vggt.track import VGGTrackExtractor

from .extractor import VGGTTTExtractor


class VGGTTTTrackExtractor(VGGTTTExtractor, VGGTrackExtractor):
    """Feature extractor based on VGG-T3 aggregator + TrackHead DPT feature extractor.

    Preprocessing is identical to ``VGGTTTExtractor``: center-pad to square
    (black) -> bicubic resize to *img_load_resolution* -> bilinear resize to
    518x518 -> TTT aggregator. Then the TrackHead's DPT feature extractor turns
    aggregated tokens into per-image feature maps at half resolution, which
    are cropped to the valid (non-padded) region -> (C, h_f, w_f).

    VGG-T3 requires multiple images (multi-view aggregation). Use
    ``extract_all`` instead of ``__call__``.
    """

    def __init__(self, model, feature_dim: int, img_load_resolution: int = 1024):
        if model.track_head is None:
            raise ValueError("The loaded VGG-T3 model does not include a track head")
        super().__init__(
            model=model,
            feature_dim=feature_dim,
            img_load_resolution=img_load_resolution,
        )
