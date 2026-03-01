import tqdm
import torch
from gaussian_splatting import GaussianModel, Camera
from feature_3dgs.extractor.dataset import FeatureCameraDataset
from typing import Callable

from .fusion import feature_fusion


@torch.no_grad()
def feature_fusion_alpha_max(
    gaussians: GaussianModel,
    dataset: FeatureCameraDataset,
    encode_feature_map: Callable[[torch.Tensor, Camera], torch.Tensor],
    fusion_alpha_threshold: float = 0.,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-view feature maps by keeping the highest-scoring observation.

    For every camera in *dataset*, the feature map is transformed by
    ``encode_feature_map(feature_map, camera)`` and then fused onto the custom
    rasteriser.  Each Gaussian retains the feature vector from the view where
    its rendering alpha is the highest, acting as a "winner-takes-all" fusion.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H, W)``.
        encode_feature_map: callable that maps extractor feature map
            ``(C_ext, H, W)`` to encoded map ``(C_encoded, H, W)`` for a camera.
        fusion_alpha_threshold: passed through to :func:`feature_fusion`.

    Returns:
        A tuple ``(result, best_score)`` where *result* has shape
        ``(N, C_encoded)`` containing the feature vector from the best view
        for each Gaussian, and *best_score* has shape ``(N,)`` with the
        corresponding alpha value (``-inf`` if the Gaussian was never
        observed).
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians._xyz.device

    best_score = torch.full((N,), -float('inf'), device=device, dtype=torch.float32)
    # best_score = torch.full((N,), 0, device=device, dtype=torch.int32) # debug: use pixhit count as score instead of alpha
    result = None

    for idx in tqdm.tqdm(range(len(dataset)), desc="Fusing features (max)"):
        camera = dataset[idx]
        feature_map = camera.custom_data['feature_map']
        if feature_map is None:
            continue

        fm = encode_feature_map(feature_map, camera).permute(1, 2, 0)  # (H, W, C_encoded)
        assert fm.shape[0] == camera.image_height and fm.shape[1] == camera.image_width, (
            f"Encoded feature map size {fm.shape[:2]} does not match camera image size {(camera.image_height, camera.image_width)}.")
        if result is None:
            C_encoded = fm.shape[-1]
            result = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)

        _, features, features_alpha, pixhit, features_idx = feature_fusion(
            gaussians, camera, fm, fusion_alpha_threshold,
        )
        del fm, camera, _

        w = features_alpha                                        # (K,)
        features.div_(w.unsqueeze(-1).clamp(min=1e-12))           # in-place: αx → x

        score = features_alpha
        # score = pixhit # debug: use pixhit count as score instead of alpha
        better = score > best_score[features_idx]                     # (K,)
        if better.any():
            update_idx = features_idx[better]
            result[update_idx] = features[better]
            best_score[update_idx] = score[better]

        del features, features_alpha, pixhit, features_idx, w, score, better

    return result, best_score
