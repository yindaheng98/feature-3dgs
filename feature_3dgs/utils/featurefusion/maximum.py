import tqdm
import torch
import torch.nn.functional as F
from gaussian_splatting import GaussianModel
from feature_3dgs.extractor.dataset import FeatureCameraDataset

from .fusion import feature_fusion


@torch.no_grad()
def feature_fusion_alpha_max(
    gaussians: GaussianModel,
    dataset: FeatureCameraDataset,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    fusion_alpha_threshold: float = 0.,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-view feature maps by keeping the highest-scoring observation.

    For every camera in *dataset*, the feature map is linearly projected with
    (*weight*, *bias*) and then fused onto the Gaussians via the custom
    rasteriser.  Each Gaussian retains the feature vector from the view where
    its rendering alpha is the highest, acting as a "winner-takes-all" fusion.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H, W)``.
        weight: ``(C_encoded, C_ext)`` - linear projection weight.
        bias: ``(C_encoded,)`` or *None* - linear projection bias.
        fusion_alpha_threshold: passed through to :func:`feature_fusion`.

    Returns:
        A tuple ``(result, best_score)`` where *result* has shape
        ``(N, C_encoded)`` containing the feature vector from the best view
        for each Gaussian, and *best_score* has shape ``(N,)`` with the
        corresponding alpha value (``-inf`` if the Gaussian was never
        observed).
    """
    N = gaussians.get_xyz.shape[0]
    C_encoded = weight.shape[0]
    device = gaussians._xyz.device

    best_score = torch.full((N,), -float('inf'), device=device, dtype=torch.float32)
    # best_score = torch.full((N,), 0, device=device, dtype=torch.int32) # debug: use pixhit count as score instead of alpha
    result = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)

    for idx in tqdm.tqdm(range(len(dataset)), desc="Fusing features (max)"):
        camera = dataset[idx]
        feature_map = camera.custom_data['feature_map']
        if feature_map is None:
            continue

        C_ext, height, width = feature_map.shape
        fm = feature_map.permute(1, 2, 0).reshape(-1, C_ext)  # (H*W, C_ext)
        fm = F.linear(fm, weight, bias)                       # (H*W, C_encoded)
        fm = fm.reshape(height, width, -1)                    # (H, W, C_encoded)

        target_height, target_width = int(camera.image_height), int(camera.image_width)
        if height != target_height or width != target_width:
            fm = F.interpolate(fm.permute(2, 0, 1).unsqueeze(0), size=(target_height, target_width), mode='nearest').squeeze(0).permute(1, 2, 0)

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
