import tqdm
import torch
from gaussian_splatting import GaussianModel, Camera
from feature_3dgs.extractor.dataset import FeatureCameraDataset
from typing import Callable

from .fusion import feature_fusion


@torch.no_grad()
def feature_fusion_alpha_avg(
    gaussians: GaussianModel,
    dataset: FeatureCameraDataset,
    encode_feature_map: Callable[[torch.Tensor, Camera], torch.Tensor],
    fusion_alpha_threshold: float = 0.,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-view feature maps into per-Gaussian encoded semantics.

    For every camera in *dataset*, the feature map is transformed by
    ``encode_feature_map(feature_map, camera)`` and then fused onto the
    Gaussians via the custom rasteriser.  The alpha-weighted mean and
    variance are computed in a single streaming pass using the Welford/West
    online algorithm, which is more numerically stable than the naive
    ``E[X²] - E[X]²`` approach.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H, W)``.
        encode_feature_map: callable that maps extractor feature map
            ``(C_ext, H, W)`` to encoded map ``(C_encoded, H, W)`` for a camera.
        fusion_alpha_threshold: passed through to :func:`feature_fusion`.

    Returns:
        A tuple ``(mean, variance)`` where both tensors have shape
        ``(N, C_encoded)``.  *mean* is the alpha-weighted average and
        *variance* is the alpha-weighted population variance per Gaussian
        per channel.
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians._xyz.device

    W = torch.zeros((N,), device=device, dtype=torch.float32)
    mean = None
    M2 = None

    for idx in tqdm.tqdm(range(len(dataset)), desc="Fusing features"):
        camera = dataset[idx]
        feature_map = camera.custom_data['feature_map']
        if feature_map is None:
            continue

        fm = encode_feature_map(feature_map, camera).permute(1, 2, 0)  # (H, W, C_encoded)
        assert fm.shape[0] == camera.image_height and fm.shape[1] == camera.image_width, f"Encoded feature map size {fm.shape[:2]} does not match camera image size {(camera.image_height, camera.image_width)}."
        if mean is None or M2 is None:
            C_encoded = fm.shape[-1]
            mean = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)
            M2 = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)

        _, features, features_alpha, _, features_idx = feature_fusion(
            gaussians, camera, fm, fusion_alpha_threshold,
        )
        del fm, _

        # Welford/West online weighted update
        w = features_alpha                                        # (K,)
        features.div_(w.unsqueeze(-1).clamp(min=1e-12))           # (K, C_encoded) in-place: αx → x

        W_old = W[features_idx]                                   # (K,)
        W_new = W_old + w                                         # (K,)
        old_mean = mean[features_idx]                             # (K, C)
        features.sub_(old_mean)                                   # in-place: x → δ

        # M2 += (w·W_old / W_new)·δ²  — avoids allocating x and new_mean together
        scale = ((w * W_old) / W_new.clamp(min=1e-12)).unsqueeze(-1)  # (K, 1)
        delta_sq = features.square()                              # (K, C)
        delta_sq.mul_(scale)                                      # in-place
        M2[features_idx] += delta_sq
        del delta_sq, scale

        # μ_new = μ_old + (w / W_new)·δ
        r = (w / W_new.clamp(min=1e-12)).unsqueeze(-1)            # (K, 1)
        mean[features_idx] = old_mean.addcmul_(r, features)
        W[features_idx] = W_new
        del old_mean, features, features_alpha, features_idx, W_old, W_new, r, w

    valid = W > 1e-12
    variance = torch.zeros_like(mean)
    variance[valid] = (M2[valid] / W[valid].unsqueeze(-1)).clamp(min=0.)

    return mean, variance
