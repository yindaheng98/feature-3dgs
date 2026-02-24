import tqdm
import torch
import torch.nn.functional as F
from gaussian_splatting import GaussianModel
from feature_3dgs.extractor.dataset import FeatureCameraDataset

from .fusion import feature_fusion


@torch.no_grad()
def feature_fusion_alpha_avg(
    gaussians: GaussianModel,
    dataset: FeatureCameraDataset,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    fusion_alpha_threshold: float = 0.,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-view feature maps into per-Gaussian encoded semantics.

    For every camera in *dataset*, the feature map is linearly projected with
    (*weight*, *bias*) and then fused onto the Gaussians via the custom
    rasteriser.  The alpha-weighted mean and variance are computed in a
    single streaming pass using the Welford/West online algorithm, which is
    more numerically stable than the naive ``E[X²] - E[X]²`` approach.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H, W)``.
        weight: ``(C_encoded, C_ext)`` - linear projection weight.
        bias: ``(C_encoded,)`` or *None* - linear projection bias.
        fusion_alpha_threshold: passed through to :func:`feature_fusion`.

    Returns:
        A tuple ``(mean, variance)`` where both tensors have shape
        ``(N, C_encoded)``.  *mean* is the alpha-weighted average and
        *variance* is the alpha-weighted population variance per Gaussian
        per channel.
    """
    N = gaussians.get_xyz.shape[0]
    C_encoded = weight.shape[0]
    device = gaussians._xyz.device

    W = torch.zeros((N,), device=device, dtype=torch.float32)
    mean = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)
    M2 = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)

    for idx in tqdm.tqdm(range(len(dataset)), desc="Fusing features"):
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
