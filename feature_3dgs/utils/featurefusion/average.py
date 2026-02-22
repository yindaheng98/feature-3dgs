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
) -> torch.Tensor:
    """Fuse per-view feature maps into per-Gaussian encoded semantics.

    For every camera in *dataset*, the feature map is linearly projected with
    (*weight*, *bias*) and then fused onto the Gaussians via the custom
    rasteriser.  The results are alpha-weighted averaged across all views.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H, W)``.
        weight: ``(C_encoded, C_ext)`` - linear projection weight.
        bias: ``(C_encoded,)`` or *None* - linear projection bias.
        fusion_alpha_threshold: passed through to :func:`feature_fusion`.

    Returns:
        ``(N, C_encoded)`` tensor in the same format as
        ``SemanticGaussianModel._encoded_semantics``.
    """
    N = gaussians.get_xyz.shape[0]
    C_encoded = weight.shape[0]

    feature_sum = torch.zeros((N, C_encoded), device=gaussians._xyz.device, dtype=torch.float32)
    alpha_sum = torch.zeros((N,), device=gaussians._xyz.device, dtype=torch.float32)

    for idx in tqdm.tqdm(range(len(dataset)), desc="Fusing features"):
        camera = dataset[idx]
        feature_map = camera.custom_data['feature_map']
        if feature_map is None:
            continue

        C_ext, H, W = feature_map.shape
        fm = feature_map.permute(1, 2, 0).reshape(-1, C_ext)   # (H*W, C_ext)
        fm = F.linear(fm, weight, bias)                          # (H*W, C_encoded)
        fm = fm.reshape(H, W, -1)        # (C_encoded, H, W)

        target_H, target_W = int(camera.image_height), int(camera.image_width)
        if H != target_H or W != target_W:
            fm = F.interpolate(fm.permute(2, 0, 1).unsqueeze(0), size=(target_H, target_W),
                               mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        _, features, features_alpha, _, features_idx = feature_fusion(
            gaussians, camera, fm, fusion_alpha_threshold,
        )

        feature_sum[features_idx] += features
        alpha_sum[features_idx] += features_alpha

    valid = alpha_sum > 1e-12
    result = torch.zeros_like(feature_sum)
    result[valid] = feature_sum[valid] / alpha_sum[valid].unsqueeze(-1)
    return result
