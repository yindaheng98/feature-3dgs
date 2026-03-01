import math
import tqdm
import torch
from gaussian_splatting import GaussianModel, Camera
from feature_3dgs.extractor.dataset import FeatureCameraDataset
from typing import Callable

from .pickup import feature_pickup

BLOCK_X = 16  # submodules/featurepickup/cuda_rasterizer/config.h
BLOCK_Y = 16  # submodules/featurepickup/cuda_rasterizer/config.h


@torch.no_grad()
def feature_pickup_alpha_max(
    gaussians: GaussianModel,
    dataset: FeatureCameraDataset,
    encode_feature_pixels: Callable[[torch.Tensor], torch.Tensor],
    fusion_alpha_threshold: float = 0.,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick per-Gaussian features from highest-alpha rendered patch.

    For every camera in *dataset*, patch features are transformed by
    ``encode_feature_pixels(feature_map)`` and the custom pickup rasteriser
    returns per-instance patch coordinates and alpha.  Each Gaussian keeps the
    feature vector from the instance with the highest alpha.

    Args:
        gaussians: GaussianModel whose geometry is used for splatting.
        dataset: FeatureCameraDataset - each item must carry
            ``camera.custom_data['feature_map']`` of shape ``(C_ext, H_p, W_p)``.
        encode_feature_pixels: callable that maps extractor patch features
            ``(C_ext, H_p, W_p)`` to encoded patch features
            ``(C_encoded, H_p, W_p)``.
        fusion_alpha_threshold: passed through to :func:`feature_pickup`.

    Returns:
        A tuple ``(result, best_score)`` where *result* has shape
        ``(N, C_encoded)`` containing the feature vector from the best patch
        for each Gaussian, and *best_score* has shape ``(N,)`` with the
        corresponding alpha value (``-inf`` if the Gaussian was never observed).
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians._xyz.device

    best_score = torch.full((N,), -float('inf'), device=device, dtype=torch.float32)
    # best_score = torch.full((N,), 0, device=device, dtype=torch.int32) # debug: use pixhit count as score instead of alpha
    result = None

    for idx in tqdm.tqdm(range(len(dataset)), desc="Picking features (max alpha)"):
        camera = dataset[idx]
        feature_map = camera.custom_data['feature_map']
        if feature_map is None:
            continue

        fm = encode_feature_pixels(feature_map).permute(1, 2, 0)  # (H_p, W_p, C_encoded)
        assert math.ceil(camera.image_height / BLOCK_Y) == fm.shape[0] and math.ceil(camera.image_width / BLOCK_X) == fm.shape[1], (
            f"Camera image size {(int(camera.image_height), int(camera.image_width))} and encoded feature_map size {fm.shape[:2]} mismatch for block size {(BLOCK_Y, BLOCK_X)}.")
        if result is None:
            C_encoded = fm.shape[-1]
            result = torch.zeros((N, C_encoded), device=device, dtype=torch.float32)

        _, tiles_idx, features_alpha, pixhit, features_idx = feature_pickup(
            gaussians, camera, fusion_alpha_threshold,
        )
        del camera, _

        unique_idx, inv = torch.unique(features_idx, return_inverse=True)  # unique_idx[inv] == features_idx
        group_max = features_alpha.new_full((unique_idx.numel(),), -float('inf'))
        group_max.scatter_reduce_(0, inv, features_alpha, reduce='amax')  # for each unique_idx, group_max is the max features_alpha among all features_idx that equal to that unique_idx
        is_best = features_alpha >= group_max[inv]  # who is the group_max in features_alpha
        big_val = inv.numel()
        pos = torch.arange(inv.numel(), device=device)
        best_pos = torch.full((unique_idx.numel(),), big_val, device=device, dtype=torch.long)
        best_pos.scatter_reduce_(0, inv, torch.where(is_best, pos, big_val), reduce='amin', include_self=False)
        best_local = best_pos

        tile_x = tiles_idx[best_local, 0].long()
        tile_y = tiles_idx[best_local, 1].long()
        features = fm[tile_y, tile_x]
        score = features_alpha[best_local]
        features_idx = unique_idx
        del fm, tiles_idx, tile_x, tile_y, unique_idx, best_local, inv, group_max, is_best, big_val, pos, best_pos

        better = score > best_score[features_idx]                     # (K,)
        if better.any():
            update_idx = features_idx[better]
            result[update_idx] = features[better]
            best_score[update_idx] = score[better]

        del features, features_alpha, pixhit, features_idx, score, better

    return result, best_score
