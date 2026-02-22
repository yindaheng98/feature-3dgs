import math
from typing import Sequence

import tqdm
import torch
from gaussian_splatting import Camera


def pca_inverse_transform_params(
    cameras: Sequence[Camera],
    n_components: int,
    whiten: bool = False,
    cache_device: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA inverse_transform parameters from camera feature maps.

    Returns ``(weight, bias)`` usable as ``nn.Linear(n_components, D)``
    weights, so that ``self.linear(z)`` (i.e. ``z @ weight.T + bias``)
    reconstructs the original *D*-dimensional features from *n_components*
    latent codes.

    When *whiten* is ``True``, each component is scaled so that the
    matching :func:`pca_transform_params` produces latent codes with
    unit variance.

    Returns:
        weight: ``(D, n_components)``
        bias:   ``(D,)``
    """
    all_features = []
    for camera in tqdm.tqdm(cameras, desc="PCA: collecting features"):
        feature_map = camera.custom_data['feature_map']             # (D, H_p, W_p)
        features = feature_map.reshape(feature_map.shape[0], -1).T  # (N_i, D)
        if cache_device is not None:
            features = features.to(cache_device)
        all_features.append(features)
    all_features = torch.cat(all_features, dim=0)                   # (N_total, D)

    n_total = all_features.shape[0]
    mean = all_features.mean(dim=0)                                 # (D,)
    centered = all_features - mean
    _, S, V = torch.pca_lowrank(centered, q=n_components)           # S: (k,), V: (D, k)

    if whiten:
        V = V * (S / math.sqrt(n_total - 1))                       # (D, k)

    device = cameras[0].custom_data['feature_map'].device
    return V.to(device), mean.to(device)


def pca_inverse_transform_params_to_transform_params(weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert PCA inverse_transform parameters to transform parameters.

    Given ``(weight, bias)`` returned by :func:`pca_inverse_transform_params`,
    returns ``(weight_proj, bias_proj)`` usable as ``nn.Linear(D, n_components)``
    weights, so that ``F.linear(x, weight_proj, bias_proj)`` produces the same
    latent codes as :func:`pca_transform_params` would.

    Returns:
        weight_proj: ``(n_components, D)``
        bias_proj:   ``(n_components,)``
    """
    W_dec, mean = weight, bias          # (D, k), (D,)

    sigma_sq = (W_dec ** 2).sum(dim=0)                              # (k,)
    W_proj = W_dec / sigma_sq                                       # (D, k)

    weight = W_proj.T                                               # (k, D)
    bias = -(mean @ W_proj)                                         # (k,)
    return weight, bias


def pca_transform_params(
    cameras: Sequence[Camera],
    n_components: int = 3,
    whiten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA transform (projection) parameters from camera feature maps.

    Returns ``(weight, bias)`` so that ``F.linear(x, weight, bias)``
    projects *D*-dimensional features down to *n_components* dimensions.

    When *whiten* is ``True`` (default), the projected components are
    normalised to unit variance.  The transform / inverse_transform pair
    is consistent::

        z     = F.linear(x, weight, bias)           # transform
        x_hat = F.linear(z, W_inv.T, b_inv)         # inverse_transform
        x_hat ≈ x   (up to rank-k approximation)

    Returns:
        weight: ``(n_components, D)``
        bias:   ``(n_components,)``
    """
    W_dec, mean = pca_inverse_transform_params(cameras, n_components, whiten=whiten)
    return pca_inverse_transform_params_to_transform_params(W_dec, mean)
