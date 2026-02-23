import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from gaussian_splatting import Camera, GaussianModel
from feature_3dgs import SemanticGaussianModel, get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.render import prepare_rendering
from feature_3dgs.segmentation2d import get_feature


def compute_3d_similarity(query: torch.Tensor, gaussians: SemanticGaussianModel, batch_size: int = 2 ** 16) -> torch.Tensor:
    """Cosine similarity between *query* (D,) and every Gaussian's decoded semantic.

    Decoding and similarity are computed in batches of *batch_size* to avoid
    materialising the full (N, C_feat) tensor on the GPU at once.

    Returns a (N,) tensor with values in [-1, 1].
    """
    encoded = gaussians.get_encoded_semantics          # (N, C_enc)
    decoder = gaussians.get_decoder
    q = query.unsqueeze(0)                             # (1, D)
    parts = []
    for start in range(0, encoded.shape[0], batch_size):
        chunk = encoded[start:start + batch_size]      # (B, C_enc)
        feat = decoder.transform_features(chunk)       # (B, C_feat)
        sim = F.cosine_similarity(q, feat, dim=1)      # (B,)
        parts.append(sim)
        del feat
    return torch.cat(parts, dim=0)


def similarity_to_colors(similarity: torch.Tensor) -> torch.Tensor:
    """Map similarity values in [-1, 1] to RGB via a matplotlib colormap.

    Returns a float tensor of shape (N, 3) in [0, 1].
    """
    device = similarity.device
    sim_np = similarity.detach().cpu().numpy()
    normalized = np.clip((sim_np + 1.0) / 2.0, 0.0, 1.0)
    cmap = plt.get_cmap('viridis')
    rgb = cmap(normalized)[:, :3]
    return torch.from_numpy(rgb.copy()).float().to(device)


def render_heatmap(gaussians: SemanticGaussianModel, camera: Camera, heatmap_colors: torch.Tensor) -> torch.Tensor:
    """Render the 3D model with per-Gaussian heatmap *heatmap_colors* (N, 3).

    Returns the rendered image (3, H, W) clamped to [0, 1].
    """
    out = gaussians.render_encoded(
        viewpoint_camera=camera,
        means3D=gaussians.get_xyz,
        opacity=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shs=None,
        semantic_features=gaussians.get_encoded_semantics,
        colors_precomp=heatmap_colors,
    )
    return out['render']


def render_segmented(gaussians: SemanticGaussianModel, camera: Camera, masked_opacity: torch.Tensor) -> torch.Tensor:
    """Render only the Gaussians where *masked_opacity* (N,) is non-zero.

    Below-threshold Gaussians get zero opacity so they become fully transparent.
    Returns the rendered image (3, H, W).
    """
    out = gaussians.render_encoded(
        viewpoint_camera=camera,
        means3D=gaussians.get_xyz,
        opacity=masked_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shs=gaussians.get_features,
        semantic_features=gaussians.get_encoded_semantics,
    )
    return out['render']


def save_segmented_ply(gaussians: SemanticGaussianModel, mask: torch.Tensor, path: str) -> None:
    """Save only the Gaussians where *mask* (N,) is True to a PLY file.

    Constructs a new SemanticGaussianModel containing the masked subset and
    delegates to its ``save_ply``, which also writes ``.semantic.pt`` and
    ``.decoder.pt`` side-cars.
    """
    m = mask
    seg = SemanticGaussianModel(sh_degree=gaussians.max_sh_degree, decoder=gaussians._decoder)
    seg._xyz = nn.Parameter(gaussians._xyz.detach()[m])
    seg._features_dc = nn.Parameter(gaussians._features_dc.detach()[m])
    seg._features_rest = nn.Parameter(gaussians._features_rest.detach()[m])
    seg._opacity = nn.Parameter(gaussians._opacity.detach()[m])
    seg._scaling = nn.Parameter(gaussians._scaling.detach()[m])
    seg._rotation = nn.Parameter(gaussians._rotation.detach()[m])
    seg._encoded_semantics = nn.Parameter(gaussians._encoded_semantics.detach()[m])
    seg.active_sh_degree = gaussians.active_sh_degree

    os.makedirs(os.path.dirname(path), exist_ok=True)
    seg.save_ply(path)
    print(f"Saved {int(mask.sum())} / {mask.shape[0]} points to {path}")


def save_segmentation(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset, query: torch.Tensor, threshold: float, save_dir: str, save_ply: str) -> None:
    """Render the 3D model from every dataset viewpoint and save."""
    os.makedirs(save_dir, exist_ok=True)

    sim_3d = compute_3d_similarity(query, gaussians)
    heatmap_colors = similarity_to_colors(sim_3d).to(gaussians.get_xyz.device)
    mask_3d = sim_3d > threshold
    masked_opacity = gaussians.get_opacity.clone()
    masked_opacity[~mask_3d] = 0.0

    for idx in tqdm(range(len(dataset)), desc="Rendering 3D segmentation"):
        camera = dataset[idx]

        img_heatmap = render_heatmap(gaussians, camera, heatmap_colors)
        torchvision.utils.save_image(img_heatmap, os.path.join(save_dir, f"{idx:05d}_3d_heatmap.png"))

        img_seg = render_segmented(gaussians, camera, masked_opacity)
        torchvision.utils.save_image(img_seg, os.path.join(save_dir, f"{idx:05d}_3d_segmented.png"))

    save_segmented_ply(gaussians, mask_3d, save_ply)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--name", choices=get_available_extractor_decoders(), required=True, type=str)
    parser.add_argument("--embed_dim", required=True, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("-e", "--option_extractor", default=[], action='append', type=str)
    parser.add_argument("-n", "--image_index", required=True, type=int)
    parser.add_argument("-x", required=True, type=int)
    parser.add_argument("-y", required=True, type=int)
    parser.add_argument("-t", "--threshold", required=True, type=float)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration), f"segmentation{args.image_index:05d}x{args.x}y{args.y}t{args.threshold:.2f}")
    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            name=args.name, sh_degree=args.sh_degree,
            source=args.source, embed_dim=args.embed_dim,
            device=args.device, dataset_cache_device=args.dataset_cache_device,
            trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=not args.no_image_mask,
            extractor_configs=extractor_configs,
        )
        feature = get_feature(dataset, args.image_index, args.x, args.y)
        save_ply = os.path.join(save, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
        save_segmentation(gaussians, dataset, feature, args.threshold, save, save_ply)
