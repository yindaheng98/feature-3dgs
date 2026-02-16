import os
from typing import List, Tuple
import torch
import torchvision
from sklearn.decomposition import PCA
from tqdm import tqdm
from feature_3dgs import SemanticGaussianModel, get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.prepare import prepare_dataset_and_decoder, prepare_gaussians


def prepare_rendering(
        name: str, sh_degree: int, source: str, embed_dim: int, device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None,
        load_mask=True, extractor_configs={}) -> Tuple[FeatureCameraDataset, SemanticGaussianModel]:
    dataset, decoder = prepare_dataset_and_decoder(
        name=name, source=source, embed_dim=embed_dim, device=device, dataset_cache_device=dataset_cache_device,
        trainable_camera=trainable_camera, load_camera=load_camera,
        load_mask=load_mask, load_depth=False, configs=extractor_configs)
    gaussians = prepare_gaussians(
        decoder=decoder, sh_degree=sh_degree, source=source, device=device,
        trainable_camera=trainable_camera, load_ply=load_ply)
    return dataset, gaussians


def build_pca(
    dataset: FeatureCameraDataset,
    gaussians: SemanticGaussianModel = None,
) -> Tuple[PCA, List[torch.Tensor]]:
    """Collect feature maps and fit a PCA with 3 components.

    If *gaussians* is not ``None``, each view is rendered through the model and
    the **rendered** feature maps are used for PCA fitting.

    If *gaussians* is ``None``, feature maps are taken directly from the
    dataset (i.e. extractor output).

    Returns:
        pca: fitted ``PCA(n_components=3, whiten=True)``
        feature_maps: list of ``(D, H, W)`` CPU tensors
    """
    all_features: List[torch.Tensor] = []
    feature_maps: List[torch.Tensor] = []

    desc = ("Rendering" if gaussians is not None else "Extracting") + " features for PCA fitting"
    for idx in tqdm(range(len(dataset)), dynamic_ncols=True, desc=desc):
        if gaussians is not None:
            camera = dataset.cameras[idx]  # base camera - skip feature extraction
            out = gaussians(camera)
            feature_map = out["feature_map"].cpu()  # (D, H, W)
        else:
            feature_map = dataset[idx].custom_data['feature_map'].cpu()  # (D, H, W)

        feature_maps.append(feature_map)
        D = feature_map.shape[0]
        all_features.append(feature_map.reshape(D, -1).permute(1, 0))  # (H*W, D)

    all_features_np = torch.cat(all_features, dim=0).numpy()  # (N_total, D)
    del all_features

    pca = PCA(n_components=3, whiten=True)
    pca.fit(all_features_np)
    del all_features_np

    return pca, feature_maps


def colorize_feature_map(feature_map: torch.Tensor, pca: PCA) -> torch.Tensor:
    """Project a single feature map to an RGB image via PCA + sigmoid.

    Args:
        feature_map: ``(D, H, W)`` tensor (CPU).
        pca: a fitted ``PCA`` with ``n_components=3``.

    Returns:
        ``(H, W, 3)`` tensor with values in ``[0, 1]``.
    """
    D, H, W = feature_map.shape
    features = feature_map.reshape(D, -1).permute(1, 0).cpu().numpy()  # (H*W, D)
    projected = torch.from_numpy(pca.transform(features)).view(H, W, 3)
    # Vibrant colours: multiply by 2 and pass through sigmoid
    return torch.sigmoid(projected * 2.0).permute(2, 0, 1)


def rendering(dataset: FeatureCameraDataset, gaussians: SemanticGaussianModel, save: str) -> None:
    pca, gt_feature_maps = build_pca(dataset, gaussians=None)

    os.makedirs(save, exist_ok=True)
    dataset.save_cameras(os.path.join(save, "cameras.json"))
    render_path = os.path.join(save, "renders")
    gt_path = os.path.join(save, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    pbar = tqdm(dataset, dynamic_ncols=True, desc="Rendering")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = colorize_feature_map(out["feature_map"], pca)  # (H, W, 3)
        gt = colorize_feature_map(gt_feature_maps[idx], pca)  # (H, W, 3)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_semantic.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + "_semantic.png"))


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
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            name=args.name, sh_degree=args.sh_degree,
            source=args.source, embed_dim=args.embed_dim,
            device=args.device, dataset_cache_device=args.dataset_cache_device,
            trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=not args.no_image_mask,
            extractor_configs=extractor_configs)
        rendering(dataset, gaussians, save)
