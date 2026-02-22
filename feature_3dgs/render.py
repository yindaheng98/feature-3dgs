import os
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from gaussian_splatting.utils import psnr
from feature_3dgs import SemanticGaussianModel, get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.prepare import prepare_dataset_and_decoder, prepare_gaussians
from feature_3dgs.utils import pca_transform_params


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


def colorize_feature_map(
    feature_map: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Project a feature map to RGB via linear projection + sigmoid.

    Args:
        feature_map: ``(D, H, W)`` tensor.
        weight: ``(3, D)`` linear weight.
        bias:   ``(3,)`` linear bias.

    Returns:
        ``(3, H, W)`` tensor with values in ``[0, 1]``.
    """
    D, H, W = feature_map.shape
    x = feature_map.reshape(D, -1).permute(1, 0)                  # (H*W, D)
    x = F.linear(x, weight.to(x.device), bias.to(x.device))       # (H*W, 3)
    return torch.sigmoid(x.reshape(H, W, 3).permute(2, 0, 1) * 2.0)


def rendering(dataset: FeatureCameraDataset, gaussians: SemanticGaussianModel, save: str) -> None:
    weight, bias = pca_transform_params(dataset, n_components=3)

    os.makedirs(save, exist_ok=True)
    dataset.save_cameras(os.path.join(save, "cameras.json"))
    render_path = os.path.join(save, "renders")
    gt_path = os.path.join(save, "gt")
    feature_map_path = os.path.join(save, "semantics")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(feature_map_path, exist_ok=True)

    pbar = tqdm(dataset, dynamic_ncols=True, desc="Rendering")
    with open(os.path.join(save, "quality_semantic.csv"), "w") as f:
        f.write("name,psnr,psnr_pca\n")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = out["feature_map"]  # (D, H, W)
        gt = camera.custom_data['feature_map']  # (D, H, W)
        psnr_value = psnr(rendering, gt).mean().item()
        rendering_rgb = colorize_feature_map(rendering, weight, bias)
        gt_rgb = colorize_feature_map(gt, weight, bias)
        psnr_rgb_value = psnr(rendering_rgb, gt_rgb).mean().item()
        pbar.set_postfix({"PSNR": psnr_value, "PSNR PCA": psnr_rgb_value})
        with open(os.path.join(save, "quality_semantic.csv"), "a") as f:
            f.write('{0:05d}'.format(idx) + f",{psnr_value},{psnr_rgb_value}\n")
        torchvision.utils.save_image(rendering_rgb, os.path.join(render_path, '{0:05d}'.format(idx) + "_semantic.png"))
        torchvision.utils.save_image(gt_rgb, os.path.join(gt_path, '{0:05d}'.format(idx) + "_semantic.png"))
        out = gaussians.forward_linear_projection(camera, weight=weight, bias=bias)
        rendering = torch.sigmoid(out["feature_map"] * 2.0)
        gt = colorize_feature_map(camera.custom_data['feature_map'], weight, bias)
        torchvision.utils.save_image(rendering, os.path.join(feature_map_path, '{0:05d}'.format(idx) + ".png"))


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
