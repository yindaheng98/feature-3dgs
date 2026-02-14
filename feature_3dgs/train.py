import os
from typing import Tuple
import torch
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.train import save_cfg_args, training
from feature_3dgs import SemanticGaussianModel
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.prepare import prepare_dataset_and_decoder, prepare_gaussians, prepare_trainer, modes


def prepare_training(
        name: str, sh_degree: int, mode: str, source: str, embed_dim: int, device: str, dataset_cache_device: str = None,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None,
        load_mask=True, load_depth=True, load_semantic: bool = True,
        configs={}, **kwargs) -> Tuple[FeatureCameraDataset, SemanticGaussianModel, AbstractTrainer]:
    dataset, decoder = prepare_dataset_and_decoder(
        name=name, source=source, embed_dim=embed_dim, device=device, dataset_cache_device=dataset_cache_device,
        trainable_camera=trainable_camera, load_camera=load_camera,
        load_mask=load_mask, load_depth=load_depth, **kwargs)
    gaussians = prepare_gaussians(
        decoder=decoder, sh_degree=sh_degree, source=source, device=device,
        trainable_camera=trainable_camera, load_ply=load_ply, load_semantic=load_semantic)
    trainer = prepare_trainer(
        gaussians=gaussians, dataset=dataset, mode=mode,
        trainable_camera=trainable_camera, configs=configs)
    return dataset, gaussians, trainer


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--embed_dim", required=True, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("-l", "--load_ply", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--no_load_semantic", action="store_true")
    parser.add_argument("--mode", choices=sorted(modes.keys()), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("--empty_cache_every_step", action='store_true')
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer = prepare_training(
        name=args.name, sh_degree=args.sh_degree, mode=args.mode,
        source=args.source, embed_dim=args.embed_dim,
        device=args.device, dataset_cache_device=args.dataset_cache_device,
        trainable_camera="camera" in args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera,
        load_mask=not args.no_image_mask, load_depth=not args.no_depth_data, load_semantic=not args.no_load_semantic,
        configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        device=args.device, empty_cache_every_step=args.empty_cache_every_step)
