import os
import random
import shutil
from typing import List, Tuple
import torch
from tqdm import tqdm
from argparse import Namespace
from gaussian_splatting.train import training
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset import CameraDataset
from feature_3dgs import FeatureGaussian
from feature_3dgs.trainer import FeatureTrainer
from feature_3dgs.prepare import basemodes, shliftmodes, prepare_feature_dataset, prepare_feature_gaussians, prepare_feature_trainer, prepare_feature_extractor

def prepare_training(
        sh_degree: int, source: str, device: str, mode: str,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None,
        load_mask=True, load_depth=True,
        with_scale_reg=False, configs={}
) -> Tuple[CameraDataset, FeatureGaussian, FeatureTrainer]:
    dataset = prepare_feature_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_mask=load_mask, load_depth=load_depth)
    gaussians = prepare_feature_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    decoder = prepare_feature_extractor()
    trainer = prepare_feature_trainer(gaussians=gaussians, decoder=decoder, dataset=dataset, mode=mode, trainable_camera=trainable_camera, load_ply=load_ply, with_scale_reg=with_scale_reg, configs=configs)
    return dataset, gaussians, trainer


# TODO
def save_cfg_args(destination: str, sh_degree: int, source: str):
    os.makedirs(destination, exist_ok=True)
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))


# TODO
if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("-l", "--load_ply", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_depth_data", action="store_true")
    parser.add_argument("--with_scale_reg", action="store_true")
    parser.add_argument("--mode", choices=sorted(list(set(list(basemodes.keys()) + list(shliftmodes.keys())))), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--empty_cache_every_step", action='store_true')
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode, trainable_camera="camera" in args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera,
        load_mask=not args.no_image_mask, load_depth=not args.no_depth_data,
        with_scale_reg=args.with_scale_reg, configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        device=args.device, empty_cache_every_step=args.empty_cache_every_step)
