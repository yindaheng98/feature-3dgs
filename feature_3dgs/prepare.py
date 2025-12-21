import os
import random
import shutil
from typing import List, Tuple
import torch
from tqdm import tqdm
from argparse import Namespace
from gaussian_splatting.prepare import basemodes, shliftmodes, colmap_init
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from gaussian_model import FeatureGaussian
from trainer.base import FeatureTrainer
from dataset.dataset import FeatureDataset

def prepare_feature_gaussians(
        sh_degree: int,
        source: str,
        device: str,
        trainable_camera: bool = False,
        load_ply: str = None
) -> FeatureGaussian:
    assert trainable_camera == False, "Camera trainable not implemented!"
    gaussians = FeatureGaussian(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians

# TODO
def prepare_feature_trainer(
        gaussians: FeatureGaussian,
        dataset: FeatureDataset,
        mode: str,
        trainable_camera: bool = False,
        load_ply: str = None,
        with_scale_reg=False,
        configs={}
) -> FeatureTrainer:
    assert trainable_camera == False, "Camera trainable not implemented!"
    modes = shliftmodes if load_ply else basemodes
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    if trainable_camera:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            dataset=dataset,
            **configs
        )
    else:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            **configs
        )
    return trainer

# TODO:
def prepare_feature_dataset(
        source: str,
        device: str,
        trainable_camera: bool = False,
        load_camera: str = None,
        load_mask=True,
        load_depth=True
) -> FeatureDataset:
    pass
