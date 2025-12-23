from gaussian_splatting.prepare import basemodes, shliftmodes, colmap_init, prepare_trainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from gaussian_splatting.dataset import CameraDataset
from feature_3dgs import FeatureGaussian
from feature_3dgs.trainer import FeatureTrainer


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
        decoder: nn.Module,
        *args, **kwargs) -> FeatureTrainer:
    trainer = prepare_trainer(gaussians=gaussians, *args, **kwargs)
    feature_trainer = FeatureTrainer(base_trainer=trainer, decoder=decoder)
    return feature_trainer

# TODO
def prepare_feature_extractor():
    pass