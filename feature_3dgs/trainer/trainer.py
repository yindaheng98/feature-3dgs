from typing import Callable
import torch
from gaussian_splatting.utils import l1_loss
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer, BaseTrainer
from gaussian_splatting import Camera
from feature_3dgs import SemanticGaussianModel


class SemanticTrainer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            semantic_lr=0.001,
            semantic_decoder_lr=0.0001,
            semantic_loss_weight=1.0,
            semantic_mask_mode="none",
    ):
        super().__init__(base_trainer=base_trainer)
        model = self.model
        assert isinstance(model, SemanticGaussianModel), "SemanticTrainer's model must be a SemanticGaussianModel"
        self.optimizer.add_param_group({"lr": semantic_lr, "params": model._semantic_features, "name": "semantic"})
        self.optimizer.add_param_group({"lr": semantic_decoder_lr, "params": model.get_decoder.parameters(), "name": "semantic_decoder"})
        self.semantic_loss_weight = semantic_loss_weight
        self.mask_mode = semantic_mask_mode

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)
        render = out['feature_map']
        gt = camera.custom_data['feature_map']
        mask = camera.ground_truth_image_mask
        match self.mask_mode:
            case "none":
                pass
            case "ignore":
                assert mask is not None, "Mask is required for 'ignore' mask policy"
                render = render * mask.unsqueeze(0)
                gt = gt * mask.unsqueeze(0)
            case _:
                raise ValueError(f"Unknown mask policy: {self.mask_mode}")
        semantic_loss = l1_loss(render, gt)
        return loss + semantic_loss * self.semantic_loss_weight


def SemanticTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: SemanticGaussianModel,
        scene_extent: float,
        *args,
        semantic_lr=0.001,
        semantic_decoder_lr=0.0001,
        semantic_loss_weight=1.0,
        **kwargs) -> SemanticTrainer:
    return SemanticTrainer(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        semantic_lr=semantic_lr,
        semantic_decoder_lr=semantic_decoder_lr,
        semantic_loss_weight=semantic_loss_weight,
    )


def BaseSemanticTrainer(
        model: SemanticGaussianModel,
        scene_extent: float,
        semantic_lr=0.001,
        semantic_decoder_lr=0.0001,
        semantic_loss_weight=1.0,
        *args, **kwargs) -> SemanticTrainer:
    return SemanticTrainerWrapper(
        lambda model, *args, **kwargs: BaseTrainer(model, scene_extent, *args, **kwargs),
        model=model,
        scene_extent=scene_extent,
        semantic_lr=semantic_lr,
        semantic_decoder_lr=semantic_decoder_lr,
        semantic_loss_weight=semantic_loss_weight,
        *args, **kwargs,
    )
