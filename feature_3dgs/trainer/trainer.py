from typing import Callable
import torch
from gaussian_splatting.utils import l1_loss
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer
from feature_3dgs import SemanticGaussianModel, FeatureCamera


class SemanticTrainer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            semantic_decoder_lr=0.0001,
            semantic_loss_weight=1.0,
    ):
        super().__init__(base_trainer=base_trainer)
        model = self.model
        assert isinstance(model, SemanticGaussianModel), "SemanticTrainer's model must be a SemanticGaussianModel"
        self.optimizer.add_param_group({"lr": semantic_decoder_lr, "params": model.get_decoder.parameters()})
        self.semantic_loss_weight = semantic_loss_weight

    def loss(self, out: dict, camera: FeatureCamera) -> torch.Tensor:
        loss = super().loss(out, camera)
        render = out['feature_map']
        gt = camera.feature_map
        feature_loss = l1_loss(render, gt)
        return loss + feature_loss * self.semantic_loss_weight


def SemanticTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: SemanticGaussianModel,
        scene_extent: float,
        *args,
        semantic_decoder_lr=0.0001,
        semantic_loss_weight=1.0,
        **kwargs) -> SemanticTrainer:
    return SemanticTrainer(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        semantic_decoder_lr=semantic_decoder_lr,
        semantic_loss_weight=semantic_loss_weight,
    )
