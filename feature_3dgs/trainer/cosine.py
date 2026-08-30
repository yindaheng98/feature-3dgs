from typing import Callable
import torch.nn.functional as F
from gaussian_splatting.trainer import AbstractTrainer, BaseTrainer
from feature_3dgs import SemanticGaussianModel, FeatureCameraDataset
from .trainer import SemanticTrainer


class CosineSemanticTrainer(SemanticTrainer):
    def __init__(self, *args, semantic_loss_cosine_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_loss_cosine_weight = semantic_loss_cosine_weight

    def semantic_loss(self, render, gt):
        return super().semantic_loss(render, gt) + self.semantic_loss_cosine_weight * (1 - F.cosine_similarity(render, gt, dim=0))


def CosineSemanticTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: SemanticGaussianModel,
        dataset: FeatureCameraDataset,
        *args,
        semantic_lr=0.1,
        semantic_decoder_lr_init=0.001,
        semantic_decoder_lr_final=0.00001,
        semantic_decoder_lr_delay_mult=0.01,
        semantic_decoder_lr_max_steps=30_000,
        semantic_loss_weight=1.0,
        semantic_smooth_weight=0.1,
        semantic_mask_mode="none",
        semantic_loss_cosine_weight=1.0,
        **configs) -> CosineSemanticTrainer:
    return CosineSemanticTrainer(
        base_trainer=base_trainer_constructor(model, dataset, *args, **configs),
        dataset=dataset,
        semantic_lr=semantic_lr,
        semantic_decoder_lr_init=semantic_decoder_lr_init,
        semantic_decoder_lr_final=semantic_decoder_lr_final,
        semantic_decoder_lr_delay_mult=semantic_decoder_lr_delay_mult,
        semantic_decoder_lr_max_steps=semantic_decoder_lr_max_steps,
        semantic_loss_weight=semantic_loss_weight,
        semantic_smooth_weight=semantic_smooth_weight,
        semantic_mask_mode=semantic_mask_mode,
        semantic_loss_cosine_weight=semantic_loss_cosine_weight,
    )


def BaseCosineSemanticTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs) -> CosineSemanticTrainer:
    return CosineSemanticTrainerWrapper(BaseTrainer, model, dataset, **configs)
