from typing import Callable
import torch
from gaussian_splatting.utils import l1_loss, get_expon_lr_func
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer, BaseTrainer
from gaussian_splatting import Camera
from feature_3dgs import SemanticGaussianModel, FeatureCameraDataset


class SemanticTrainer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            dataset: FeatureCameraDataset,
            semantic_lr=0.1,
            semantic_decoder_lr_init=0.001,
            semantic_decoder_lr_final=0.00001,
            semantic_decoder_lr_delay_mult=0.01,
            semantic_decoder_lr_max_steps=30_000,
            semantic_loss_weight=1.0,
            semantic_mask_mode="none",
    ):
        super().__init__(base_trainer=base_trainer)
        model = self.model
        assert isinstance(model, SemanticGaussianModel), "SemanticTrainer's model must be a SemanticGaussianModel"
        self.optimizer.add_param_group({"lr": semantic_lr, "params": model._encoded_semantics, "name": "semantic"})
        model.get_decoder.init(dataset)  # Init decoder before adding its parameters to the optimizer
        self.optimizer.add_param_group({"lr": semantic_decoder_lr_init, "params": model.get_decoder.parameters(), "name": "semantic_decoder"})
        self.schedulers['semantic_decoder'] = get_expon_lr_func(
            lr_init=semantic_decoder_lr_init,
            lr_final=semantic_decoder_lr_final,
            lr_delay_mult=semantic_decoder_lr_delay_mult,
            max_steps=semantic_decoder_lr_max_steps,
        )
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
        dataset: FeatureCameraDataset,
        *args,
        semantic_lr=0.1,
        semantic_decoder_lr_init=0.001,
        semantic_decoder_lr_final=0.00001,
        semantic_decoder_lr_delay_mult=0.01,
        semantic_decoder_lr_max_steps=30_000,
        semantic_loss_weight=1.0,
        **configs) -> SemanticTrainer:
    return SemanticTrainer(
        base_trainer=base_trainer_constructor(model, dataset, *args, **configs),
        dataset=dataset,
        semantic_lr=semantic_lr,
        semantic_decoder_lr_init=semantic_decoder_lr_init,
        semantic_decoder_lr_final=semantic_decoder_lr_final,
        semantic_decoder_lr_delay_mult=semantic_decoder_lr_delay_mult,
        semantic_decoder_lr_max_steps=semantic_decoder_lr_max_steps,
        semantic_loss_weight=semantic_loss_weight,
    )


def BaseSemanticTrainer(
        model: SemanticGaussianModel,
        dataset: FeatureCameraDataset,
        semantic_lr=0.1,
        semantic_decoder_lr_init=0.001,
        semantic_decoder_lr_final=0.00001,
        semantic_decoder_lr_delay_mult=0.01,
        semantic_decoder_lr_max_steps=30_000,
        semantic_loss_weight=1.0,
        **configs) -> SemanticTrainer:
    return SemanticTrainerWrapper(
        lambda model, dataset, **configs: BaseTrainer(model, dataset, **configs),
        model=model,
        dataset=dataset,
        semantic_lr=semantic_lr,
        semantic_decoder_lr_init=semantic_decoder_lr_init,
        semantic_decoder_lr_final=semantic_decoder_lr_final,
        semantic_decoder_lr_delay_mult=semantic_decoder_lr_delay_mult,
        semantic_decoder_lr_max_steps=semantic_decoder_lr_max_steps,
        semantic_loss_weight=semantic_loss_weight,
        **configs,
    )
