from typing import Callable
import torch
from gaussian_splatting.utils import get_expon_lr_func
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
            semantic_smooth_weight=0.1,
            semantic_mask_mode="none",
    ):
        super().__init__(base_trainer=base_trainer)
        model = self.model
        assert isinstance(model, SemanticGaussianModel), "SemanticTrainer's model must be a SemanticGaussianModel"
        self.optimizer.add_param_group({"lr": semantic_lr, "params": model._encoded_semantics, "name": "semantic"})
        self.optimizer.add_param_group({"lr": semantic_decoder_lr_init, "params": model.get_decoder.parameters(), "name": "semantic_decoder"})
        self.schedulers['semantic_decoder'] = get_expon_lr_func(
            lr_init=semantic_decoder_lr_init,
            lr_final=semantic_decoder_lr_final,
            lr_delay_mult=semantic_decoder_lr_delay_mult,
            max_steps=semantic_decoder_lr_max_steps,
        )
        self.semantic_loss_weight = semantic_loss_weight
        self.semantic_smooth_weight = semantic_smooth_weight
        self.mask_mode = semantic_mask_mode

    @property
    def model(self) -> SemanticGaussianModel:
        return self.base_trainer.model

    def semantic_loss(self, render, gt):
        return torch.abs((render - gt))  # L1 loss

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)

        render = out['feature_map']
        gt = camera.custom_data['feature_map']
        semantic_loss = self.semantic_loss(render, gt)

        smooth_loss = None
        encoded = out['feature_map_encoded']
        if gt.shape[1:] != encoded.shape[1:]:
            gt_encoded = self.model.get_decoder.encode_feature_map(camera.custom_data['feature_map'], camera)
            smooth_loss = self.semantic_loss(encoded, gt_encoded)

        match self.mask_mode:
            case "none":
                pass
            case "ignore":
                mask = camera.ground_truth_image_mask
                assert mask is not None, "Mask is required for 'ignore' mask policy"
                decoder = self.model.get_decoder
                semantic_loss = semantic_loss * decoder.resize_mask(mask, render).unsqueeze(0)
                if smooth_loss is not None:
                    smooth_loss = smooth_loss * decoder.resize_mask(mask, encoded).unsqueeze(0)
            case _:
                raise ValueError(f"Unknown mask policy: {self.mask_mode}")

        loss = loss + semantic_loss.mean() * self.semantic_loss_weight
        if smooth_loss is not None:
            loss = loss + smooth_loss.mean() * self.semantic_smooth_weight
        return loss


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
        semantic_smooth_weight=0.1,
        semantic_mask_mode="none",
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
        semantic_smooth_weight=semantic_smooth_weight,
        semantic_mask_mode=semantic_mask_mode,
    )


def BaseSemanticTrainer(model: SemanticGaussianModel, dataset: FeatureCameraDataset, **configs) -> SemanticTrainer:
    return SemanticTrainerWrapper(BaseTrainer, model, dataset, **configs)
