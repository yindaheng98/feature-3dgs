import torch
from gaussian_splatting.utils import l1_loss
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer
from feature_3dgs import FeatureGaussianModel
from feature_3dgs import FeatureCamera


class FeatureTrainer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            feature_l1_weight=1.0,
    ):
        super().__init__(base_trainer=base_trainer)
        model = self.model
        assert isinstance(model, FeatureGaussianModel), "FeatureTrainer's model must be a FeatureGaussianModel"
        self.optimizer.add_param_group([{"lr": 0.0001, "params": model.get_decoder.parameters()}])
        self.feature_l1_weight = feature_l1_weight

    def loss(self, out: dict, camera: FeatureCamera) -> torch.Tensor:
        loss = super().loss(out, camera)
        render = out['feature_map']
        gt = camera.feature_map
        feature_loss = l1_loss(render, gt)
        return loss + feature_loss * self.feature_l1_weight
