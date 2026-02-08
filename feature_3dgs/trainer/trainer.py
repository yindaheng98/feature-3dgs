import torch
from gaussian_splatting import Camera
from gaussian_splatting.utils import l1_loss
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer
from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs import FeatureGaussian

class FeatureTrainer(TrainerWrapper):
    def __init__(
        self,
        base_trainer: AbstractTrainer,
        extractor: AbstractFeatureExtractor
    ):
        super().__init__(base_trainer=base_trainer)
        self.optimizer.add_param_group([{"lr":0.0001, "params":extractor.parameters()}])
        self.extractor = extractor
        if isinstance(self.model, FeatureGaussian) and self.model.get_decoder is not None:
            self.optimizer.add_param_group([{"lr":0.0001, "params":self.model.get_decoder.parameters()}])


    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = self.base_trainer.loss(out, camera)
        feature_map_3dgs = out['feature_map']
        feature_map_extractor = self.extractor(camera.ground_truth_image)
        feature_map_loss = l1_loss(feature_map_3dgs, feature_map_extractor) 
        return loss + feature_map_loss

    def update_learning_rate(self):
        self.optim_step()

    def step(self, camera: Camera):
        self.update_learning_rate()
        camera = self.preprocess(camera)
        out = self.model(camera)
        loss = self.loss(out, camera)
        loss.backward()
        self.before_optim_hook(loss=loss, out=out, camera=camera)
        self.optim_step()
        self.after_optim_hook(loss=loss, out=out, camera=camera)
        return loss, out

    def extract_features(self, input):
        return self.extractor.extract(input)