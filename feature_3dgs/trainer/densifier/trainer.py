from dataclasses import dataclass

import torch

from gaussian_splatting.trainer.densifier import DensificationInstruct, DensificationTrainer, NoopDensifier
from feature_3dgs import SemanticGaussianModel


@dataclass(frozen=True)
class SemanticDensificationInstruct(DensificationInstruct):
    new_encoded_semantics: torch.Tensor = None
    replace_encoded_semantics_mask: torch.Tensor = None
    replace_encoded_semantics: torch.Tensor = None


class SemanticDensificationTrainer(DensificationTrainer):
    """DensificationTrainer that manages encoded_semantics in the optimizer."""

    optim_attr_names = {
        **DensificationTrainer.optim_attr_names,
        "semantic": "encoded_semantics",
    }


class SemanticNoopDensifier(NoopDensifier):

    @property
    def model(self) -> SemanticGaussianModel:
        return super().model

    def densify_and_prune(self, loss, out, camera, step: int) -> SemanticDensificationInstruct:
        return SemanticDensificationInstruct()
