from dataclasses import dataclass

import torch

from gaussian_splatting.trainer.densifier import DensificationInstruct, DensificationTrainer


@dataclass(frozen=True)
class SemanticDensificationInstruct(DensificationInstruct):
    new_semantic_features: torch.Tensor = None
    replace_semantic_features_mask: torch.Tensor = None
    replace_semantic_features: torch.Tensor = None


class SemanticDensificationTrainer(DensificationTrainer):
    """DensificationTrainer that manages semantic_features in the optimizer."""

    optim_attr_names = {
        **DensificationTrainer.optim_attr_names,
        "semantic": "semantic_features",
    }
