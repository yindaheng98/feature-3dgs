from functools import partial
from typing import Callable

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer.densifier import SplitCloneDensifier, AbstractDensifier
from feature_3dgs import SemanticGaussianModel

from .trainer import SemanticDensificationInstruct, SemanticDensificationTrainer


class SemanticSplitCloneDensifier(SplitCloneDensifier):
    """SplitCloneDensifier that also handles _semantic_features."""

    @property
    def model(self) -> SemanticGaussianModel:
        return super().model

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        base = super().densify_and_split(grads, grad_threshold, scene_extent, N)
        # base.replace_xyz_mask is exactly the selected_pts_mask
        selected_pts_mask = base.replace_xyz_mask
        new_semantic_features = self.model._semantic_features[selected_pts_mask]
        return SemanticDensificationInstruct(
            **base._asdict(),
            new_semantic_features=new_semantic_features,
        )

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values <= self.densify_percent_dense * scene_extent,
        )

        return SemanticDensificationInstruct(
            new_xyz=self.model._xyz[selected_pts_mask],
            new_features_dc=self.model._features_dc[selected_pts_mask],
            new_features_rest=self.model._features_rest[selected_pts_mask],
            new_opacity=self.model._opacity[selected_pts_mask],
            new_scaling=self.model._scaling[selected_pts_mask],
            new_rotation=self.model._rotation[selected_pts_mask],
            new_semantic_features=self.model._semantic_features[selected_pts_mask],
        )


# ======================================================================
# Convenience wrapper functions (mirror the base module's pattern)
# ======================================================================

def SemanticSplitCloneDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        densify_limit_n=None,
        **configs):
    return SemanticSplitCloneDensifier(
        base_densifier_constructor(model, dataset, *args, **configs),
        dataset,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big,
        densify_limit_n=densify_limit_n
    )


def SemanticSplitCloneDensifierTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        *args, **configs):
    return SemanticDensificationTrainer.from_densifier_constructor(
        partial(SemanticSplitCloneDensifierWrapper, base_densifier_constructor),
        model, dataset,
        *args, **configs,
    )
