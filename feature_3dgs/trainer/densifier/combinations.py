from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier, OpacityPrunerDensifierWrapper

from .densifier import SemanticSplitCloneDensifierWrapper
from .trainer import SemanticDensificationTrainer


def SemanticDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs) -> AbstractDensifier:
    return OpacityPrunerDensifierWrapper(
        partial(SemanticSplitCloneDensifierWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs,
    )


def SemanticDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return SemanticDensificationTrainer.from_densifier_constructor(
        partial(SemanticDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs,
    )


def BaseSemanticDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return SemanticDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent,
        *args, **kwargs,
    )
