from functools import partial
from typing import Callable

from gaussian_splatting.trainer.densifier import AbstractDensifier, OpacityPrunerDensifierWrapper
from feature_3dgs import SemanticGaussianModel, FeatureCameraDataset

from .densifier import SemanticSplitCloneDensifierWrapper
from .trainer import SemanticDensificationTrainer, SemanticNoopDensifier


def SemanticDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: SemanticGaussianModel, dataset: FeatureCameraDataset, *args,
        **configs) -> AbstractDensifier:
    return OpacityPrunerDensifierWrapper(
        partial(SemanticSplitCloneDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs,
    )


def SemanticDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: SemanticGaussianModel, dataset: FeatureCameraDataset, *args,
        **configs):
    return SemanticDensificationTrainer.from_densifier_constructor(
        partial(SemanticDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs,
    )


def BaseSemanticDensificationTrainer(
        model: SemanticGaussianModel, dataset: FeatureCameraDataset,
        **configs):
    return SemanticDensificationTrainerWrapper(
        lambda model, dataset, **configs: SemanticNoopDensifier(model),
        model, dataset,
        **configs,
    )
