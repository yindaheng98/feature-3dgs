from __future__ import annotations

from abc import abstractmethod
from .abc import AbstractSemanticDecoder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature_3dgs.extractor import FeatureCameraDataset
    from feature_3dgs.gaussian_model import SemanticGaussianModel


class AbstractTrainableDecoder(AbstractSemanticDecoder):
    """Interface for trainable feature decoders that map from extractor feature space to a custom
    feature space.  Provides two more operations:

    - ``init_semantic``: initialise the decoder (e.g. via PCA on extractor features).
    """

    @staticmethod
    def init_semantic(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset):
        """Build the feature mapping from data (e.g. PCA). Called before training."""
        pass
