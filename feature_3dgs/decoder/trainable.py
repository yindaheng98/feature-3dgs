from __future__ import annotations

from .abc import AbstractSemanticDecoder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from feature_3dgs.extractor import FeatureCameraDataset


class AbstractTrainableDecoder(AbstractSemanticDecoder):
    """Interface for trainable feature decoders that map from extractor feature space to a custom
    feature space.  Provides two more operations:

    - ``init_semantic``: initialise the decoder (e.g. via PCA on extractor features).
    """

    def init_semantic(
            self,
            dataset: FeatureCameraDataset,
            decoder: AbstractSemanticDecoder | None = None):
        """Initialise decoder parameters from data (e.g. PCA). Called before training.

        Args:
            dataset: Dataset used to initialise decoder parameters.
            decoder: Optional preloaded decoder whose parameters are copied instead of
                fitting from *dataset*.
        """
        pass
