from abc import abstractmethod
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.gaussian_model import AbstractFeatureDecoder, SemanticGaussianModel


class AbstractTrainableFeatureDecoder(AbstractFeatureDecoder):
    """Interface for trainable feature decoders that map from extractor feature space to a custom
    feature space.  Provides two more operations:

    - ``init_semantic``: initialise the decoder (e.g. via PCA on extractor features).
    - ``parameters``: return trainable parameters to be optimised by the trainer.
    """

    @staticmethod
    def init_semantic(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset):
        """Build the feature mapping from data (e.g. PCA). Called before training."""
        pass

    @abstractmethod
    def parameters(self):
        return []
