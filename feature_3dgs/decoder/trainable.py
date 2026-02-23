import torch
from abc import abstractmethod
from gaussian_splatting import Camera
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.gaussian_model import AbstractFeatureDecoder, SemanticGaussianModel


class AbstractTrainableFeatureDecoder(AbstractFeatureDecoder):
    """Interface for trainable feature decoders that map from extractor feature space to a custom
    feature space.  Provides two more operations:

    - ``init_semantic``: initialise the decoder (e.g. via PCA on extractor features).
    - ``parameters``: return trainable parameters to be optimised by the trainer.
    - ``transform_feature_map_inverse``: reverse of ``transform_feature_map``,
      mapping a ground-truth extractor feature map back into the encoded space
      at full rendered resolution.  Used to compute a smoothness loss that
      bypasses avg-pool information loss.
    """

    @staticmethod
    def init_semantic(gaussians: SemanticGaussianModel, dataset: FeatureCameraDataset):
        """Build the feature mapping from data (e.g. PCA). Called before training."""
        pass

    @abstractmethod
    def parameters(self):
        return []

    def transform_feature_map_inverse(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Inverse of ``transform_feature_map``: map extractor GT back to encoded space.

        Reverses the channel mapping and spatial downsampling so the result
        can be compared pixel-by-pixel with the raw encoded feature map
        ``(C_enc, H, W)`` at the full rendered resolution.

        Args:
            feature_map: ``(C_feat, H', W')`` — ground-truth feature map from
                the extractor (low-resolution, extractor channel space).
            camera: Camera object containing the target spatial dimensions.

        Returns:
            ``(C_enc, H, W)`` — feature map projected into the
            encoded space at full resolution.
        """
        return feature_map
