import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.decoder import NoopFeatureDecoder
from feature_3dgs.extractor import FeatureCameraDataset

from .extractor import padding


class DINOv3LinearAvgDecoder(NoopFeatureDecoder):
    """Decoder that aligns Gaussian features with DINOv3 extractor output.

    Two-stage pipeline:
      1. **transform_features** - a learnable per-point linear mapping
         ``(N, C_in) -> (N, C_out)`` that converts each feature vector
         from the Gaussian embedding space to the DINOv3 feature space.
      2. **postprocess** - spatial downsampling that matches DINOv3's
         patch-level resolution: pad to patch-size multiples, then average
         each non-overlapping patch into a single pixel.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        """
        Args:
            in_channels:  Per-point semantic embedding dimension rendered by
                          the Gaussian rasteriser.
            out_channels: Feature dimension D produced by DINOv3Extractor.
            patch_size:   Patch size used by the paired DINOv3Extractor.
        """
        super().__init__(embed_dim=in_channels)
        self.patch_size = patch_size
        # Step 1: trainable linear mapping  (C_in -> C_out per point)
        self.linear = nn.Linear(in_channels, out_channels)

    # ------------------------------------------------------------------
    # Two-stage interface
    # ------------------------------------------------------------------

    def transform_features(self, feature: torch.Tensor) -> torch.Tensor:
        """Pointwise linear mapping from Gaussian space to DINOv3 space.

        Args:
            feature: (N, C_in) batch of feature vectors.

        Returns:
            (N, C_out) transformed feature vectors.
        """
        return self.linear(feature)

    def postprocess(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pad to patch-size multiples then average-pool each patch.

        Args:
            feature_map: (C, H, W) tensor (already in DINOv3 feature space).

        Returns:
            (C, H_patches, W_patches) tensor matching DINOv3Extractor's
            spatial output exactly.
        """
        x = padding(feature_map, self.patch_size)          # (C, H', W')
        x = F.avg_pool2d(
            x.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(0)                                        # (C, H_p, W_p)
        return x

    # ------------------------------------------------------------------
    # AbstractDecoder interface
    # ------------------------------------------------------------------

    def init(self, dataset: FeatureCameraDataset):
        """Initialise linear layer weights via PCA on the extractor features.

        Collects all feature vectors from the dataset, computes PCA, and
        sets ``self.linear`` so that it initially performs PCA reconstruction:
          - weight = top-k principal components  (out_channels, in_channels)
          - bias   = feature mean                (out_channels,)
        """
        # Collect all extractor feature vectors  -> (total_pixels, D)
        all_features = []
        for idx in tqdm.tqdm(range(len(dataset)), desc="PCA init: collecting features"):
            feature_map = dataset[idx].custom_data['feature_map']  # (D, H_p, W_p)
            features = feature_map.reshape(feature_map.shape[0], -1).T  # (N_i, D)
            all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0).float()  # (N_total, D)

        # PCA
        mean = all_features.mean(dim=0)                        # (D,)
        centered = all_features - mean
        _, _, V = torch.pca_lowrank(centered, q=self.linear.in_features)
        # V: (D, k) = (out_channels, in_channels)

        # nn.Linear computes: output = input @ weight.T + bias
        # PCA reconstruction: Z @ V.T + mean ≈ original
        # So weight = V, bias = mean
        with torch.no_grad():
            device = self.linear.weight.device
            self.linear.weight.copy_(V.to(device))
            self.linear.bias.copy_(mean.to(device))

    def to(self, device) -> 'DINOv3LinearAvgDecoder':
        self.linear = self.linear.to(device)
        return self

    def load(self, path: str) -> None:
        state_dict = torch.load(path, weights_only=True)
        self.linear.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save(self.linear.state_dict(), path)

    def parameters(self):
        return self.linear.parameters()
