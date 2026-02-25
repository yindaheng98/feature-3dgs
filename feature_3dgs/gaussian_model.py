from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gaussian_splatting import GaussianModel, Camera, CameraTrainableGaussianModel
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class AbstractFeatureDecoder(ABC):
    """Feature decoder that bridges encoded Gaussian semantics and extractor
    feature space.  Provides three core operations:

    Terminology
    -----------
    - **C_enc** — channel dimension of the *encoded* (compact) representation
      stored per Gaussian and produced by rasterisation.
    - **C_feat** — channel dimension of the *decoded* feature, which matches
      the ground-truth extractor output (e.g. DINOv2 feature dim).

    Operations
    ----------
    - ``decode_features``: per-point decoding ``(N, C_enc) -> (N, C_feat)``,
      applicable directly to per-Gaussian encoded semantics.
    - ``decode_feature_map``: convert a full rendered feature map
      ``(C_enc, H, W)`` into extractor output format ``(C_feat, H', W')``,
      matching both channel dimension and spatial resolution.  Subclasses
      may override it with reparameterized, memory-efficient implementations
      — e.g. a single Conv2d that fuses per-point mapping and spatial
      downsampling without materialising a large intermediate tensor.
    - ``decode_feature_pixels``: per-pixel projection that applies
      ``decode_features`` and optionally a custom linear layer
      (``weight`` / ``bias``), always preserving the original spatial
      resolution.  Useful for producing full-resolution feature maps with
      arbitrary output dimensions — e.g. a 3-channel PCA visualisation.
    - ``encode_features``: per-point encoding ``(N, C_feat) -> (N, C_enc)``,
      the inverse of ``decode_features``.
    - ``encode_feature_map``: convert an extractor feature map
      ``(C_feat, H', W')`` back to encoded format ``(C_enc, H, W)``,
      the inverse of ``decode_feature_map``.
    """

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Per-point decoding: (N, C_enc) -> (N, C_feat)."""
        return features

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Per-point encoding: (N, C_feat) -> (N, C_enc).

        Inverse of ``decode_features``.
        """
        return features

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Convert rendered feature map to extractor output format.

        Default: apply decode_features per pixel (no spatial change).
        Subclasses may override for fused spatial downsampling.

        Args:
            feature_map: (C_enc, H, W) — encoded (rasterised) feature map.

        Returns:
            (C_feat, H', W') matching extractor ground-truth output.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_enc)
        x = self.decode_features(x)                       # (H*W, C_feat)
        return x.reshape(H, W, -1).permute(2, 0, 1)       # (C_feat, H, W), override to change spatial resolution as well

    def encode_feature_map(self, feature_map: torch.Tensor, camera: Camera) -> torch.Tensor:
        """Convert extractor feature map back to encoded format.

        Default: apply encode_features per pixel (no spatial change).
        Subclasses may override to add spatial upsampling.

        Args:
            feature_map: (C_feat, H', W') — extractor feature map.
            camera: Camera with target spatial dimensions.

        Returns:
            (C_enc, H, W) in the encoded space.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_feat)
        x = self.encode_features(x)                       # (H*W, C_enc)
        return x.reshape(H, W, -1).permute(2, 0, 1)       # (C_enc, H, W)

    def decode_feature_pixels(self, feature_map: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Per-pixel feature projection (spatial resolution preserved).

        Applies ``decode_features`` to **every pixel**, then optionally a
        custom linear layer ``F.linear(x, weight, bias)``.

        Args:
            feature_map: (C_enc, H, W) — encoded (rasterised) feature map.
            weight: (C_proj, C_feat) or None.  Skipped when None.
            bias:   (C_proj,) or None.

        Returns:
            (C_feat, H, W) when weight is None,
            (C_proj, H, W) when weight is given.
        """
        C, H, W = feature_map.shape
        x = feature_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C_enc)
        x = self.decode_features(x)                       # (H*W, C_feat)
        if weight is not None:
            x = F.linear(x, weight, bias)                  # (H*W, C_proj)
        return x.reshape(H, W, -1).permute(2, 0, 1)

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.decode_feature_map(feature_map)

    @abstractmethod
    def to(self, device) -> 'AbstractFeatureDecoder':
        return self

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        return 0


class SemanticGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, decoder: AbstractFeatureDecoder):
        super(SemanticGaussianModel, self).__init__(sh_degree)
        self._encoded_semantics = torch.empty(0)
        self._decoder = decoder

    def to(self, device):
        super().to(device)
        self._encoded_semantics = self._encoded_semantics.to(device)
        self._decoder = self._decoder.to(device)
        return self

    @property
    def get_encoded_semantics(self):
        """Raw per-Gaussian encoded semantics (before decoder transform)."""
        return self._encoded_semantics

    @property
    def get_semantics(self):
        """Per-Gaussian semantic features aligned with the extractor's space.

        Applies the decoder's pointwise decode_features (no spatial
        post-processing) so the result is directly comparable to the
        extractor output.

        Returns:
            (N, C_feat) tensor in the extractor's feature space.
        """
        return self._decoder.decode_features(self._encoded_semantics)

    @property
    def get_decoder(self):
        return self._decoder

    def forward(self, viewpoint_camera: Camera):
        return self.render(
            viewpoint_camera=viewpoint_camera,
            means3D=self.get_xyz,
            opacity=self.get_opacity,
            scales=self.get_scaling,
            rotations=self.get_rotation,
            shs=self.get_features,
            semantic_features=self.get_encoded_semantics,
        )

    def render(
        self,
        viewpoint_camera: Camera,
        means3D: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        shs: torch.Tensor,
        semantic_features: torch.Tensor,
        colors_precomp=None,
        cov3D_precomp=None,
    ) -> dict:
        out = self.render_encoded(
            viewpoint_camera=viewpoint_camera,
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            shs=shs,
            semantic_features=semantic_features,
            colors_precomp=colors_precomp,
            cov3D_precomp=cov3D_precomp,
        )
        out['feature_map_encoded'] = out['feature_map']
        out['feature_map'] = self.get_decoder(out['feature_map_encoded'])
        return out

    def forward_projection(self, viewpoint_camera: Camera, weight: torch.Tensor, bias: torch.Tensor = None):
        """Render and apply a custom linear projection to the feature map.

        The decoder's ``decode_features`` is applied per pixel, followed
        by the supplied linear mapping.  Spatial resolution is preserved.
        """
        return self.render_projection(
            viewpoint_camera=viewpoint_camera,
            means3D=self.get_xyz,
            opacity=self.get_opacity,
            scales=self.get_scaling,
            rotations=self.get_rotation,
            shs=self.get_features,
            semantic_features=self.get_encoded_semantics,
            weight=weight,
            bias=bias,
        )

    def render_projection(
        self,
        viewpoint_camera: Camera,
        means3D: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        shs: torch.Tensor,
        semantic_features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        colors_precomp=None,
        cov3D_precomp=None,
    ) -> dict:
        out = self.render_encoded(
            viewpoint_camera=viewpoint_camera,
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            shs=shs,
            semantic_features=semantic_features,
            colors_precomp=colors_precomp,
            cov3D_precomp=cov3D_precomp,
        )
        out['feature_map_encoded'] = out['feature_map']
        out['feature_map'] = self._decoder.decode_feature_pixels(out['feature_map_encoded'], weight=weight, bias=bias)
        return out

    def render_encoded(
        self,
        viewpoint_camera: Camera,
        means3D: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        shs: torch.Tensor,
        semantic_features: torch.Tensor,
        colors_precomp=None,
        cov3D_precomp=None,
    ) -> dict:
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=viewpoint_camera.bg_color,
            scale_modifier=self.scale_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            antialiasing=self.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, feature_map, radii, invdepth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            semantic_features=semantic_features,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rendered_image = rendered_image.clamp(0, 1)
        out = {
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "invdepth": invdepth_image,
            # Used by the densifier to get the gradient of the viewspace points
            "get_viewspace_grad": lambda out: out["viewspace_points"].grad,
            "viewspace_points": screenspace_points,
            # Feature maps: encoded (raw rasterised) and decoded (extractor-aligned)
            'feature_map': feature_map,
        }
        return out

    def reset_encoded_semantics(self):
        encoded_semantics = torch.zeros((self._xyz.shape[0], self._decoder.embed_dim), dtype=torch.float, device=self._xyz.device)
        self._encoded_semantics = nn.Parameter(encoded_semantics.requires_grad_(True))
        return self

    def create_from_pcd(self, points: torch.Tensor, colors: torch.Tensor):
        super().create_from_pcd(points, colors)
        return self.reset_encoded_semantics()

    def save_ply(self, path: str):
        super().save_ply(path)
        encoded_semantics = self._encoded_semantics.detach()
        torch.save(encoded_semantics, path + '.semantic.pt')
        self._decoder.save(path + '.decoder.pt')

    def load_ply(self, path: str, load_semantic: bool = True):
        super().load_ply(path)
        if load_semantic:
            encoded_semantics = torch.load(path + '.semantic.pt').to(self._xyz.device)
            self._encoded_semantics = nn.Parameter(encoded_semantics.requires_grad_(True))
            self._decoder.load(path + '.decoder.pt')
        else:
            self.reset_encoded_semantics()

    def update_points_add(
        self,
        xyz: nn.Parameter,
        features_dc: nn.Parameter,
        features_rest: nn.Parameter,
        scaling: nn.Parameter,
        rotation: nn.Parameter,
        opacity: nn.Parameter,
        encoded_semantics: nn.Parameter,
    ):
        super().update_points_add(
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        def is_same_prefix(attr: nn.Parameter, ref: nn.Parameter):
            return (attr[:ref.shape[0]] == ref).all()
        assert is_same_prefix(encoded_semantics, self._encoded_semantics)
        self._encoded_semantics = encoded_semantics

    def update_points_replace(
            self,
            xyz_mask: torch.Tensor, xyz: nn.Parameter,
            features_dc_mask: torch.Tensor, features_dc: nn.Parameter,
            features_rest_mask: torch.Tensor, features_rest: nn.Parameter,
            scaling_mask: torch.Tensor, scaling: nn.Parameter,
            rotation_mask: torch.Tensor, rotation: nn.Parameter,
            opacity_mask: torch.Tensor, opacity: nn.Parameter,
            encoded_semantics_mask: torch.Tensor, encoded_semantics: nn.Parameter
    ):
        super().update_points_replace(
            xyz_mask=xyz_mask, xyz=xyz,
            features_dc_mask=features_dc_mask, features_dc=features_dc,
            features_rest_mask=features_rest_mask, features_rest=features_rest,
            scaling_mask=scaling_mask, scaling=scaling,
            rotation_mask=rotation_mask, rotation=rotation,
            opacity_mask=opacity_mask, opacity=opacity,
        )

        def is_same_rest(attr: nn.Parameter, ref: nn.Parameter, mask: torch.Tensor):
            return (attr[~mask, ...] == ref[~mask, ...]).all()
        assert encoded_semantics_mask is None or is_same_rest(encoded_semantics, self._encoded_semantics, encoded_semantics_mask)
        self._encoded_semantics = encoded_semantics

    def update_points_remove(
            self,
            removed_mask: torch.Tensor,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            encoded_semantics
    ):
        super().update_points_remove(
            removed_mask=removed_mask,
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        def is_same_rest(attr: nn.Parameter, ref: nn.Parameter):
            return (attr == ref[~removed_mask, ...]).all()
        assert is_same_rest(encoded_semantics, self._encoded_semantics)
        self._encoded_semantics = encoded_semantics


class CameraTrainableSemanticGaussianModel(SemanticGaussianModel):
    def forward(self, camera: Camera):
        return CameraTrainableGaussianModel.forward(self, camera)
