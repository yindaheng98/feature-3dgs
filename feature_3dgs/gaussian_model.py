import math
import os
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel, Camera, CameraTrainableGaussianModel
from feature_3dgs.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .decoder import AbstractFeatureDecoder


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

        Applies the decoder's pointwise transform_features (no spatial
        post-processing) so the result is directly comparable to the
        extractor output.

        Returns:
            (N, C_out) tensor in the extractor's feature space.
        """
        return self._decoder.transform_features(self._encoded_semantics)

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

    def render(self, *args, **kwargs) -> dict:
        out = self.render_encoded(*args, **kwargs)
        out['feature_map'] = self.get_decoder(out['feature_map'])
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

    def forward_linear_projection(self, viewpoint_camera: Camera, weight: torch.Tensor, bias: torch.Tensor = None):
        """Render and apply a custom linear projection to the feature map.

        The decoder's ``transform_features`` is applied per pixel, followed
        by the supplied linear mapping.  Spatial resolution is preserved.
        """
        out = self.render_encoded(
            viewpoint_camera=viewpoint_camera,
            means3D=self.get_xyz,
            opacity=self.get_opacity,
            scales=self.get_scaling,
            rotations=self.get_rotation,
            shs=self.get_features,
            semantic_features=self.get_encoded_semantics,
        )
        out['feature_map'] = self._decoder.project_feature_map(out['feature_map'], weight=weight, bias=bias)
        return out

    def init_encoded_semantics(self):
        encoded_semantics = torch.zeros((self._xyz.shape[0], self._decoder.embed_dim), dtype=torch.float, device=self._xyz.device)
        self._encoded_semantics = nn.Parameter(encoded_semantics.requires_grad_(True))
        return self

    def create_from_pcd(self, points: torch.Tensor, colors: torch.Tensor):
        super().create_from_pcd(points, colors)
        return self.init_encoded_semantics()

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
            self.init_encoded_semantics()

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
