from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from einops import rearrange

from feature_3dgs.extractor import AbstractFeatureExtractor

from ._impl.device import to_gpu
from .preprocess import PATCH_SIZE, prepare_views


class TTT3RExtractor(AbstractFeatureExtractor):
    """Multi-view feature extractor backed by the recurrent TTT3R rollout."""

    def __init__(
        self,
        model,
        resize: int = 512,
        reset_interval: int = 1_000_000,
        square_ok: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.resize = resize
        self.reset_interval = reset_interval
        self.square_ok = square_ok
        self.patch_size = model.patch_embed.patch_size[0]
        self.feature_dim = model.dec_embed_dim

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TTT3R requires multiple images. Use extract_all() instead.")

    @torch.no_grad()
    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        device = next(self.model.parameters()).device
        image_list = list(images)
        views, output_mask = prepare_views(
            image_list,
            resize=self.resize,
            reset_interval=self.reset_interval,
            square_ok=self.square_ok,
        )

        state_feat = None
        state_pos = None
        init_state_feat = None
        init_mem = None
        mem = None
        reset_mask = False

        for i, (_view, keep_output) in enumerate(zip(views, output_mask)):
            view = to_gpu(_view, device)
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(-1, batch_size)
            ray_mask = view["ray_mask"].reshape(-1, batch_size)
            imgs = view["img"].unsqueeze(0).view(-1, *view["img"].shape[1:])
            ray_maps = view["ray_map"].unsqueeze(0).view(-1, *view["ray_map"].shape[1:])
            shapes = _view["true_shape"].unsqueeze(0).view(-1, 2).to(device)

            img_masks_flat = img_mask.view(-1)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self.model._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            ray_maps = ray_maps.permute(0, 3, 1, 2)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self.model._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError("TTT3R requires either image or ray-map input.")

            if i == 0:
                state_feat, state_pos = self.model._init_state(feat_i, pos_i)
                mem = self.model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.model.pose_head_flag:
                global_img_feat_i = self.model._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.model.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.model.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                global_img_feat_i = None
                pose_feat_i = None
                pose_pos_i = None

            (
                new_state_feat,
                dec,
                _self_attn_state,
                cross_attn_state,
                _self_attn_img,
                _cross_attn_img,
            ) = self.model._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )

            out_pose_feat_i = dec[-1][:, 0:1]
            if global_img_feat_i is not None:
                new_mem = self.model.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )
            else:
                new_mem = mem

            if keep_output:
                tokens = dec[-1][:, 1:] if self.model.pose_head_flag else dec[-1]
                height = int(_view["true_shape"][0, 0].item())
                width = int(_view["true_shape"][0, 1].item())
                patch_height = height // self.patch_size
                patch_width = width // self.patch_size
                feature_map = tokens.reshape(
                    tokens.shape[0], patch_height, patch_width, tokens.shape[-1]
                )
                yield feature_map[0].permute(2, 0, 1).contiguous()

            update = view.get("update", None)
            if update is not None:
                update_mask = view["img_mask"] & update
            else:
                update_mask = view["img_mask"]
            update_mask = update_mask[:, None, None].float()

            if i == 0 or reset_mask:
                update_mask1 = update_mask
            elif self.model.config.model_update_type == "cut3r":
                update_mask1 = update_mask
            elif self.model.config.model_update_type == "ttt3r":
                cross_attn_state = rearrange(
                    torch.cat(cross_attn_state, dim=0),
                    "l h nstate nimg -> 1 nstate nimg (l h)",
                )
                state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None]
            else:
                raise ValueError(
                    f"Invalid model type: {self.model.config.model_update_type}"
                )

            state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)
            mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = bool(view["reset"].item())
            if reset_mask:
                reset_mask_tensor = view["reset"][:, None, None].float()
                state_feat = init_state_feat * reset_mask_tensor + state_feat * (
                    1 - reset_mask_tensor
                )
                mem = init_mem * reset_mask_tensor + mem * (1 - reset_mask_tensor)

    def to(self, device) -> "TTT3RExtractor":
        self.model.to(device)
        return self
