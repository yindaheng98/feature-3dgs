from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import PIL
import torch
import torchvision.transforms.functional as TF


PATCH_SIZE = 16


def compute_resized_resolution(height: int, width: int, resize: int) -> tuple[int, int]:
    if resize == 224:
        long_edge = round(resize * max(width / height, height / width))
    else:
        long_edge = resize
    scale = long_edge / max(height, width)
    resized_height = max(int(round(height * scale)), 1)
    resized_width = max(int(round(width * scale)), 1)
    return resized_height, resized_width


def compute_crop_window(
    height: int,
    width: int,
    resize: int,
    square_ok: bool = False,
    patch_size: int = PATCH_SIZE,
) -> tuple[int, int, int, int, int, int]:
    resized_height, resized_width = compute_resized_resolution(height, width, resize)
    center_x = resized_width // 2
    center_y = resized_height // 2

    if resize == 224:
        half = min(center_x, center_y)
        crop_width = 2 * half
        crop_height = 2 * half
    else:
        crop_width = ((2 * center_x) // patch_size) * patch_size
        crop_height = ((2 * center_y) // patch_size) * patch_size
        if not square_ok and resized_width == resized_height:
            crop_height = int(round(3 * crop_width / 4))

    left = max((resized_width - crop_width) // 2, 0)
    top = max((resized_height - crop_height) // 2, 0)
    return resized_height, resized_width, top, left, crop_height, crop_width


def compute_processed_resolution(
    height: int,
    width: int,
    resize: int,
    square_ok: bool = False,
    patch_size: int = PATCH_SIZE,
) -> tuple[int, int]:
    _, _, _, _, crop_height, crop_width = compute_crop_window(
        height, width, resize, square_ok=square_ok, patch_size=patch_size
    )
    return crop_height, crop_width


def preprocess_image(
    image: torch.Tensor,
    resize: int,
    square_ok: bool = False,
    patch_size: int = PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C, H, W), got {tuple(image.shape)}")
    if image.shape[0] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got {image.shape[0]} channels")

    height, width = int(image.shape[-2]), int(image.shape[-1])
    resized_height, resized_width, top, left, crop_height, crop_width = compute_crop_window(
        height, width, resize, square_ok=square_ok, patch_size=patch_size
    )

    image_cpu = image.detach().float().cpu().clamp(0.0, 1.0)
    pil_image = TF.to_pil_image(image_cpu)
    interp = (
        PIL.Image.LANCZOS
        if max(pil_image.size) > max(resized_height, resized_width)
        else PIL.Image.BICUBIC
    )
    pil_image = pil_image.resize((resized_width, resized_height), interp)
    pil_image = pil_image.crop((left, top, left + crop_width, top + crop_height))

    processed = TF.to_tensor(pil_image)
    processed = TF.normalize(processed, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    true_shape = torch.tensor([[crop_height, crop_width]], dtype=torch.int64)
    return processed, true_shape


def prepare_views(
    images: Iterable[torch.Tensor],
    resize: int,
    reset_interval: int,
    square_ok: bool = False,
    revisit: int = 1,
    update: bool = True,
) -> tuple[list[dict[str, torch.Tensor | int | str]], list[bool]]:
    views: list[dict[str, torch.Tensor | int | str]] = []
    output_mask: list[bool] = []

    for index, image in enumerate(images):
        processed, true_shape = preprocess_image(
            image, resize, square_ok=square_ok, patch_size=PATCH_SIZE
        )
        view = {
            "img": processed.unsqueeze(0),
            "ray_map": torch.full(
                (1, 6, processed.shape[-2], processed.shape[-1]),
                torch.nan,
                dtype=processed.dtype,
            ),
            "true_shape": true_shape,
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.eye(4, dtype=processed.dtype).unsqueeze(0),
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([(index + 1) % reset_interval == 0]),
        }
        views.append(view)
        output_mask.append(True)
        if (index + 1) % reset_interval == 0:
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor([False])
            views.append(overlap_view)
            output_mask.append(False)

    if revisit > 1:
        expanded_views: list[dict[str, torch.Tensor | int | str]] = []
        expanded_mask: list[bool] = []
        for revisit_index in range(revisit):
            for view, keep_output in zip(views, output_mask):
                new_view = deepcopy(view)
                expanded_index = len(expanded_views)
                new_view["idx"] = expanded_index
                new_view["instance"] = str(expanded_index)
                if revisit_index > 0 and not update:
                    new_view["update"] = torch.tensor([False])
                expanded_views.append(new_view)
                expanded_mask.append(keep_output and revisit_index == revisit - 1)
        return expanded_views, expanded_mask

    return views, output_mask
