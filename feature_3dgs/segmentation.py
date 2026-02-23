import os

import torch
import torch.nn.functional as F
from feature_3dgs import get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.render import prepare_rendering, rendering


def get_feature(
        dataset: FeatureCameraDataset, image_index: int, x: int, y: int,
) -> torch.Tensor:
    """Extract the feature vector at pixel (x, y) from the GT feature map.

    The GT feature map is at patch resolution (D, H_p, W_p); pixel coordinates
    are normalised to [-1, 1] and sampled via nearest interpolation so that
    ``grid_sample`` handles the image-to-feature coordinate mapping.
    """
    camera = dataset[image_index]
    feature_map = camera.custom_data['feature_map']
    grid = torch.tensor([[[[2.0 * x / (camera.image_width - 1) - 1.0, 2.0 * y / (camera.image_height - 1) - 1.0]]]], device=feature_map.device)
    feature = F.grid_sample(feature_map.unsqueeze(0), grid, mode='nearest', align_corners=True).squeeze(0)
    return feature.squeeze(-1).squeeze(-1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--name", choices=get_available_extractor_decoders(), required=True, type=str)
    parser.add_argument("--embed_dim", required=True, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("-e", "--option_extractor", default=[], action='append', type=str)
    parser.add_argument("-n", "--image_index", required=True, type=int)
    parser.add_argument("-x", required=True, type=int)
    parser.add_argument("-y", required=True, type=int)
    parser.add_argument("-t", "--threshold", required=True, type=float)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    extractor_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_extractor}
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            name=args.name, sh_degree=args.sh_degree,
            source=args.source, embed_dim=args.embed_dim,
            device=args.device, dataset_cache_device=args.dataset_cache_device,
            trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=not args.no_image_mask,
            extractor_configs=extractor_configs)
        feature = get_feature(dataset, args.image_index, args.x, args.y)
        rendering(dataset, gaussians, save)
