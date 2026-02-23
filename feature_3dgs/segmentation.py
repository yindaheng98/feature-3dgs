import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from feature_3dgs import get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.render import prepare_rendering, rendering


def get_feature(dataset: FeatureCameraDataset, image_index: int, x: int, y: int) -> torch.Tensor:
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


def compute_similarity_map(query: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
    """Per-patch cosine similarity between *query* (D,) and *feature_map* (D, H_p, W_p)."""
    D, H_p, W_p = feature_map.shape
    features = feature_map.reshape(D, -1).T
    sim = F.cosine_similarity(query.unsqueeze(0), features, dim=1)
    return sim.reshape(H_p, W_p)


def segment_image(image: torch.Tensor, similarity: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mask out pixels whose similarity is below *threshold*. Returns (3, H, W)."""
    H, W = image.shape[1], image.shape[2]
    sim_full = F.interpolate(similarity.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    out = image.clone()
    out[:, sim_full < threshold] = 0
    return out


def show_segmentation(image: torch.Tensor, similarity: torch.Tensor, segmented: torch.Tensor, x: int, y: int, threshold: float) -> None:
    """Show selected image, cosine similarity heatmap with red cross, and segmented image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    axes[0].imshow(image.permute(1, 2, 0).cpu().clamp(0, 1).numpy())
    axes[0].set_title("Selected Image")
    axes[0].axis("off")

    H, W = image.shape[1], image.shape[2]
    im = axes[1].imshow(similarity.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1, extent=[0, W, H, 0])
    axes[1].plot(x, y, 'r+', markersize=20, markeredgewidth=3)
    axes[1].set_title("Cosine Similarity Heatmap")
    axes[1].axis("off")

    axes[2].imshow(segmented.permute(1, 2, 0).cpu().clamp(0, 1).numpy())
    axes[2].set_title(f"Segmented (threshold={threshold:.2f})")
    axes[2].axis("off")

    fig.tight_layout()
    plt.show()


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
        similarity_map = compute_similarity_map(feature, dataset[args.image_index].custom_data['feature_map'])
        img = dataset[args.image_index].ground_truth_image  # (3, H, W)
        img_seg = segment_image(img, similarity_map, args.threshold)
        show_segmentation(img, similarity_map, img_seg, args.x, args.y, args.threshold)

        rendering(dataset, gaussians, save)
