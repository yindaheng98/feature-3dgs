import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from feature_3dgs import get_available_extractor_decoders
from feature_3dgs.prepare import prepare_dataset_and_decoder
from feature_3dgs.render import build_linear_for_visualization, colorize_feature_map


def show_dataset(dataset, destination: str):
    """Compute PCA on all feature maps in the dataset, then save projected images."""
    weight, bias = build_linear_for_visualization(dataset, gaussians=None)

    # ---- Save visualisations ----
    pbar = tqdm(dataset, dynamic_ncols=True, desc="Saving feature maps")
    for idx, camera in enumerate(pbar):
        feature_map = camera.custom_data['feature_map']
        projected_image = colorize_feature_map(feature_map, weight, bias)  # (3, H, W)

        # Save the PCA visualisation
        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(8, 4))
        gt = camera.ground_truth_image
        axes[0].imshow(gt.permute(1, 2, 0).cpu().clamp(0, 1).numpy())
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(projected_image.permute(1, 2, 0).cpu().clamp(0, 1).numpy())
        axes[1].set_title("PCA Projection")
        axes[1].axis("off")

        image_name = os.path.splitext(os.path.basename(dataset[idx].ground_truth_image_path))[0]
        fig.suptitle(image_name, fontsize=10)
        fig.tight_layout()
        save_path = os.path.join(destination, f"{image_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"Done - {len(dataset)} images saved to {destination}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", choices=get_available_extractor_decoders(), type=str)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--embed-dim", default=3, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_cache_device", default="cpu", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, decoder = prepare_dataset_and_decoder(
        name=args.name, source=args.source, embed_dim=args.embed_dim, device=args.device,
        dataset_cache_device=args.dataset_cache_device,
        trainable_camera=False, load_camera=None, load_mask=False, load_depth=False,
        configs=configs,
    )
    del decoder
    torch.cuda.empty_cache()  # Clear GPU memory before starting the show process
    with torch.no_grad():
        show_dataset(dataset, destination=args.destination)
