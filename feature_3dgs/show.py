import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from feature_3dgs import get_available_extractor_decoders
from feature_3dgs.prepare import prepare_dataset_and_decoder


def show_dataset(dataset, destination: str):
    """Compute PCA on all feature maps in the dataset, then save projected images.

    The dataset is iterated twice (relies on built-in caching):
      1st pass - collect all patch features and fit a PCA.
      2nd pass - transform each feature map with the fitted PCA and save the
                 visualisation to *destination*.
    """
    # ---- First pass: gather all features and fit PCA ----
    all_features = []
    print("Pass 1/2: collecting features for PCA fitting ...")
    for idx in tqdm(range(len(dataset)), desc="Extracting"):
        feature_map = dataset[idx].feature_map  # (D, H_patches, W_patches)
        D, _, _ = feature_map.shape
        features = feature_map.reshape(D, -1).permute(1, 0).cpu()  # Reshape to (H*W, D)
        all_features.append(features)
    all_features_np = torch.cat(all_features, dim=0).numpy()  # (N_total, D)

    pca = PCA(n_components=3, whiten=True)
    pca.fit(all_features_np)
    print(f"PCA fitted - explained variance ratio: {pca.explained_variance_ratio_}")

    # ---- Second pass: transform and save visualisations ----
    print("Pass 2/2: projecting features and saving images ...")
    for idx in tqdm(range(len(dataset)), desc="Saving"):
        feature_map = dataset[idx].feature_map  # (D, H_patches, W_patches)
        D, H, W = feature_map.shape
        features = feature_map.reshape(D, -1).permute(1, 0).cpu().numpy()  # (H*W, D)
        # PCA projection -> (H*W, 3) -> (H, W, 3)
        projected = torch.from_numpy(pca.transform(features)).view(H, W, 3)
        # Vibrant colours: multiply by 2 and pass through sigmoid (following the notebook)
        projected_image = torch.sigmoid(projected * 2.0)  # (H, W, 3)

        # Save the PCA visualisation
        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(8, 4))
        gt = dataset[idx].ground_truth_image
        axes[0].imshow(gt.permute(1, 2, 0).cpu().clamp(0, 1).numpy())
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(projected_image.numpy())
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
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, decoder = prepare_dataset_and_decoder(
        name=args.name, source=args.source,
        embed_dim=args.embed_dim, device=args.device, **configs
    )
    del decoder
    torch.cuda.empty_cache()  # Clear GPU memory before starting the show process
    with torch.no_grad():
        show_dataset(dataset, destination=args.destination)
