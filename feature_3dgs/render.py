import os
from typing import Tuple
from tqdm import tqdm
import tifffile
import torch, torchvision
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from gaussian_splatting.utils import psnr, ssim, unproject
from gaussian_splatting.utils.lpipsPyTorch import lpips
from gaussian_splatting.dataset import CameraDataset
from feature_3dgs.prepare import prepare_feature_dataset, prepare_feature_gaussians
from feature_3dgs import FeatureGaussian
from feature_3dgs.decoder import AbstractDecoder

# TODO: add decoder
def prepare_rendering(
        sh_degree: int,
        source: str,
        device: str,
        trainable_camera: bool = False,
        load_ply: str = None,
        load_camera: str = None,
        load_mask=True,
        load_depth=True
) -> Tuple[CameraDataset, FeatureGaussian]:
    dataset = prepare_feature_dataset(
                source=source,
                device=device,
                trainable_camera=trainable_camera,
                load_camera=load_camera,
                load_mask=load_mask,
                load_depth=load_depth
              )
    gaussians = prepare_feature_gaussians(
                    sh_degree=sh_degree,
                    source=source,
                    device=device,
                    trainable_camera=trainable_camera,
                    load_ply=load_ply
                )
    return dataset, gaussians

def build_pcd(color: torch.Tensor, invdepth: torch.Tensor, mask: torch.Tensor, FoVx, FoVy) -> torch.Tensor:
    assert color.shape[-2:] == invdepth.shape[-2:], ValueError("Size of depth map should match color image")
    xyz = unproject(1 / invdepth, FoVx, FoVy)
    color = color.permute(1, 2, 0)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[mask, ...].cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color[mask, ...].cpu().numpy())
    return pcd

# TODO
def build_pcd_rescale(
        color: torch.Tensor, color_gt: torch.Tensor,
        invdepth: torch.Tensor, invdepth_gt: torch.Tensor, mask: torch.Tensor,
        FoVx, FoVy,
        rescale_depth_gt=True
) -> torch.Tensor:
    invdepth_gt_rescale = invdepth_gt
    mask = (mask > 1e-6)
    if rescale_depth_gt:
        mean_gt, std_gt = invdepth_gt.mean(), invdepth_gt.std()
        mean, std = invdepth.mean(), invdepth.std()
        invdepth_gt_rescale = (invdepth_gt - mean_gt) / std_gt * std + mean
    pcd = build_pcd(color, invdepth, mask, FoVx, FoVy)
    pcd_gt = build_pcd(color_gt, invdepth_gt_rescale, mask, FoVx, FoVy)
    return pcd, pcd_gt, invdepth_gt_rescale

# TODO: add decoder
# initialize Decoder in render
def rendering(views: CameraDataset,
              gaussians: FeatureGaussian,
              render_path: str,
              decoder: AbstractDecoder | None = None):

    depth_path = os.path.join(render_path, "depth")
    feature_path = os.paht.join(render_path, "features")
    ground_truth_path = os.path.join(render_path, "ground_truth")

    for index, view in enumerate(tqdm(views, desc="rendering progress")):
        render_package = gaussians(view)
        gt = view.original_image[0 : 3, :, :]
        gt_feature_map = view.semantic_feature.cuda()
        torchvision.utils.save_image(render_package["render"],
                                     os.pasth.join(render_path, '{0:05d}.png'.format(index)))
        torchvision.utils.save_image(gt, os.path.join(ground_truth_path, '{0:05d}.png'.format(index)))

        depth = render_package["depth"]
        scale_nor = depth.max().item()
        depth_nor = depth / scale_nor
        depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        colormap = plt.get_cmap('jet')
        depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        depth_colored_rgb = depth_colored[:, :, :3]
        depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        output_path = os.path.join(depth_path, '{0:05d}.png'.format(index))
        depth_image.save(output_path)
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
        if decoder is not None:
            feature_map = decoder(feature_map)

        # TODO: visualize and save feature maps

        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(feature_path, '{0:05d}_fmap_CxHxW.pt'.format(index)))

# TODO: add decoder
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("--no_rescale_depth_gt", action="store_true")
    parser.add_argument("--save_depth_pcd", action="store_true")
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera,
            load_mask=not args.no_image_mask, load_depth=args.save_depth_pcd)
        rendering(dataset, gaussians, save, save_pcd=args.save_depth_pcd, rescale_depth_gt=not args.no_rescale_depth_gt)