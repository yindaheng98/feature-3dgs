import time
import os

import torch
import viser
import nerfview

from gaussian_splatting import build_camera
from gaussian_splatting.utils import focal2fov
from feature_3dgs import SemanticGaussianModel, get_available_extractor_decoders
from feature_3dgs.extractor import FeatureCameraDataset
from feature_3dgs.render import prepare_rendering
from feature_3dgs.utils import pca_transform_params


@torch.no_grad()
def viewer_render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
        gaussians: SemanticGaussianModel, device: str,
        pca_weight: torch.Tensor, pca_bias: torch.Tensor,
        bg_color=(0., 0., 0.)):
    if render_tab_state.preview_render:
        width = render_tab_state.render_width
        height = render_tab_state.render_height
    else:
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height

    c2w = camera_state.c2w  # [4, 4] numpy float64
    K = camera_state.get_K((width, height))  # [3, 3] numpy float64

    c2w_torch = torch.from_numpy(c2w).float().to(device)
    w2c = torch.linalg.inv(c2w_torch)
    R = w2c[:3, :3]
    T = w2c[:3, 3]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    camera = build_camera(
        image_height=int(height), image_width=int(width),
        FoVx=FoVx, FoVy=FoVy,
        R=R, T=T,
        bg_color=bg_color, device=device,
    )
    print(f"Resolution: {width}x{height}")

    out = gaussians.forward_linear_projection(camera, weight=pca_weight, bias=pca_bias)
    feature_map = out["feature_map"]  # (3, H, W)
    rgb = torch.sigmoid(feature_map * 2.0)

    return rgb.permute(1, 2, 0).cpu().numpy()


def viewing(
        gaussians: SemanticGaussianModel,
        dataset: FeatureCameraDataset,
        device: str,
        port: int = 8080, bg_color=(0., 0., 0.)) -> None:
    pca_weight, pca_bias = pca_transform_params(dataset, n_components=3)

    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=lambda cs, rts: viewer_render_fn(
            cs, rts, gaussians, device, pca_weight, pca_bias, bg_color),
        mode="rendering",
    )
    print(f"Viewer running on port {port}... Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        pass


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
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-e", "--option_extractor", default=[], action='append', type=str)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
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
        viewing(gaussians, dataset, device=args.device, port=args.port)
