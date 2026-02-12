# FeatureGS (Packaged Python Version)

### Development Install

```shell
pip install --upgrade git+https://github.com/facebookresearch/dinov3@main
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --target . --upgrade . --no-deps
```

## Download model

Request access and download [dinov3](https://github.com/facebookresearch/dinov3) to `checkpoints/`:

```
checkpoints
 |- dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
 |- dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
 |- dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
 |- dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
 |- dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
 |- dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
 |- dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
 |- dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
 |- dinov3_vits16_pretrain_lvd1689m-08c60483.pth
 |- dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```