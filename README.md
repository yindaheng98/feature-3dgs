# Feature 3DGS (Packaged Python Version)

This repo is the **refactored Python training and inference code for [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)**.
Built on top of [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), we **reorganised the original code as a standard Python package** with a modular Extractor-Decoder architecture, making it easy to swap foundation models without changing the core pipeline.

Each Gaussian point carries a learnable semantic embedding alongside standard 3DGS attributes. A frozen **Extractor** produces ground-truth feature maps from training images, while a lightweight learnable **Decoder** maps the rasterised per-point embeddings back to the extractor's feature space. The framework is backbone-agnostic: new foundation models can be plugged in by implementing an Extractor-Decoder pair and registering it.

## Features

* [x] Organised as a standard Python package with `pip install` support
* [x] Modular Extractor-Decoder architecture for plugging in arbitrary foundation models
* [x] Built-in DINOv3 support (ViT and ConvNeXt backbones)
* [x] Auto-registration pattern — add new models with zero changes to core code
* [x] PCA-based feature visualisation for both ground-truth and rendered feature maps
* [x] All training modes from upstream: base, densify, camera, camera-densify

## Install

### Prerequisites

* [Pytorch](https://pytorch.org/) (>= v2.4 recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, match with PyTorch version)
* [gsplat](https://github.com/nerfstudio-project/gsplat)

### Development Install

```shell
pip install --upgrade git+https://github.com/facebookresearch/dinov3@main
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --target . --upgrade . --no-deps
```

### Download Checkpoints

Request access and download [DINOv3](https://github.com/facebookresearch/dinov3) weights to `checkpoints/`:

```
checkpoints/
 ├── dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
 ├── dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
 ├── dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
 ├── dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
 ├── dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
 ├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
 ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
 ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
 ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
 ├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
 └── ...
```

## Command-Line Usage

### Visualise Extractor Output

Verify that the extractor produces meaningful features before training:

```shell
python -m feature_3dgs.show \
    --name dinov3_vitl16 \
    -s data/truck -d output/truck-dinov3_vitl16 \
    -o checkpoint_dir="'checkpoints'"
```

### Train

```shell
python -m feature_3dgs.train \
    --name dinov3_vitl16 --embed_dim 32 \
    -s data/truck -d output/truck-semantic -i 30000 \
    --mode densify
```

### Render

```shell
python -m feature_3dgs.render \
    --name dinov3_vitl16 --embed_dim 32 \
    -s data/truck -d output/truck-semantic -i 30000
```

Rendered feature maps are PCA-projected to RGB and saved alongside ground-truth feature visualisations.

## API Usage

### Dataset & Decoder

```python
from feature_3dgs.prepare import prepare_dataset_and_decoder

dataset, decoder = prepare_dataset_and_decoder(
    name="dinov3_vitl16",   # registered extractor-decoder name
    source="data/truck",
    embed_dim=32,
    device="cuda",
)
# dataset is a FeatureCameraDataset; each camera carries a 'feature_map' in custom_data
# decoder is the learnable AbstractFeatureDecoder
```

### Gaussian Model

```python
from feature_3dgs.prepare import prepare_gaussians

gaussians = prepare_gaussians(
    decoder=decoder, sh_degree=3,
    source="data/truck", device="cuda",
)
```

`SemanticGaussianModel` extends `GaussianModel` with `_semantic_features` (per-point learnable embeddings) and a `_decoder`. During rendering, the rasteriser splatts the semantic features into a 2D feature map, and the decoder transforms it to match the extractor's output space.

### Training

```python
from feature_3dgs.prepare import prepare_trainer

trainer = prepare_trainer(gaussians, dataset, mode="densify")
for camera in dataset:
    loss, out = trainer.step(camera)
```

### Inference

```python
import torch
with torch.no_grad():
    for camera in dataset:
        out = gaussians(camera)
        rgb = out["render"]           # (3, H, W)
        feat = out["feature_map"]     # (D, H', W')
```

### Save & Load

```python
gaussians.save_ply("output/point_cloud.ply")
# also saves point_cloud.ply.semantic.pt and point_cloud.ply.decoder.pt

gaussians.load_ply("output/point_cloud.ply")
```

## Design: Extractor & Decoder

The core abstraction decouples **what features to distill** (Extractor) from **how to map rasterised embeddings back** (Decoder).

### Extractor (`AbstractFeatureExtractor`)

The extractor is a **frozen** foundation model that converts training images into dense feature maps. It runs **only on the dataset side** — each training view is processed once, cached, and served as the ground-truth supervision signal.

```
Image (C, H, W)  ──► Extractor (frozen) ──► Feature Map (D, H', W')
```

The extractor defines the target feature space (dimension `D` and spatial resolution `H'×W'`). It is never updated during training.

### Decoder (`AbstractFeatureDecoder`)

The decoder is a **learnable** module that sits **inside the rendering pipeline**. After the Gaussian rasteriser splatts per-point semantic embeddings (dimension `embed_dim`) into a 2D feature map at image resolution, the decoder transforms this rasterised map to match the extractor's output:

```
Per-point embeddings ──► Rasteriser ──► Raw Feature Map (embed_dim, H, W)
                                              │
                                              ▼
                                        Decoder (learnable)
                                              │
                                              ▼
                                    Decoded Feature Map (D, H', W')
```

The training loss is `L1(Decoded Feature Map, Extractor Feature Map)`. The decoder's role is to bridge the gap between the compact per-point embedding (`embed_dim`, typically 32) and the extractor's high-dimensional output (`D`, e.g. 1024 for ViT-L), while also handling any spatial resolution change.

### Why this split?

1. **Memory efficiency**: Only `embed_dim` channels are stored per Gaussian and rasterised, not the full `D` channels. The decoder upprojects after rasterisation.
2. **Spatial alignment**: Foundation models often output at patch resolution (e.g. 1/16 for ViT). The decoder can downsample the rasterised full-resolution map to match, avoiding expensive full-resolution feature supervision.
3. **Modularity**: Swapping the foundation model only requires a new Extractor-Decoder pair. The Gaussian model, trainer, and rendering pipeline remain unchanged.

## Extending: Adding a New Foundation Model

The project uses an **auto-registration** pattern. To add support for a new model (e.g. a hypothetical `MyModel`), follow the DINOv3 implementation as a reference:

### Step 1: Implement the Extractor

Create `feature_3dgs/mymodel/extractor.py`:

```python
import torch
from feature_3dgs.extractor import AbstractFeatureExtractor

class MyModelExtractor(AbstractFeatureExtractor):
    def __init__(self, model, ...):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: (C, H, W) in [0, 1]
        # Return: (D, H', W') feature map
        ...

    def to(self, device) -> 'MyModelExtractor':
        self.model.to(device)
        return self
```

### Step 2: Implement the Decoder

Create `feature_3dgs/mymodel/decoder.py`:

```python
import torch
import torch.nn as nn
from feature_3dgs.decoder import NoopFeatureDecoder

class MyModelDecoder(NoopFeatureDecoder):
    def __init__(self, in_channels: int, out_channels: int, ...):
        super().__init__(embed_dim=in_channels)
        # Build a small network that maps
        #   (in_channels, H, W) -> (out_channels, H', W')
        # to match the extractor's spatial and channel dimensions.
        self.net = nn.Sequential(...)

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map: (in_channels, H, W), same resolution as input image
        # Return: (out_channels, H', W'), matching extractor output
        return self.net(feature_map.unsqueeze(0)).squeeze(0)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def parameters(self):
        return self.net.parameters()
```

The key design constraint: **the decoder's output spatial size and channel count must exactly match the extractor's output**, so that L1 loss can be computed directly.

For example, the DINOv3 ViT extractor outputs at patch resolution `(D, H/16, W/16)`, so `DINOv3CNNDecoder` uses a `Conv2d(kernel_size=16, stride=16)` to downsample the full-resolution rasterised map to the same grid.

### Step 3: Register via Factory

Create `feature_3dgs/mymodel/registry.py`:

```python
from feature_3dgs.registry import register_extractor_decoder
from .extractor import MyModelExtractor
from .decoder import MyModelDecoder

FEATURE_DIM = 768  # D of your model's output

def factory(embed_dim: int, **configs):
    extractor = MyModelExtractor(...)
    decoder = MyModelDecoder(
        in_channels=embed_dim,
        out_channels=FEATURE_DIM,
        ...
    )
    return extractor, decoder

register_extractor_decoder("mymodel", factory)
```

### Step 4: Trigger Registration on Import

Create `feature_3dgs/mymodel/__init__.py`:

```python
from . import registry  # triggers register_extractor_decoder() at import time
```

Then add the import in `feature_3dgs/__init__.py`:

```python
from . import mymodel  # auto-registers "mymodel"
```

After these steps, the new model is available everywhere:

```shell
python -m feature_3dgs.train --name mymodel --embed_dim 32 -s data/truck -d output/truck-mymodel -i 30000
```

## Acknowledgement

This repo is developed based on [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [gaussian-splatting (packaged)](https://github.com/yindaheng98/gaussian-splatting). Many thanks to the authors for open-sourcing their codebases.

# Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields

Shijie Zhou, Haoran Chang\*, Sicheng Jiang\*, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi (\* indicates equal contribution)<br>
| [Webpage](https://feature-3dgs.github.io/) | [Full Paper](https://arxiv.org/abs/2312.03203) | [Video](https://www.youtube.com/watch?v=h4zmQsCV_Qw) | [Original Code](https://github.com/ShijieZhou-UCLA/feature-3dgs) |

Abstract: *3D scene representations have gained immense popularity in recent years. Methods that use Neural Radiance fields are versatile for traditional tasks such as novel view synthesis. In recent times, some work has emerged that aims to extend the functionality of NeRF beyond view synthesis, for semantically aware tasks such as editing and segmentation using 3D feature field distillation from 2D foundation models. However, these methods have two major limitations: (a) they are limited by the rendering speed of NeRF pipelines, and (b) implicitly represented feature fields suffer from continuity artifacts reducing feature quality. Recently, 3D Gaussian Splatting has shown state-of-the-art performance on real-time radiance field rendering. In this work, we go one step further: in addition to radiance field rendering, we enable 3D Gaussian splatting on arbitrary-dimension semantic features via 2D foundation model distillation. This translation is not straightforward: naively incorporating feature fields in the 3DGS framework encounters significant challenges, notably the disparities in spatial resolution and channel consistency between RGB images and feature maps. We propose architectural and training changes to efficiently avert this problem. Our proposed method is general, and our experiments showcase novel view semantic segmentation, language-guided editing and segment anything through learning feature fields from state-of-the-art 2D foundation models such as SAM and CLIP-LSeg. Across experiments, our distillation method is able to provide comparable or better results, while being significantly faster to both train and render. Additionally, to the best of our knowledge, we are the first method to enable point and bounding-box prompting for radiance field manipulation, by leveraging the SAM model.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{zhou2024feature,
  title={Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields},
  author={Zhou, Shijie and Chang, Haoran and Jiang, Sicheng and Fan, Zhiwen and Zhu, Zehao and Xu, Dejia and Chari, Pradyumna and You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21676--21685},
  year={2024}
}</code></pre>
  </div>
</section>
