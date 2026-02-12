# Copy from https://github.com/facebookresearch/dinov3/blob/54694f7627fd815f62a5dcc82944ffa6153bbb76/hubconf.py
from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)
from dinov3.hub.classifiers import dinov3_vit7b16_lc
from dinov3.hub.detectors import dinov3_vit7b16_de
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.hub.segmentors import dinov3_vit7b16_ms

from dinov3.hub.depthers import dinov3_vit7b16_dd
