import torch
from vggttt.nets.ttt import TTTOperator

from feature_3dgs.vggt.extractor import VGGTExtractor


class VGGTTTExtractor(VGGTExtractor):
    """Feature extractor based on VGG-T3 aggregator.

    Preprocessing matches ``VGGTExtractor``: center-pad to square (black) ->
    bicubic resize to *img_load_resolution* -> bilinear resize to 518x518 ->
    TTT aggregator -> crop valid patch tokens -> (D, h_p, w_p) feature map.

    VGG-T3 requires multiple images (multi-view aggregation). Use
    ``extract_all`` instead of ``__call__``.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "VGG-T3 requires multiple images. Use extract_all() instead."
        )

    def run_aggregator(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        if batch.device.type != "cuda":
            raise RuntimeError("VGG-T3 feature extraction requires a CUDA device")
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(batch.device)[0] >= 8
            else torch.float16
        )
        with torch.cuda.amp.autocast(dtype=dtype):
            compact_tokens, ps_idx, _ = self.model.aggregator(
                batch,
                attn_kwargs={"info": {"ttt_op_order": [
                    TTTOperator(start=0, end=None, compute_grad=True, update=True, apply=False),
                    TTTOperator(start=0, end=None, compute_grad=False, update=False, apply=True),
                ]}},
            )
        aggregated_tokens_list = [None] * self.model.aggregator.depth
        for layer_idx, tokens in zip(self.model.intermediate_layer_idx, compact_tokens, strict=True):
            aggregated_tokens_list[layer_idx] = tokens
        return aggregated_tokens_list, ps_idx
