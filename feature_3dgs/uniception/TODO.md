# UniCeption 与原版 UFM 的数值差

和原版 UFM 对过同一张图。量化、归一化、resize 之后送进编码器的张量最大差为 0。差从编码器前向开始出现。

- `UniCeptionExtractor` 只编码一张图，batch 是 1。
- 原版 UFM 在 `_encode_image_pairs` 里把 source 和 target 在 batch 维拼成 2，一次前向算完再 `chunk` 开。

编码器就是 DINOv2，只吃一张图。拼 batch 是 UFM 为了一次前向算完两帧，不是编码器的输入格式。注意力和 LayerNorm 都在单张图内部，第二张图的像素不会进入第一张图的特征。两视图的交互在后面的 global-attention Transformer，提取器不跑那一层。

`batch=1` 和 `batch=2` 对不上，是因为这段前向包在 CUDA 的 bfloat16 autocast 里。patch 卷积和 SDPA 按 batch 形状选不同 kernel，累加顺序不同，所以同一张图的输出不是逐元素相同。

提取器保持单图 `batch=1`。不要为了逐元素对齐，用一张假的第二张图填成 `batch=2`。要和某一次 UFM 配对前向逐元素一致，只能把同一对 source/target 原样拼进去。
