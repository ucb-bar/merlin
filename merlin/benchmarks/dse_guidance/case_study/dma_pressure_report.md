# DMA / stream pressure report

> Structural data-movement streams each region implies, and which might justify a separate DMA/channel abstraction. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — needs a explicit design YAML. Streams a flat dequantized capture cannot size (scale sideband, KV, intermediate) are `unavailable`.

## Streams per region (bytes-known vs unavailable)

| stream | bytes-known | direction | candidate abstraction | prefetchable | overlap |
|---|---|---|---|---|---|
| weight | yes | read | prefetch_weight_once | yes | yes |
| activation_input | yes | read | double_buffered_activation_tile | unknown | yes |
| output | yes | write | output_commit_stream | no | yes |
| scale_sideband | unavailable | read | scale_sideband_stream | unknown | unknown |
| kv_prefix | unavailable | unavailable | kv_stream_handle | unknown | unknown |
| intermediate_writeback | unavailable | unavailable | activation_ring_buffer | unknown | unknown |
| command_descriptor | unavailable | read | multi_stream_dma_descriptor | unknown | unknown |

## Independent byte-carrying streams per workload

| workload | regions | byte-carrying streams (weight/act/output) |
|---|---|---|
| rdt | 1 | 3 |
| openvla | 2 | 6 |
| small_llama | 1 | 3 |
| tiny_llama | 1 | 3 |
| rdt2 | 1 | 3 |
| groot_n1d7 | 2 | 6 |
| molmoact | 1 | 3 |
| smolvla | 2 | 6 |
| pi05 | 2 | 6 |

## Findings

- **Three byte-carrying streams per region** (weight read, activation read, output write) structurally suggest a `multi_stream_dma_descriptor` with independent channels.
- **The weight stream is prefetchable and reused** (`prefetch_weight_once`); the activation stream structurally suggests a `double_buffered_activation_tile`.
- **Scale-sideband, KV, and intermediate streams are `unavailable`** — the capture is dequantized (scales erased), attention is lowered (no KV), and fused intermediates are not materialized. They are named, not invented.

## Missing for real bandwidth feasibility

- per-stream bandwidth and a target memory hierarchy (a design YAML) — absent here; **no bandwidth, channel count, or overlap feasibility is claimed.**
