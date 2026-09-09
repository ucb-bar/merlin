# Immutable pre-run structural baseline

This baseline is bound to compiler tree `f8992bc35be76e805dc8dce9bfe5864167d50f0276da685306a2fb7d9dddfa36`. It is the comparison point for the next four-model Phase-2 authoring run. These are compiler and structural measurements, not end-to-end timing estimates.

| Model | Fresh bounded compile | Accelerator coverage | Principal structural debt |
|---|---:|---:|---|
| ResNet-50 W8A8 | passed in 39.45 s | 54/105 tasks; 53 convolution regions | 974 residual host operations; 89 required layout conversions; 53,130,988 source-IR conversion bytes; 71 unattributed boundaries |
| TinyLLaMA | passed in 3.41 s with explicit dynamic-weight contract | 15/31 tasks | 763 residual host operations; physical-layout census is not applicable because there is no ranked convolution activation |
| LSTMNetViT W8A8 | target emission exceeded the 60 s bounded compile | physical planning covers 11/11 convolution regions | 23 required layout conversions; 2,611,424 source-IR conversion bytes; 70 unattributed, 26 unknown, and 29 unsafe-view boundaries |
| SmolVLA denoise step W8A8 | target emission exceeded the 60 s bounded compile | convolution-only physical-layout census not applicable | large non-convolution graph: 11,911 top-level operations, including batch matmul, integer matmul, reductions, views, and elementwise regions |

ResNet's exact-default global quantization analysis found four candidate chains and 42,549,248 potentially eliminable materialization bytes, but selected none. The explicit blockers are: an accuracy contract, target capability, resident-capacity proof, and bounded source-quantizer saturation. This is opportunity evidence only; it is not a speedup claim.

The current emitted ResNet command buffer contains 53 `CONV2D`, one `MATMUL_RESIDENT`, 54 `RES_PACK`, and one `COMMIT` command. The physical-layout plans are structural and have not yet been lowered into runtime allocation/index rewrites; their conversion-byte totals are source-IR storage volumes, not measured DMA traffic.

Authoritative evidence:

- `../development_phase2_integrated_macro_v1_20260908/validation/integrated_receipt.json`
- `../development_phase2_integrated_macro_v1_20260908/validation/physical_layout/resnet50_report.json`
- `../development_phase2_integrated_macro_v1_20260908/validation/physical_layout/lstmnetvit_report.json`
- `../development_phase2_integrated_macro_v1_20260908/validation/resnet50/command_buffer.json`

Each retained round must compare against its own preceding revision. Promotion requires exact semantic and emission evidence across the portfolio; unlike model costs are never summed into one synthetic score.
