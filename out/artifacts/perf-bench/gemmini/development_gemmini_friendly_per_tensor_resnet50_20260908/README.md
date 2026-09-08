# Gemmini-friendly per-tensor ResNet-50 investigation

## Outcome

The experiment produced a complete compiler artifact, but it is **not a performance candidate**.
The portable per-tensor W8A8 capture covers all 53 convolutions and the final linear layer, and the
compiler places all 54 resulting integer contractions on Gemmini. The full compiler finishes in
46.90 seconds after fixing superlinear task-provenance tagging. A subsequent explicit
`native_aligned_i32_bias_scalar_requant_v1` run implements accumulator-unit bias preload and native
scalar narrowing. It recovers the terminal FC epilogue and completes in 52.38 seconds; it does not
weaken the deployment rejection below.

The deployment gate rejects the quantization contract: on the one measured image, top-1 remains
258 but cosine similarity to FP32 is only 0.8852357 and maximum absolute logit error is 1.8771.
ExecuTorch execution and full ImageNet validation are also absent. This result must not replace the
canonical PT2E graph or support a speed/accuracy claim.

## Exact structural result

| Measure | Before exact layout reuse | After |
|---|---:|---:|
| Accelerator tasks | 54 | 54 |
| Host tasks | 55 | 55 |
| Maximal accelerator regions | 54 | 54 |
| Residual host source operations | 4,348 | 2,826 |
| Planned host spill bytes | 1,060,250,304 | 28,555,648 |
| Native narrow epilogues | 0 | 1 (terminal FC) |

The native epilogue pass absorbs 9 source operations at the FC, removes 24 residual host source
operations, 3 spill tensors, 4,160 padded spill bytes, and one 4,000-byte i32 boundary (3,000 bytes
of output DMA). The task and region counts do not fall because the returned logits still require a
terminal i8-to-f32 host dequantization.

The final command buffer contains 162 commands: 54 `RES_PACK`, 54 `MATMUL_RESIDENT`, and 54
`COMMIT`. These are 54 contraction tasks, not 54 direct convolution tasks. model2MLIR has already
normalized each source convolution into im2col/view operations plus a rank-2 matmul, so direct
`LOOP_CONV` recovery is a later graph rewrite. Each accelerator task is still separated by host
work around the convolution layers. The one FC native epilogue is a real i32 repeating-D bias
preload followed by `CONFIG_ST` scalar scale, round-even/saturating i8 readout.

All 53 convolution epilogues fail closed with `bias_axis_not_gemmini_column`: their im2col form is
`[Co,K] x [K,P] -> [Co,P]`, while Gemmini's repeating D preload broadcasts a length-N vector down
M. Recovering those sites requires direct source-convolution recovery or a globally propagated
`[P,Co]`/NHWC physical layout; treating the row bias as a column bias would be incorrect.

The bundled integer preparation improvement commutes calibrated symmetric per-tensor QDQ through proven
layout-only collapse, expand, transpose, zero-pad, and gather chains. It reuses 108/108 calibrated
operands and erases 462 replaced f32-layout operations. Unknown transforms, nonzero padding, and
nonzero zero-points fail closed.

## Compiler completion fix

Whole-model emission previously rescanned every accumulated CFG block after each scheduled task to
attach `merlin.global_task`. This grows as tasks times accumulated blocks and exceeded the 120-second
cap. `FnBuilder` now applies scoped provenance when each operation is inserted, so the same work is
linear in emitted operations. The full run completed in 46.90 seconds, used 838,000 KB maximum RSS,
and emitted a verified 68,293,031-byte LLVM MLIR module. All task IDs from -1 through 108 occur in
the output. The compressed target is stored alongside the schema-valid command buffer.

## Reproduce

The capture needs the model2MLIR environment with PyTorch/TorchAO:

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/agustin/projects/model2MLIR/.venv/bin/python \
  validation/capture_native_aligned_resnet50.py --out validation/native_aligned_capture
```

Run the fail-closed gate (exit 2 is the expected current result):

```sh
python validation/evaluate_deployment_gate.py \
  --receipt validation/capture_receipt.json
```

Produce a placement census without LLVM emission:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python validation/run_structural_census.py \
  --native-aligned-epilogue /path/to/linalg.mlir \
  --out validation/structural_census_i32_epilogue.json
```

Run focused regression tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=compiler:../../../../../merlin/python \
../../../../../.venv/bin/pytest -q tests
```

See `validation/full_compile_receipt.json` for the initial compiler evidence and
`validation/i32_epilogue_receipt.json` for the native epilogue hashes, census, and resource use.
No FireSim, private payload, or 92/96 compiler was used or modified.
