# Gemmini-friendly per-tensor ResNet-50 investigation

## Outcome

The experiment produced a complete compiler artifact, but it is **not a performance candidate**.
The portable per-tensor W8A8 capture covers all 53 convolutions and the final linear layer, and the
compiler places all 54 resulting integer contractions on Gemmini. The full compiler finishes in
46.90 seconds after fixing superlinear task-provenance tagging.

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
| Residual host source operations | 4,348 | 2,850 |
| Planned host spill bytes | 1,060,250,304 | 28,559,808 |
| Native narrow epilogues | 0 | 0 |

The final command buffer contains 162 commands: 54 `RES_PACK`, 54 `MATMUL_RESIDENT`, and 54
`COMMIT`. These are 54 contraction tasks, not 54 direct convolution tasks. model2MLIR has already
normalized each source convolution into im2col/view operations plus a rank-2 matmul, so direct
`LOOP_CONV` recovery is a later graph rewrite. Each accelerator task is still separated by host
work because no native output epilogue forms.

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
  /path/to/linalg.mlir --out validation/structural_census_after.json
```

Run focused regression tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=compiler:../../../../../merlin/python \
../../../../../.venv/bin/pytest -q tests
```

See `validation/full_compile_receipt.json` for hashes and exact resource use, and `NEXT_STEPS.md`
for the native epilogue/residency seam. No FireSim, private payload, or 92/96 compiler was used or
modified.
