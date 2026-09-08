# Radiance semantic-family LayerNorm L3 qualification (2026-09-08)

## Outcome

Merlin parsed and lowered the public `RP5_layernorm_fp32_pt` workload, reconstructed the semantic
request `layernorm(fp32, rows=16, cols=16)` from only its command/tensor ABI, derived Radiance SIMT
capability from the hardware contract, selected `kernels/layernorm`, and dispatched the existing native
LLVM-dialect MLIR emitter.

| Arm | Private oracle change | GSIM verdict | Elements | Active simulation |
|---|---|---:|---:|---:|
| positive | none | **PASS** | 256/256 | 131.252 s |
| negative control | first expected value +100 | **FAIL (expected)** | 256 | 129.079 s |

The arms have identical public command buffers, selection reports, emitted MLIR, and submitted Muon ELF
(`9a38cc3dda048de035a2f1dee552a04db5623991d87ae09c4472adb4691cfcac`). No PR reference source is
copied, linked, called, or dispatched, and expected values are absent from the public compiler artifact.

## Why WS MXFP8 was not selected next

`kernels/gemm_mxgemmini_ws` ranks above LayerNorm for its qualified `M=256, N=64, K%32=0` contract, but
the current compiler cannot produce an independent qualifying artifact for it:

- `muon_codegen_mlir.emit_kernel_mlir` returns a non-MLIR placeholder for every MX command buffer.
- The downstream fallback in `muon_harness.program_from_cb` calls `muon_mx_codegen.emit_mx_kernel`, whose
  own contract identifies it as a public-capsule reference path and requires golden-only operand codes.
- Its current packing width is fixed at 128, smaller than the selected WS strategy's required M=256.

Using that path would violate this task's prohibition on reference dispatch and answer-derived inputs.
LayerNorm is the next priority-80 SmolVLA family already reachable through a genuine compiler emitter.

This L3 pair qualifies only the generated 16x16 fp32 LayerNorm implementation. Selection metadata is not
itself numeric qualification, and simulator wall time is not a kernel-performance measurement.

## Replay

The reusable replay driver lives in the preceding bias-add evidence directory. Invoke it with:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/l3_semantic_bias_add_20260908/run_qualification.py \
  --capsule RP5_layernorm_fp32_pt --case positive \
  --emulator /scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator \
  --work /scratch/agustin/tmp/radiance-layernorm-positive-replay \
  --publish /scratch/agustin/tmp/radiance-layernorm-positive-public
```

Use `--case negative` for the fail-capable control. Offline verification:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/l3_semantic_layernorm_20260908/verify.py
```
