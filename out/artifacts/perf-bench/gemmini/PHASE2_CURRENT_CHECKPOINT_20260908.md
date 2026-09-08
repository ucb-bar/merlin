# Merlin Phase 2: current optimized compiler checkpoint

Date: 2026-09-08. Branch: `feat/target-generalization`.

## Outcome now

The current usable optimized compiler is:

```text
out/artifacts/perf-bench/gemmini/development_bf62_target_neutral_epilogue_fusion_20260908
```

Its compiler tree SHA-256 is
`0b1e14e9b61dd3a4eb3472a87702cc53c5a05b98198c08cec3bfbce85363db8a`.
Run both `verify.py` and `verify_native_scalar_epilogue.py` before use.
The small reviewable compiler delta is preserved in `native_scalar_epilogue_q534.patch`; the
curated hardware receipt is `native_scalar_epilogue_q534_receipt.json`.

The first substantial Phase-2 optimization replaces per-element software reconstruction of
round-to-nearest-even and float-to-signed-integer conversion with their standard target-neutral LLVM
operations. The RISC-V backend selects native `fcvt` instructions. This is a compiler lowering
repair, not a ResNet or Gemmini shape special case. It changes no accelerator command, DMA request,
tile, fence, or task boundary.

Exact warm/measured results:

| engine | predecessor | optimized | saved | reduction | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike comparative proxy | 976,190,118 | 751,827,143 | 224,362,975 | 22.9835% | 1.2984x |
| FireSim/U250 q530 -> q534 | 1,704,064,223 | **1,537,416,019** | **166,648,204** | **9.7795%** | **1.1084x** |

Both executions checked all 1,000 logits exactly: `bad=0`, `nonfinite=0`, top-1 258, checksum
`c6e777c3fe0aae90`. This also quantifies the proxy error: Spike ranked the optimization correctly
but overpredicted cycles saved by 1.3463x. Phase 2 therefore uses cheap engines to rank candidates,
not to publish absolute hardware cycles.

The sealed hardware package is:

```text
out/artifacts/perf-bench/gemmini/resnet50_merlin_phase2_native_scalar_ops_w8a8_warm_measured_firesim_candidate_20260908
```

Its `verify_bundle.sh` passes. The accepted receipt is
`validation/firesim_queue_job_534_success.json`, SHA-256
`0bf58c0304aed6aae5d5eec8e4d0e2a67edd065ab11591eed2b022a6395c3c6f`.
Queue job 534 used the required uninterrupted lifecycle:
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill`.
The package is sealed against unchanged resubmission.

## Invoke this compiler directly

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_bf62_target_neutral_epilogue_fusion_20260908
input=/absolute/path/to/model.mlir
output_dir=/absolute/path/to/output
mkdir -p "$output_dir"

MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  "$artifact/run-gemmini-opt" \
  --source-convolution \
  --convert-iface-to-gemmini \
  --emit-command-buffer="$output_dir/command_buffer.json" \
  --emit-target-artifact \
  -o "$output_dir/target.mlir" \
  "$input"
```

This emits the compiler-owned target LLVM MLIR and command buffer. Object/ELF construction remains
runner-owned because it must use the selected target's pinned compiler, ISA, linker script, harness
ABI, and hardware facts.

## Cross-model gate

The same compiler snapshot compiles all four complete portfolio graphs without L3 or FireSim:

| graph | source ops | mesh regions | host regions | status |
| --- | ---: | ---: | ---: | --- |
| ResNet-50 W8A8 | 1,240 | 54 | 119 | exact local + exact FireSim |
| TinyLLaMA | 718 | 0 | 247 | compiles; all-host gap remains |
| LSTMNetViT | 2,302 | 31 | 304 | compiles |
| SmolVLA denoise | 11,910 | 116 | 4,934 | compiles after generic `i1` storage sizing |

The non-ResNet command-buffer hashes are unchanged from the previous four-model checkpoint. This
optimization is therefore a general lowering improvement with a measured ResNet benefit; it does
not claim that the other models are already performance-complete.

## What q534 says to optimize next

q534 still executes 751,827,149 guest instructions and moves 1,142,808,576 read-DMA bytes plus
54,136,736 write-DMA bytes. `loop_matmul_active` is 53,479,788 cycles and reservation-station active
time is 53,892,695 cycles, only 3.51% of the measured interval. The accelerator schedule is not the
dominant remaining cost.

The macro-first order is therefore:

1. Form an exact target-neutral quantized epilogue across contraction -> per-channel affine/bias ->
   optional ReLU -> round/clamp -> i8. Delete the host pass and the i32 boundary.
2. Preserve/legalize the narrow NHWC/HWIO boundary so capability-selected native convolution can
   delete row-streamed im2col. The existing safe selector refuses current i32/NCHW boundaries.
3. Add a real second-tensor residual operation; a residual is not a bias epilogue.
4. Add explicit, default-off dynamic-activation quantization contracts for weight-only transformer
   contractions. Never claim source bit-equivalence where the source is f32.
5. Only then tune tiling, queue depth, and overlap. Removing blanket fences already measured only
   0.6277% on q530, so synchronization micro-tuning is not the first lever.

No further FireSim run is justified until a cheap four-model analysis shows a material new change
to host work, boundary bytes, im2col expansion, or accelerator occupancy. Phase 1 remains frozen at
92/96; Phase 2 does not rerun or repair it.

## Detailed experiment and figure handoff

The complete Phase-2 structure, trusted/agent-editable boundaries, search loop, objective evidence,
fast-cost hierarchy, four-model portfolio, and two paper-figure node/edge inventories are in:

```text
out/artifacts/perf-bench/gemmini/handoff_20260906/phase2_handoff.md
```

The original gap analysis and the corrected evidence record are in:

```text
out/artifacts/perf-bench/gemmini/handoff_20260906/closing_the_gap.md
```
