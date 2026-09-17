# Merlin Phase 2: current optimized compiler checkpoint

Date: 2026-09-08. Branch: `feat/target-generalization`.

## Outcome now

The current usable optimized compiler is the final combined multi-model assembly:

```text
out/artifacts/perf-bench/gemmini/development_phase2_final_combined_exact_residual_global_encoding_20260908
```

Its compiler tree SHA-256 is
`48694957d14c9608960f7ac2b6cd08b72b15c55e4a11ff26e1cf952e0cd7607e` over 45 files.
Run `verify.py` before use; it binds the 65-test suite, two exact warm/reentrant witnesses, all four
complete model compiles, the opt-in bridge, source ownership, encoding/refusal evidence, the
review patch, and hardware lineage. This assembly contains q534 native-scalar lowering, q535
generic affine-im2col, exact ordered non-residual epilogues, exact true two-tensor residual fusion,
target-neutral global encoding/lifetime planning, the fail-closed native `LOOP_CONV_WS` route, and
the default-off dynamic-weight contraction bridge under one compiler identity.

The final ResNet target is byte-identical to the sealed residual child, not q535. Its exact local
same-process warm/measured result therefore transfers as 492,147,976 Spike proxy cycles versus
506,265,226 for q535's compiler (14,117,250 saved; 2.7885%; 1.028685x), with 1,000/1,000 logits
exact. No hardware number transfers to this changed object. The latest honest full-Merlin hardware
checkpoint remains q535 at 1,316,619,699 cycles.

The final reviewable compiler delta is `final_combined_compiler.patch` (SHA-256
`0e611e3c9c2231196653bfdb906a45843b89f8ee14f4ffbefe683116a9f2477a`). Detailed evidence is in
`FINAL_COMBINED.md` and `validation/final_combined_receipt.json` inside the artifact. The original
canonical and affine assemblies remain preserved separately for lineage.

The standalone native-scalar source artifact remains
`development_bf62_target_neutral_epilogue_fusion_20260908`. The small reviewable compiler delta is
preserved in `native_scalar_epilogue_q534.patch`; the
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

## Newest hardware result: affine im2col spans

Queue job 535 hardware-qualified the next target-neutral full-model lever. Static convolution
geometry now classifies affine interior spans once, hoists row-base/y decisions, and retains
guarded gathers only at borders. This removes repeated address, bounds, and clamp work for 93.26%
of 14,613,760 packed bytes; 87.30% of bytes use fully guardless interior loops. It adds no model,
layer, shape, or Gemmini schedule special case.

| comparison | predecessor | q535 | saved | reduction | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike q534 -> affine im2col | 751,827,143 | 506,265,226 | 245,561,917 | 32.6620% | 1.48505x |
| FireSim q534 -> q535 | 1,537,416,019 | **1,316,619,699** | **220,796,320** | **14.3615%** | **1.16770x** |
| FireSim q530 -> q535 cumulative | 1,704,064,223 | **1,316,619,699** | **387,444,524** | **22.7365%** | **1.29427x** |

All 1,000 logits remain bit-exact. The accelerator command buffer, 3,787 launches, 1,050 fences,
and 1,196,945,312 total Gemmini DMA bytes are unchanged, isolating the gain to host dynamic-work
deletion. Spike overpredicted saved hardware cycles by only 1.11216x for this lever.

The sealed package is

```text
out/artifacts/perf-bench/gemmini/resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908
```

Its `verify_bundle.sh` passes and blocks unchanged resubmission. Job 535 used the required queue-only
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill` lifecycle. The measured
ELF, staged ELF, and executed ELF all have SHA-256
`86aea9b8bc6aad86487e652a51629f76dcd472201c85e19077ecb7fac3dbde78`.
Portable evidence is in `q535_affine_im2col_hardware_result_20260908.md`,
`q535_affine_im2col_hardware_receipt.json`, and `phase2_affine_im2col_q535.patch`.

## Native-convolution hardware headroom (hybrid diagnostic only)

FireSim queue job 536 measured **555,991,472 cycles** with all 1,000 int8 logits exact. It used the
required queue-only `firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill`
lifecycle. This is decisive evidence that the native `LOOP_CONV` route has enough hardware headroom
to cross one billion cycles, but it is **not** an end-to-end Merlin compiler score.

The q536 executable combines 53 Merlin-generated native-convolution kernels with a TVM-generated
host runner that still owns the activation arena, call graph, residual blocks, first-layer
padding/max-pool, global average pool, flatten, and dense output. Its apparent 2.368x ratio versus
q535 is therefore not an apples-to-apples compiler speedup. The hybrid stitcher and TVM arena must
not enter the canonical compiler. Only the target-neutral mechanisms—narrow scalar epilogues, i32
bias loads, descriptor guards, and warm/reentrant tests—are eligible for integration through
Merlin's source graph.

The complete ownership audit is `q536_native_loopconv_hybrid_diagnostic_20260908.md`; the portable
machine receipt is `q536_native_loopconv_hybrid_receipt.json`. The reusable mechanisms now pass the
combined four-model and exact local gates, but frozen PT2E ResNet still admits 0/53 exact narrow
convolutions. q535 therefore remains the latest accepted full-Merlin hardware result.

## Invoke this compiler directly

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_phase2_final_combined_exact_residual_global_encoding_20260908
input=/absolute/path/to/model.mlir
output_dir=/absolute/path/to/output
mkdir -p "$output_dir"

PYTHONDONTWRITEBYTECODE=1 python "$artifact/verify.py"

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

The same final snapshot compiles all four complete portfolio graphs without L3 or FireSim,
sequentially with at most 308,264 KiB compiler RSS:

| graph | source ops | accelerator tasks | host tasks | status |
| --- | ---: | ---: | ---: | --- |
| ResNet-50 W8A8 | 1,240 | 54 | 51 | exact local; q535 is latest exact FireSim |
| TinyLLaMA | 718 | 0 | 1 | compiles; all-host gap remains |
| LSTMNetViT | 2,302 | 37 | 38 | compiles |
| SmolVLA denoise | 11,910 | 116 | 117 | compiles after generic `i1` storage sizing |

The non-ResNet targets remain byte-identical to the affine parent; this is a non-regression result,
not a performance win. ResNet selects 15 exact residual formations, reducing tasks 109->105, host
tasks 55->51, ABI pointers 393->389, and physical intermediates by 7,340,032 bytes. It still retains
all 53 proven convolution fallbacks. The same target-neutral mechanisms and refusal rules are used
for every model; none contains a model, layer, or fixed-shape dispatch.

TinyLLaMA can additionally be compiled with the explicit default-off contract
`--dynamic-weight-only-contract symmetric_per_output_channel_roundeven_v1`. That route moves 15
contractions to the mesh and is numerically qualified on the exact frozen witness: 0/2,048
tolerance violations, maximum absolute error 0.018019676, relative L2 0.007236148, cosine
0.999973894, and unchanged top-1 for all 8 tokens. It is not source-f32 bit equivalence, a dataset
accuracy result, hardware execution, or a performance claim.

## What the full-model evidence says to optimize next

q534 still executes 751,827,149 guest instructions and moves 1,142,808,576 read-DMA bytes plus
54,136,736 write-DMA bytes. `loop_matmul_active` is 53,479,788 cycles and reservation-station active
time is 53,892,695 cycles, only 3.51% of the measured interval. The accelerator schedule is not the
dominant remaining cost.

Because q530 and q534 have byte-identical accelerator schedules and DMA volume, their difference
also provides a same-hardware local sensitivity model. Removing 229,709,973 guest instructions
saved 166,648,204 target cycles, or **0.72547 target cycles per removed instruction** over this
interval. The two-point intercept is about 991,986,308 cycles. That intercept is not a physical
roofline or a general prediction—it folds fixed memory stalls and overlap into one number—but it
says something actionable: deleting all scalar instructions would only barely cross one billion if
the rest stayed fixed. Epilogue work deletion and movement/im2col deletion must therefore progress
together; scalar peepholes alone cannot provide a robust sub-billion result.

The fail-closed headroom artifact is
`out/artifacts/perf-bench/gemmini/q534_whole_model_headroom_roofline_20260908`. It invokes the
repository's empirical-roofline API and intentionally receives `refused`: q534 lacks the required
calibration sweeps, sustainable measured bandwidth, empty-run baselines, and complete joint
occupancy partition. The report therefore exposes exact observations—2.661 MAC/cycle end to end,
3.418 MAC/Gemmini-DMA-byte, 3.479% loop-matmul occupancy, and 3.505% reservation-station
occupancy—without inventing a physical roofline or composition rule.
The portable copies are `q534_whole_model_headroom.md` and
`q534_whole_model_headroom_receipt.json`.

The compiler now has the first three target-neutral representations: ordered quantized epilogues,
true second-tensor residuals, and global encoding/lifetime selection. The exact epilogue witness
removes two host tasks plus 1,292 full-width and 969 DMA bytes (13,760->56 Spike proxy cycles); the
two-convolution encoding witness keeps an internal i8 NHWC value across two native convolutions
(96/96 exact, 102 proxy cycles, zero im2col tasks). These prove the machinery, not ResNet gains.

Frozen PT2E ResNet admits 0/53 native narrow convolutions. All layers use per-channel weight scales
and floating-point bias; the current Gemmini `LOOP_CONV` endpoint offers one scalar `CONFIG_ST`
scale and a narrow output. Conv1's naive accumulator-bias/combined-scale fold mismatches
143/802,816 values before pooling and 33/200,704 after pooling. The next macro-first order is:

1. Add a target-neutral endpoint contract with `{narrow_i8, full_i32, accumulator_handle}` and
   explicit completion/lifetime. On Gemmini, implement full-i32/accumulator readout so the exact
   ordered per-channel host/RVV epilogue can follow native convolution without im2col.
2. Admit narrow native output only after arithmetic-equivalence proof. Treat native-aligned W8A8
   as a separate calibration/golden/accuracy/provenance contract, never a silent semantics change.
3. Extend exact residual handling to the four deferred projection branches and close the remaining
   source-1204 terminus only when unique ownership is provable.
4. Continue explicit, default-off dynamic-activation contracts for weight-only transformer
   contractions; never claim source-f32 bit equivalence.
5. Then tune tiling, queue depth, prefetch, and overlap. Blanket fence removal measured only 0.6277%
   on q530 and remains behind boundary/movement deletion.

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
