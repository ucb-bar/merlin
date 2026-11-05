# Qwen compilation and Atlas compiler findings

This is the historical manually authored microbenchmark report. The newer
[pretrained-model bringup](qwen-model-bringup.md) captures the actual model and
derives its regions and launch inventory from the exported graph.

Measured locally on 2026-11-04, starting from `feat/pip-package` at
`a92d173`. This report accompanies the package, capture, and benchmark changes
in this checkout. Measurements below are **NPU simulator cycles**, not RTL,
FPGA, silicon, GPU timings, or full-model inference latency.

## Model and workload boundaries

The model is **deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B**, the Qwen2 model used
in [Agustin's Radiance PR #1](https://github.com/ucb-bar/radiance-kernels/pull/1),
pinned at `b60401712885ad0bbbc5d8946726f39f8d43ecb6`. Its
[configuration](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json)
has hidden width 1536, FFN width 8960, 28 layers, 12 query heads, 2 KV heads,
128 channels per head, RMSNorm epsilon 1e-6, and RoPE theta 10000.
The head grouping is 6:1. The Radiance PR also benchmarks a generic d=64,
8-way GQA kernel; that is not this Qwen model's head configuration.

The workload definitions in [kernels.py](../kernels.py) are plain
PyTorch. The PR supplies model selection and dimensions; the subgraphs were
written independently, with 32×32 tiles chosen for Atlas. Capture starts from
these modules, without loading a pretrained checkpoint or importing Radiance
kernel implementations. Capture uses torch.export and torch-mlir, followed by Merlin's source
precision lowering, instruction selection, allocation, and the Atlas virtual
and machine pipelines. Simulator inputs and float32 reference outputs are
produced separately from compilation.

| Workload | What is inside the measured program | Scope |
| --- | --- | --- |
| Attention | Four Q/K contractions and their sum, 1/sqrt(128) scaling, stable softmax, four probability/V contractions | One query head; 32 query rows × 32 cache entries; full 128-channel head; four output tiles |
| SwiGLU | Gate and up projections, sigmoid/SiLU, elementwise gating, down projection | A 32-channel FFN partition; three matmuls in one program |
| Projection partitions | Ordered K=128, 256 and 512 reductions | 32×32 output tiles; 4, 8 and 16 K tiles respectively |
| Projection | One output window of a real 1536-wide Q/O projection, with ordered K continuation | Full K=1536; 48 K tiles and 96 input tile bindings |
| RMSNorm + QKV | Full-width normalization and biased Q/K/V projections | PyTorch and Linalg capture; target lowering refused |
| Full MLP residual | RMSNorm, 1536→8960 gate/up projections, SiLU gating, 8960→1536 down projection, residual | PyTorch and Linalg capture; target lowering refused |

The attention program is an **unmasked cache window** with Q/K already rotated.
It does not include RoPE, causal masking, replication over all 12 query heads,
or an online-softmax merge over a longer context. The FFN partition does not
represent an entire 8960-wide layer: K reductions must complete before SiLU,
and down-projection partitions must be accumulated. These boundaries are
explicit in the Python modules.

Inputs and parameters are seeded synthetic float32 tensors. These runs exercise
architecture and scheduling, and do **not** establish pretrained-model quality,
perplexity, token throughput, or an end-to-end deployment of all 28 layers.

## Measured results

Follow-up: [resource roofline analysis](qwen-roofline.md) runs all five compiled
kernels and all three pipelines through the newer `npu_model` roofline tool,
including a faster-memory sensitivity test and per-resource bounds.

Both compared pipelines start from the same Merlin-selected virtual program.
`unscheduled` uses `--atlas-compile-unscheduled`; `scheduled` uses
`--atlas-compile`, which also exposes and aggregates boundary DMA. Consequently,
the improvement includes DMA changes as well as instruction reordering.

| Workload | Unscheduled cycles | Scheduled cycles | Speedup | Cycle reduction | Program words, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention, d=128 | 18,866 | 16,127 | 1.170× | 14.52% | 235 → 277 |
| SwiGLU partition | 8,833 | 7,691 | 1.148× | 12.93% | 111 → 126 |
| Projection K=128 | 8,324 | 7,717 | 1.079× | 7.29% | 90 → 113 |
| Projection K=256 | 15,060 | 13,727 | 1.097× | 8.85% | 161 → 205 |
| Projection K=512 | 28,532 | 25,745 | 1.108× | 9.77% | 303 → 389 |

| Scheduled workload | DMA busy | MXU0 busy | VPU busy | DMA/MXU overlap cycles | Maximum absolute error vs PyTorch f32 | Relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention | 99.94% | 6.04% | 4.55% | 974 | 0.002415 | 0.761–0.816% across four outputs |
| SwiGLU | 97.20% | 4.69% | 6.88% | 263 | 0.007330 | 1.711% |
| Projection K=128 | 96.88% | 6.66% | 0% | 417 | 0.053543 | 1.312% |
| Projection K=256 | 98.24% | 7.47% | 0% | 929 | 0.131429 | 1.680% |
| Projection K=512 | 99.06% | 7.96% | 0% | 1,953 | 0.675877 | 9.223% |

Busy percentages are simulator execution-unit busy cycles divided by total
cycles. They are **not** matrix arithmetic utilization versus peak FLOPs;
busy time may include engine waiting. The first run's attention DMA/MXU overlap
was 57 cycles, and the SwiGLU overlap was 55 cycles.

Raw outputs, traces and generated programs remain local under
`out/qwen-final/<workload>`. The tables in this report summarize their
performance and accuracy; generated datasets and images are not tracked.

### Isolating boundary DMA from scheduling

A third pipeline first exposes boundary DMA and then runs the unscheduled
lowering. All three variants passed the same output and memory checks.

| Workload | Original unscheduled | Boundary DMA only | Fully scheduled | Scheduling gain over boundary-only |
| --- | ---: | ---: | ---: | ---: |
| Attention | 18,866 | 18,650 | 16,127 | 1.156× (13.53% fewer cycles) |
| SwiGLU | 8,833 | 8,676 | 7,691 | 1.128× (11.35% fewer cycles) |
| Projection K=128 | 8,324 | 8,138 | 7,717 | 1.055× (5.17% fewer cycles) |
| Projection K=256 | 15,060 | 14,404 | 13,727 | 1.049× (4.70% fewer cycles) |
| Projection K=512 | 28,532 | 26,934 | 25,745 | 1.046× (4.41% fewer cycles) |

Boundary changes alone account for 216 attention cycles and 157 SwiGLU cycles;
the remaining differences are 2,523 and 985 cycles respectively. These are
controlled whole-pipeline comparisons, not an additive per-instruction cost model.

### Signed-input panel

A second panel uses independent random signs and magnitudes in
[0.03125,0.15625). It also passes all 15 emitted-program and memory checks.
Scheduled cycle counts are unchanged, while source-model error changes:

| Workload | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| Attention | 0.001601 | 2.676–4.634% across outputs |
| SwiGLU | 0.004498 | **100%; every output is zero** |
| Projection K=128 | 0.017194 | 4.176% |
| Projection K=256 | 0.028932 | 4.448% |
| Projection K=512 | 0.060203 | 4.767% |

The source SwiGLU gated intermediate has maximum absolute value 0.010716,
below the unit-scale FP8 minimum normal value 0.015625. The selected pack
flushes the intermediate, and the numerical evaluator confirms all-zero
output. This is an accuracy failure of the selected precision policy even
though the machine implementation matches that policy exactly.

An initial uniform [-0.125,0.125) input attempt was explicitly refused because
flushing tiny negative values produced negative zero, outside the declared
input-code domain. The bounded signed panel avoids this input-boundary issue;
it does not establish arbitrary signed-domain support.

Signed data and compilation records remain local in
`out/qwen-signed-normal`.
K=512 was retried after an in-progress local compiler relink temporarily made
the executable unavailable. Its unscheduled/boundary/scheduled counts were
28,498/26,957/25,745; the retry selected a different unscheduled placement.
These are independent compilations, so differences between the two panels'
unscheduled programs should not be interpreted as input-dependent timing.

**Recommendation:** make activation range and zero-rate measurements part of
precision selection. Add per-intermediate scaling with compensating operand
or output scales, preserve the scale contract across fusion boundaries, and
keep a higher precision route when FP8 cannot satisfy the error budget. Merely
changing a pack scale without compensating the computation changes semantics.

### Correctness and measurement checks

- Merlin's published artifact manifests were verified.
- Emitted words were decoded against the selected RTL inventory, converted to
  simulator instruction objects, and required to re-encode to identical words.
- All 30 measured programs across both input panels halted with **zero simulator runtime errors**.
- All output bytes matched the Atlas virtual evaluator bit-for-bit.
- The complete initialized DRAM region was compared after execution, allowing
  changes only to declared output ranges; inputs and guard bytes survived.
- Float32 errors above compare with the original PyTorch computation. They are
  separate from the bit-exact check of the selected FP8/BF16 arithmetic.
- Performance includes the simulated program from reset through completion.
  It does not include Python compilation, capture, host ELF execution, or host
  launch overhead. It uses the pinned simulator's default memory timing.

The model pin is `337e9f249f938a1dff007ddf8a763e5cd4b3aa2b`, with RTL numerical
mode. The compiler RTL inventory is
`0079c0541111197741a231c002e3843fa6f545b2`; Merlin is
`85db5fb4e6eeb0ffe24c5a9938da8da8f255701d`. These Atlas measurements cannot be
compared directly to the Radiance PR's different architecture, precision,
measurement window and memory model.

## Compiler findings

### DMA dominates these fused windows

The scheduled attention program leaves almost no modeled DMA idle time, but
MXU0 is busy only 6.04%. Adding arithmetic scheduling sophistication alone is
unlikely to produce a large improvement for this particular boundary: the
compiler must reduce traffic or reuse already resident data. The workload's
small matrix windows also limit arithmetic intensity.

**Recommendation:** prioritize resident-tile lifetimes across adjacent kernels,
coalesced transfers, shared K/V reuse across Qwen's six query heads per KV head,
and retaining intermediate tensors across output-channel tiles. Measure bytes
moved, command overhead, overlap, and arithmetic work separately. Re-run with
larger token windows before generalizing the bottleneck to every Qwen stage.

### Longer reductions increase numerical error

Relative L2 error against the original f32 projection grows from 1.31% at
K=128 to 1.68% at K=256 and 9.22% at K=512 on the positive panel. The emitted
program remains bit-exact against the selected Atlas numerical model, so
that check alone cannot establish acceptable model accuracy. These results
do not isolate input FP8 quantization from accumulation rounding.

**Recommendation:** sweep reduction length, input scale and sign distribution
with explicit accuracy budgets. Attribute error to input conversion, MXU
accumulation and output conversion separately; evaluate chunked or higher
precision accumulation before treating long projections as production-ready.

### Long reductions currently stress Merlin allocation

The final campaign compiled K=128, 256 and 512 in 0.27, 1.06 and 4.37
seconds, respectively. These times cover precision lowering, native search,
emission, verification and input preparation; they exclude capture and shared
Rust snapshot construction. Attention took 1.55 seconds and SwiGLU 0.49 seconds.
The repeatable K=1536 attempt stopped after 60.62 seconds at its 60-second
search budget.

A source-captured K=1536 projection was transformed into 48 ordered K tiles
with 96 input bindings. With 256 candidate nodes, 512 candidates, 64 orders per
candidate, a 5-second solver limit and a 300-second search deadline, native
compilation failed with:

```text
search_timeout: native search deadline expired during allocation
```

A profile of a single candidate with a 30-second solver budget spent 30.0
seconds inside Z3 and 5.4 seconds in candidate extraction. An experimental
incremental SAT formulation also exhausted its first 30-second feasibility
check; it was discarded. This does not support blaming lexicographic
optimization alone.

This is a **compiler search/scalability failure**, not evidence of slow device
execution or proof that the hardware cannot execute the projection. No runtime
performance number is assigned to the failed compilation.

**Recommendation:** recognize long ordered contraction chains and schedule them
with a streaming template or loop, keeping a bounded live set. Avoid treating
all source tile bindings as simultaneously interesting allocation choices.
Cache compiled tile/loop schedules by shape and precision contract. Keep the
new K=128, 256, 512 and 1536 compile-time sweep as a regression benchmark, with
explicit time and memory budgets.

### Capture normalization was a practical frontend gap

Current torch-mlir emitted a dead argmax index result for softmax maxima,
keep-dimension reductions, and f64 literals narrowed inside elementwise regions.
The existing Merlin bridge expected simpler reductions and explicit tensor
broadcasts. The new normalization pass structurally rewrites these forms,
preserves live argmax results, and retains the raw capture for inspection.
Sigmoid capture is also normalized into Merlin's recognized expression.

**Recommendation:** maintain capture tests against a pinned torch/torch-mlir
pair, plus a periodic newer-version job. Test semantic equivalence and live
auxiliary outputs; do not rely on matching incidental SSA names or textual IR.

### Nonlinear precision needs explicit limits

The SwiGLU experiment adds a sigmoid lowering using BF16 negation, exp,
addition and reciprocal. The selected reciprocal remains qualified only for
positive BF16 denominators in [1,32]. The recorded intermediate-domain
obligations are not runtime guards. The bounded signed panel is an additional
diagnostic, and does not qualify arbitrary real-model activations. In
particular, it exposes the SwiGLU underflow failure described above.

**Recommendation:** collect activation distributions from pretrained Qwen
before choosing production precision. Qualify or guard the nonlinear domains,
consider a stable sign-split sigmoid implementation, and establish model-quality
thresholds. Complete RMSNorm/rsqrt, RoPE and causal-mask lowering without silently
changing Qwen semantics or replacing SiLU with ReLU.

## Package and workflow verification

The original branch had an `atlasc build` driver but an outdated README and no
package CI. Source-tree tests passed while a wheel built against conda LLVM
with the system C++ compiler crashed during real compilation. Printing tool
help alone did not detect the problem. A wheel built with the matching conda
compiler passed the installed-wheel tests.

Changes include optional PyTorch capture dependencies; `atlasc demo`; an
executable `python -m atlas_mlir` entry point; meaningful `doctor --capture`
checks; correct forwarding of captured example inputs; a dataclass-compatible
model loader; and a proper successful `atlas-emit --help` exit status.
The build now splits LLVM compiler definitions correctly and runs a real
compile/emission smoke check before publishing a wheel.

Packaging, usability improvements and CI live separately on `feat/pip-package`.
The `feat/qwen-kernels` branch contains no CI files or workflow changes.
The Package workflow builds an sdist and its wheel, tests the compiler,
then tests the installed wheel on separate Linux runners with Python 3.10 and
3.12 with xDSL 0.65/0.68. A separate Python 3.14 job runs the numerical
evaluator and emitted-model checks. Build artifacts supply downloadable wheels;
there is no automatic PyPI release. Qwen campaigns are run manually using the
reproduction commands below. The workflow has been authored and locally exercised in pieces;
a hosted GitHub Actions run and branch-protection enforcement are separate
from local verification.

Local validation:

- Installed-wheel base/demo, `.py` capture, `.pt2` capture and RISC-V host ELF smoke tests passed.
- 10 package/Qwen capture tests passed; 24 package/capture/importer tests and
  21 subtests also passed with xDSL 0.68.
- Full suite in the pinned numerical-model environment: 565 passed, 309 skipped,
  6,522 subtests passed (the separate `atlasc` module was excluded and tested in
  its capture environment).
- Graph-precision suite: 8 tests run, one environment-gated test skipped.
- Emitted spill-model execution: 2 tests and 21 subtests passed.
- CMake/CTest: all 149 test entries passed after correcting the numerical-model
  installation to retain its source-side arithmetic corpus. The initial run
  had 148 passes and one module failure from the absent fixture; its rerun
  passed. CI now installs this pinned model in editable mode for that reason.
- Python 3.10 installed-wheel capture and host-ELF smoke passed as well.
- The final wheel was built from its sdist and tested outside the checkout:
  10 capture/API tests each on Python 3.10 and 3.12, a base-only environment
  without PyTorch, and installed Merlin-plugin compilation using only the
  packaged support files and automatic tool lookup. Artifacts are in
  `out/release-dist`.
- The documented `--setup` followed by a default-environment run completed;
  all five positive-panel performance JSON files were exactly reproduced.
  Local records in `out/qwen-reproduced` include tool hashes, package
  versions and compilation times.

Skipped tests are not evidence of execution. FPGA, integrated SoC, and the
optional external RTL tests were not run in these results.

## Reproduction and next checks

The source-only workload entry is `examples/atlas/model_bringup/qwen/run.py`. It prepares isolated
capture, Merlin and simulator environments with `--setup`, then runs capture,
compilation and measurement without hand-authoring target IR. Its campaign
output records per-workload compilation status, precision plans, inputs,
virtual/machine IR, emitted words and performance JSON.

The remaining compiler work is full-width allocation, broader source-operator
coverage and pretrained-activation correctness panels. These results should be treated as a
repeatable compiler baseline, not a complete Qwen deployment.
