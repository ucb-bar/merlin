# Public Phase 2 shape supplement

This example adds ten public, user-selected shape families to a **new copy** of
the selected workload roster. It preserves the original iteration roster,
full-model objective and frozen Phase 1 release. It stages ordinary PyTorch
sources; it does not execute captures, compilers, simulators, search or FPGA jobs.

The shapes come from the [Autocomp paper](https://arxiv.org/abs/2505.18574)
and its author's harness metadata at revision
`a56ce8154c6992648348517489ff5db3d8267798`.
[upstream-source-pins.json](upstream-source-pins.json) records the exact URLs and
file digests. No baseline kernels, instructions, schedules or constants are
copied or imported. These source identities explain geometry, not performance.

The GEMM labels mean **M × N × K**, with `A[M,K]`, `B[K,N]` and `C[M,N]`.
They are the `harnesses/exo/test0..5.c` cases, not the distinct reuse suite or
older Exo case numbering. In particular, `784×1024×256` has reduction 256.

| Label | M | N | K | Origin |
| --- | ---: | ---: | ---: | --- |
| 512×512×512 | 512 | 512 | 512 | Paper square case |
| 12544×256×64 | 12544 | 256 | 64 | Paper |
| 12544×64×256 | 12544 | 64 | 256 | Paper |
| 3136×512×128 | 3136 | 512 | 128 | Paper |
| 3136×128×512 | 3136 | 128 | 512 | Paper |
| 784×1024×256 | 784 | 1024 | 256 | Paper |
| 12×512×512 | 12 | 512 | 512 | Additional user-requested case |

The convolution labels mean batch × kernel side × **output** side × input
channels × output channels. Each is stride 1, dilation 1, groups 1, valid
padding 0. The inputs are NHWC, weights HWIO, bias one i32 value per output
channel, and results NHWC.

| Label | Input NHWC | Weight HWIO | Output NHWC |
| --- | --- | --- | --- |
| 4×3×56×64×64 | 4×58×58×64 | 3×3×64×64 | 4×56×56×64 |
| 4×3×28×128×128 | 4×30×30×128 | 3×3×128×128 | 4×28×28×128 |
| 4×3×14×256×256 | 4×16×16×256 | 3×3×256×256 | 4×14×14×256 |

## Explicit arithmetic and inputs

The default `native_i8_scale1` computes signed-i8 products, exact i32
accumulation, convolution bias in that accumulator, then
`min(127, max(-128, accumulator))` to signed i8. There is no activation, GEMM
bias, transpose, implicit shift, floating gain or intermediate rounding. Unit
scale on an integral accumulator needs no tie rule. The optional `raw_i32`
readout returns that same accumulator as a semantic control. It is not silently
substituted for the default i8 output.

GEMM uses the stock `torch._int_mm`; convolution uses stock integer
`torch.nn.functional.conv2d` with explicit i32 operand casts and layout views.
These are mathematical input programs, not target kernels or lowering hints.
Native integral capture and all ordinary source/ABI checks remain required;
unsupported frontend or target handling must refuse rather than change formats.

Inputs use separate deterministic `torch.Generator` seeds by operand role,
sampling signed values uniformly from `[-3,3]`; convolution biases use
`[-16,16]`. This avoids repeating rows caused by linear sequences modulo 256
at the selected power-of-two dimensions. It is one reproducible workload input,
not coverage of the entire signed-i8 domain. Preparation separately checks a
worst-case full signed-i8 accumulator bound. Small tests independently evaluate
every result with NumPy integer arithmetic and test signed saturation endpoints.
They do not evaluate these full-size cases or establish target correctness.

Both readouts retain the same predetermined `source_group`, derived from the
base operation, complete geometry, bias contract, input policy and seed. Keep
all numeric/compiler variants in that group for any held-out split. A new ELF
or quantization label must not make a source independent of its other variants.

## Prepare through the ordinary roster

From the selected source checkout, choose a fresh task-owned output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B examples/gemmini/phase2/benchmarks/prepare.py \
  --descriptor examples/gemmini/target/descriptor.yaml \
  --base-roster examples/gemmini/target/phase0-workloads.yaml \
  --software-spec examples/gemmini/target/software-spec.yaml \
  --output "$PREP"
```

Use `--readout raw_i32` only for an explicitly separate control preparation.
Preparation renders the ten independent loaders, appends them to the copied
`performance_applications`, and calls the existing
`merlin_experiments.phase0.workload_roster.stage` with the copied descriptor.
Original application loaders/profiles are staged unchanged. The generated
`recipe.yaml` selects the ordinary versioned `public_programs` native-integer
path, with no default requantization shift or fake independent-V3 selection.
`preparation.json` pins the profile, implementation, upstream metadata,
software spec, descriptor, roster, loaders, complete logical ABI and work bounds.

`prepare.py` also exposes a thin `select_capture` Python function. Supply the
independently selected SHA256 of `preparation.json`, a benchmark ID, explicit
`m2m_root`, `worker`, `venv`, `schemas_root`, fresh `run_dir` and `output_dir`,
and a bounded `execution_timeout_seconds`. It reopens the preparation/staged
source pins and calls the normal
`merlin_experiments.phase0.capture_selection.select` with `dtype="int8"`,
no numerical recipe, and `worker_options={"source_mode":"native_integer_v1"}`.
This only preselects a capture; it does not issue one. Use the existing issue,
independent replay and attestation owners afterward. Do not treat diagnostic
capture reports or saved flags as source authority.

Scored intake is **pending**. The current `public_programs`-only requirement
route still requires an operator-private exclusion policy. The public benchmark
membership must not be silently pruned or selected from private model shapes.
That policy choice and all normal source/capture, HW/SW, writer and performance
generation gates need explicit resolution before the existing Phase 2 corpus
reader can admit these members. This preparation emits no `_perf` manifest,
measurement or certificate and modifies no admission code. Keep generated
artifacts in a fresh output root; never rewrite an existing Phase 1 corpus.

## Comparison objective remains unmeasured

The goal is lower paired complete-kernel cycles than **Exo optimized** and
**Autocomp**, reported per case and as a geometric mean. The intended window
includes internal packing, configuration and final completion; it excludes
external input initialization, reference checking and UART output. Matching
hardware, runtime, inputs, arithmetic/readout, allowed instructions and timing
boundaries is mandatory. The historical GEMM readout contract is not fully
resolved, so geometry alone does not establish mathematical comparability.
No baseline cycle values are inferred from chart pixels and no win is claimed.

Retain the primary Phase 2 campaign's budgets, full-model objective and
Spike-based iteration policy. FPGA remains for full-model final validation;
this metadata authorizes no paid capsule job. Figure 9 forward/backward is
deferred: the supplied labels lack complete shapes/source contracts and the
paper uses a separate 4×4 FP32 accelerator, unlike the selected 16×16 INT8
target. No equivalent INT8 case is invented.

## Complete attention supplement

`attention-profile.yaml` adds a separate complete-attention experiment source:
32 query heads, 4 KV heads, head/value dimension 64, 32-token causal prefill,
then one decode token with growing KV. A second member covers 6/2 heads,
17/19 head/value dimensions and uneven sequence lengths. Both reuse the
existing public `development_attention` source factory. Projection and RoPE
are outside this explicit Q/K/V-to-context boundary.

Prepare either member in the ordinary capture environment:

```sh
python -B examples/gemmini/phase2/benchmarks/prepare_attention.py \
  --member gqa_prefill32_decode1 --output "$ATTENTION_SOURCE"
```

The result retains typed source, inputs, captured and independent references,
portable capture evidence, ABI and work counts in `attention-member.json`.
This is a diagnostic source, with the original pending numerical/target
qualification recorded. It does not issue a scored corpus admission or start
an automatic authoring campaign. The profile records the requested 3-hour
limit, 30% roofline target and lowest-attainment priority for its campaign
consumer. Unknown attainment retains a member for measurement; it cannot
establish convergence.

The source computes signed i8/i32 QK, FP32 scaling/masking/softmax and FP32 PV.
Only final contexts and carried KV are live-outs. No probability quantization,
integer-softmax replacement or new floating tolerance is introduced. Candidate
compiler iterations compare all output bytes with the same CPU source
baseline, check carried integer state independently, and report independent
FP32-reference differences separately.

The first complete-layer Spike experiment passed 84,480 output and state values,
repeated invocations and output guards. QK offload through shared Merlin host
composition reduced instructions from 43.43M to 25.76M; the existing
generalize-before-fuse pass order reduced them further to 22.04M. That is a
1.97× instruction reduction against the CPU source baseline. The tail member
also passed 1,332 values, with instructions falling from 149,714 to 116,100.
The independent FP32 difference for the larger member was approximately
9.54e-7; the compiler candidates matched the CPU source output bytes exactly.

The frozen native timing model produces complete attention instruction counts
but lacks the executed FP32 instruction prices/resource mappings. Whole-layer
cycle predictions and complete roofline attainment therefore remain unknown.
The integer QK floor cannot stand in for a layer that executes FP32 PV on the
CPU. No latency speedup, 30% attainment, full-model numerical qualification or
new FireSim result is inferred from the instruction reductions.

Task-local results are under `out/artifacts/verification/` in
`merlin-shared-attention-v1`, `merlin-shared-attention-order-v1` and
`merlin-shared-attention-tail-source-v1`. They retain actual compiled LLVM,
backend kernels, bridges, ELFs, complete readback, guard comparisons, commands
and unsuccessful timing observations.
