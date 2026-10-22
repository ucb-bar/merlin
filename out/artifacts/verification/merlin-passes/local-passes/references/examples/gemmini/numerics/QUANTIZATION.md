# Historical r53b Gemmini quantization guide

**Scope:** this guide records the r53b captures and dated 2026-10-16 recovery
work. Its BF16 SmolVLA, per-tensor weights, 128-token prefill, floating attention
and two-multiply epilogue describe that historical selection. For the selected
successor's grouped-16 scales, INT8 attention, hardware-order integer epilogue,
exact shapes and metric definitions, use the
[shared experiment protocol](../protocol/README.md) and pinned recipe JSONs.
Historical results below are preserved; they do not certify the current run.

This guide explains how Merlin turns ResNet50, SmolVLA and TinyLlama into int8 programs for Gemmini,
and how precision is handled from the PyTorch model down to the accelerator's readout. It is written
for engineers outside this project, in particular anyone building an independent Gemmini
implementation that must run **the same int8 models with the same numerics** so that results can be
compared.

Sections 1–7 describe code and capture records. Each claim names the file or record a reader can
check. Section 8 is a dated status report (2026-10-16) and will change. Section 9 lists what is not
yet settled. The judging contract is [`docs/reference/model_check.md`](../../../docs/reference/model_check.md).
The evaluation and calibration inputs are described in [`README.md`](README.md).

"r53b" names the historical development captures. They are kept in a private, sealed capture store,
and every fact quoted from them below is structural (counts, settings, hashes). No tensor data is quoted.

## 1. Pipeline

Each workload is a PyTorch model with a loader in the model2mlir repository
(`workloads/<model>/loader.py`; the r53b captures used model2mlir commit `905d2c918f83`). Phase 0
freezes a quantization **recipe**, a `quant_recipe_v1` JSON document derived from what the target's
readout can hold (`src/merlin/targetgen/quant_recipe.py`, `derive`). The capture worker
(`src/merlin/targetgen/_m2m_capture_worker.py`) exports every program with `torch.export`.
`src/merlin/targetgen/_recipe_quantizer.py` (`apply_recipe`, `build_quantizer`) turns the recipe into
a TorchAO PT2E quantizer, calibrates it and converts. Next, model2mlir rewrites every contraction whose
two operands are **float32** dequantizes into int8 × int8 → int32 followed by scaling
(`m2m/capture/pt2e_integerize.py`, `integerize_pt2e`). The rest of the graph is lowered to linalg MLIR,
where the remaining quantization points appear as `quant_ext.quantize_per_tensor` and
`quant_ext.dequantize_per_tensor`. Merlin prepares this captured program with
`src/merlin/runtime/backends/zephyr_model.py` (`_prepare_model_mlir` with `int8_compute=True` and
`respect_captured_quantization_scope=True`). Only the pass `contraction_int8` runs, and it runs only on
contractions with an input that traces to a captured dequantize (`src/merlin/llvmlower/quant_scope.py`,
`captured_quantized_contraction`). That pass converts the remaining (bf16) quantize/dequantize
contractions to int8 × int8 → int32 plus a requantization (`src/merlin/llvmlower/passes_quant_int.py`,
`lower_contraction_int8`). `lower_quant_ext` (`src/merlin/llvmlower/passes_xdsl.py`) then spells every
leftover quantize and dequantize as explicit float arithmetic. The result, taken before any device
grouping, is the **INT8 reference**. Device grouping (`src/merlin/compile/complete_group_routing.py`,
`src/merlin/llvmlower/device_offload.py`, `rewrite_groups_to_device`) cuts that program into Gemmini
kernels and host code.

```
PyTorch model + loader (model2mlir workloads/<model>/loader.py)
  | recipe: quant_recipe_v1, frozen by Phase 0            (targetgen/quant_recipe.py)
  v
torch.export -> PT2E prepare, calibrate, convert            (targetgen/_recipe_quantizer.py)
  | f32 Q/DQ contractions -> int8 x int8 -> int32, * s_x * s_w   (m2m pt2e_integerize.py)
  v
captured model.mlir: linalg + quant_ext.{quantize,dequantize}_per_tensor, weights.safetensors
  | _prepare_model_mlir(int8_compute, captured scope): contraction_int8 on the
  | remaining bf16 Q/DQ contractions, then lower_quant_ext
  v
INT8 reference = prepared program, ungrouped; evaluated by runtime/linalg_numpy.py
  | device grouping (compile/complete_group_routing.py)
  v
Gemmini: int8 x int8 -> int32 kernels (+ an acc_scale readout for ResNet50's average pool)
host CPU: requantization epilogues, every float op, everything not grouped
```

## 2. Number formats and arithmetic

| Quantity | Format | Where it is set |
|---|---|---|
| Weights | int8, symmetric, zero point 0, range [-127, 127] (the restricted range, so a negated weight is representable) | recipe `weight`; `quant_recipe._integer_range` |
| Contraction inputs (activations) | int8, symmetric, zero point 0, range [-128, 127] | recipe `activation`; clamp from the readout facts |
| Accumulation | int32, exact | int8 × int8 products summed in i32; capture checks K·128² < 2³¹ (`accumulator_bound_checked`) |
| Scales | float32 scalars, one per tensor (r53b; SmolVLA's patch embedding is the exception, §4) | f32 constants in the captured IR |
| Everything else | float32 (ResNet50, TinyLlama) or bf16 (SmolVLA, with some f32 layers) | the source model |

**Scale computation** happens once, at capture time, in TorchAO's observers. The torch and torchao
builds are recorded in each `capture_receipt.json`: torch 2.10.0, with torchao 0.16.0 for ResNet50
and TinyLlama and torchao 0.15.0 for SmolVLA. The symmetric formula is
`s = max(absmax / ((qmax − qmin) / 2), eps)`, with `eps = 2⁻¹²` (recipe
`framework_capture_policy.observer_epsilon`). Weights use `MinMaxObserver` (absmax / 127).
Activations use `HistogramObserver`, which picks a clipping range by an approximate L2 search over a
2048-bin histogram and then divides by 127.5. An independent implementation should not recompute
scales; it should read them from the capture (§7).

**Quantize** (`lower_quant_ext`) computes `q = clamp(roundeven(f32(x) · f32(1/s)) + zp, qmin, qmax)`.
The reciprocal is formed once in f32, as TorchAO does, so the result is `x · (1/s)`, not `x / s`.
A bf16 `x` is widened to f32 first, and ties round to even. **Dequantize** computes `(f32(q) − zp) · s`
and rounds once to the result type, which is bf16 for SmolVLA's bf16 quantize/dequantize pairs.

**Contraction epilogue in the INT8 reference.** Both routes produce
`y = (f32(acc) · s₀) · s₁`: two f32 multiplications in operand order. For every linear and
convolution, that order is the activation scale first and then the weight scale (`integerize_pt2e`
applies "activation scale before weight scale"; `lower_contraction_int8` multiplies the scales of
input 0 and then input 1). In f32, `(a·b)·c` is not `a·(b·c)`, so a single product scale is a
different function. After the epilogue the captured graph continues, and the prepared program states
the exact steps:

- ResNet50 conv/FC: `+ bias` (f32; BatchNorm is folded into the conv by PT2E), then ReLU where
  present, then the next layer's quantize.
- SmolVLA bf16 linears: round to bf16, then `+ bias` as a bf16 add (SigLIP layers), then the following
  bf16 ops.
- TinyLlama linears (no bias) and SmolVLA's f32 projections: the f32 result, plus an f32 bias where
  the layer has one.

**Gemmini readout.** Sources: [`target/contracts/target_contract.yaml`](../target/contracts/target_contract.yaml) (`scaling`,
`epilogue_capability`, `stage_routes`, `scalar_abi`, `operand_sum`, which cite `gemmini_params.h`),
[`merlin/contract/command_buffer_abi.yaml`](../../../merlin/contract/command_buffer_abi.yaml), and the
RTL facts of the Phase 1 release payload, whose readout section matches.

- An int8 × int8 product accumulates into an int32 accumulator.
- The **i32 readout** writes the raw accumulator.
- The **i8 readout** applies `acc_scale`, `relu` and `maxpool`.
- `acc_scale` is **one f32 scalar per readout configuration**: a 32-bit field of the mvout config
  command (`ConfigMvoutRs2.acc_scale`). It is applied as `clamp_i8(round_half_even(f32(acc) · scale))`
  (`src/merlin/runtime/tensor.py`, `Tensor.requant_acc_scale`).
- There is no zero point, and the scaler has no bias operand. A bias can enter only as an int32 seed
  that is loaded into the accumulator before the contraction.
- Load-time scaling, which could implement an integer residual add, rounds each operand half-to-even
  before the sum. The result can be one output step away from the reference's single rounding
  (`Tensor.residual_add_scaled`).

As a result, the following run on the **host CPU**:

- per-output-channel and per-token scales;
- the two-multiply epilogue, unless Merlin proves that one integer bias plus one `acc_scale` reproduce
  it over the whole accumulator range (`src/merlin/xdsl_dialects/lowering/group_exact_float.py`,
  `prove`); when the proof fails, the kernel returns int32 and the float epilogue stays on the host;
- float biases;
- every nonlinearity, softmax, normalization, RoPE and embedding;
- residual adds;
- ResNet50's max pool, whose input is float.

In the r53b bundle **every contraction kernel returns int32**, except ResNet50's average pool, which
reads out int8 through `acc_scale`.

## 3. The recipe used by the r53b captures

All r53b captures use one recipe. Its content digest is `recipe_sha256`
`6497b3dde13e91c1787ec6bce07674866861da6389bccc7f3388cda7e386df51` (`quant_recipe.digest`, which
excludes `why`). It is recorded in each `capture-selection.json`, `meta.json` and
`session-receipt.json`. The settings are:

- `schema` `quant_recipe_v1`; `target` gemmini; `unit` systolic_mesh; `accumulator_dtype` int32.
- `families`: `contraction`, `operand_sum`, `window_mean`.
- `weight`: int8, `tensor`, symmetric, [-127, 127], static, observer `minmax`.
- `activation`: int8, `tensor`, symmetric, [-128, 127], static, observer `histogram`.
- `bias_domain` null: no readout applies a bias, so a bias stays a host add.
- `software_numerical_engine` `integer_reference`; `software_admission` holds the reviewed software
  spec's operation signatures.

A family the recipe lists is quantized only where the graph allows it. Contractions are annotated
only when their second operand is a stored float weight (`build_quantizer.eligible`). A sum or mean is
kept only when every reader of its result is a quantized contraction (`build_quantizer.closable`);
otherwise it stays float. A contraction of two run-time tensors (attention) is not covered by
`contraction` (see §5).

## 4. Per workload

All counts below come from each program's `meta.json` (`quantization_stats`,
`integerization_receipt`) and the development bundle's census.

**ResNet50** (program `model`)

| Item | Value |
|---|---|
| Source model | torchvision ResNet50 `IMAGENET1K_V2`, float32 |
| Quantized | 53 conv2d and 1 linear (FC), all integerized at capture. 16 residual adds have both inputs quantized (`operand_sum`), and the adaptive average pool's input is quantized (`window_mean`) |
| Weights | per-tensor, static, MinMax, after BatchNorm folding (each conv carries an f32 bias) |
| Activations | per-tensor, static, Histogram |
| Calibration (r53b) | the first 100 of the loader's 256-image session stream (`calibration_source: measured_session_stream`). The r53b capture ran with `M2M_RESNET_RANDOM=1`, so these are synthetic standard-normal images (seed 20260830), not ImageNet images |
| Float, on host | stem max pool, ReLU, bias adds, residual-add arithmetic (dequantize, add, ReLU, quantize in f32), all contraction epilogues, logits |
| Device kernels | 53 conv and 1 FC (int8 → int32), plus the average pool as a window-mean matmul `x[2048×49] @ ones[49×1]` with `acc_scale = s_in / (49 · s_out)` fused into the readout (int8 out). Census: 54/54 contractions on the device. Residual adds are not routed |

**SmolVLA** (programs `prefix_encode`, `flow_denoise` ×10, `action_decode`)

| Item | Value |
|---|---|
| Source model | `lerobot/smolvla_base`@`c83c3163`, VLM `SmolVLM2-500M-Video-Instruct`@`7b375e1b`. **bf16**, except `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in/out` and the expert's cross-attention `k_proj`/`v_proj` (8 of 16 layers), which are f32 |
| Quantized | prefix: 187 contractions (patch-embedding conv, 72 SigLIP linears, connector, 112 text-model linears, `state_proj`). flow: 116 (112 expert linears, 4 action/time projections). `action_decode` is a slice with no contraction and stays float. No sum or mean is quantized |
| Integerized where | the f32 linears (1 prefix, 20 flow) during capture; the bf16 ones (186 prefix, 96 flow) by Merlin's `contraction_int8`. The capture computes each bf16 linear in f32 behind an exact `arith.extf` widening of each dequantized operand. `quant_scope.value_preserving_input` lets the captured scope and grouping see through those widenings, so the captured static scales are reused |
| Weights / activations | per-tensor static, as in the recipe, with one exception. In the INT8 reference the **patch embedding** re-quantizes both of its operands at run time: the weight per output channel and the patch matrix per patch, each with `s = max(absmax/127, 1e-5)` and range ±127. Static-scale reuse does not see through that operand chain (an im2col gather), so `lower_contraction_int8` takes its dynamic path |
| Calibration (r53b) | one sample per program, the program's own capture input (ALOHA episode 0, frame 0, noise seed 0). The loader defines no calibration hook |
| Float, on host | all attention (QKᵀ and PV in the vision tower, text model and expert), LayerNorm/RMSNorm, GELU-tanh, SiLU, softmax, RoPE, embeddings, residual adds, the Euler flow update, all epilogues |
| Device kernels | int8 → int32 only. The patch embedding is an **activation-stationary** matmul, M=768 K=768 N=1024: the patch matrix occupies the stationary (weight) slot. Census: prefix 187/243 and flow 116/148 contractions on the device; the rest is attention |

**TinyLlama** (programs `prefill` with 128 tokens, `decode` ×32)

| Item | Value |
|---|---|
| Source model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, **float32**, untied `lm_head`, default (SDPA) attention |
| Quantized | 155 linears per program (`q,k,v,o,gate,up,down` × 22 layers, plus `lm_head`), all integerized at capture. No sum or mean is quantized |
| Weights / activations | per-tensor static, as in the recipe. Each program has its own activation scales; the weights are shared |
| Calibration (r53b) | one sample per program, its own capture input: 128 uniformly random token ids (seed 0) for prefill, and one decode step on the same sequence for decode |
| Float, on host | attention (44 contractions per program), RMSNorm, SiLU, RoPE, softmax, embedding, residual adds, all epilogues |
| Device kernels | int8 → int32; 155/199 contractions per program on the device |

## 5. Opt-in: quantized attention (`activation_contraction`)

Attention is float in every r53b capture. The recipe family `activation_contraction`
(`quant_recipe.with_activation_contractions`) quantizes contractions of two run-time tensors. Build a
variant of any derived static per-tensor integer recipe with:

```sh
python -m merlin.targetgen.quant_recipe activation-contractions IN.json OUT.json
```

Under this family, `spell_attention` in `_recipe_quantizer.py` rewrites each fused SDPA call as two
matmuls around a softmax. Q, K, the softmax probabilities P and V each take the activation spec: one
static symmetric int8 scale per tensor and call site, shared by all heads. The product of two
per-tensor scales is again one scale, which the readout can hold. A reduction of 16 or less is not
quantized. Calls with dropout, `is_causal` or grouped-query broadcast stay fused. The recipe digest
changes, so a capture made with the variant is never confused with one made without it.

Tested in isolation (linears in float, 8 visible inputs), this meets TinyLlama's bar: mean KL 0.011,
top-1 93.6 %. It fails SmolVLA's bar by a wide margin: worst action-dim relRMSE 0.85 on average over
the inputs, because one static scale for P over 113–1024 keys loses the probability mass. On Gemmini,
quantized attention runs as ordinary int8 matmuls with the run-time operand in the weight slot (no
prepacking).

Finer scales fix SmolVLA's attention (same isolated test; float noise 0.026 mean, 0.058 max):

| Variant | Worst-dim relRMSE, mean / max over inputs | int8 MACs vs one matmul per contraction |
|---|---|---|
| Static per-tensor Q, K, P, V (above) | 0.85 / 1.13 | 1× + 1× |
| QKᵀ with per-token Q and K scales computed at run time, K shifted by its mean over tokens; PV in float | 0.039 / 0.073 | 1×; PV on the host |
| As above, plus P in **2 int8 planes** (per-row P scale, per-channel V scale) | 0.044 / 0.075 | 1× + 2× |
| As above, P in 3 planes | 0.043 / 0.065 | 1× + 3× |
| As above, P in 1 plane | 0.148 / 0.309 | 1× + 1× |

Subtracting K's mean over tokens shifts every score of a row by the same constant, which softmax
ignores, so it changes nothing in float. All scales sit on non-reduction axes, so each int32
accumulation stays exact on the device and the host applies the scales afterwards as an outer product.
The 2-plane variant is the cheapest all-int8 option within the bar. It is being implemented as general
graph rewrites around plain int8 matmuls; no new kernel type is needed. The host cost is the
outer-product rescale of every score element per plane, plus element-wise absmax and mean passes.

## 6. How numerics are judged

Contract: [`docs/reference/model_check.md`](../../../docs/reference/model_check.md), §3–§4.

- **INT8 reference.** The prepared program (§1), evaluated by `merlin.runtime.linalg_numpy.evaluate`,
  which reads each op's semantics from the IR and shares no code with any lowering. Integers wrap at
  their width, f32 is float32, and bf16 is computed in f32 with round-to-nearest-even after each op.
- **Device kernels** must be **bit-identical** to the reference's boundary outputs (§4.2). The
  expected value of a kernel call is its interface program evaluated exactly
  (`src/merlin/model_check/kernel_reference.py`). A kernel that folds an epilogue into the readout
  must reproduce the reference's epilogue order exactly or leave it on the host.
- **Declared approximate kernels** follow the ratio rule:
  `metric(y, y_fp) ≤ max(1.1 × metric(y_ref, y_fp), floor)`, with floors (§4.3).
- **FP reference.** The float model in PyTorch eager with `ATEN_CPU_CAPABILITY=default` on one
  thread. Vectorized ATen kernels round bf16 differently: SmolVLA's final flow state moves by about 1 %
  between two valid hosts otherwise. SmolVLA's FP reference runs in float32, with the stock bf16 model
  as its baseline (`judged.reference` in [`models.yaml`](models.yaml)).
- **Reference admission (§4.5)**, for bundles rather than submissions. The bars are data:
  `judged.admission` in [`models.yaml`](models.yaml), evaluated per split when the references are built
  (into `reference_quality.json`; a rule without data is `unknown`, never a pass).
  - ResNet50: mean top-1 agreement ≥ 0.9375 **per split** (8/8 with eight B1 inputs), and
    **mean** logits cosine ≥ 0.98 (user-approved successor; original r53b bar was 0.99).
  - TinyLlama: mean KL ≤ 0.075 nats and top-1 ≥ 90 % (approved). KL is not measured corpus perplexity.
  - SmolVLA: per input, the worst action dim's relRMSE (dims 0–5, over the 50-step action chunk),
    measured against an **fp32 execution** of the same checkpoint. The current declared rule uses
    mean and p90 over four noise draws per input (32 chains per eight-input split), each ≤ 2 × the
    corresponding stock BF16 error aggregate; maximum is report-only. Historical eight-input
    mean/max BF16 errors were 0.0345 / 0.052, not a current p90 certificate. The old comparison with
    BF16 float noise is reported, not gated. See the shared protocol for exact reduction semantics.
- **Inputs.** 8 visible inputs per model, plus calibration item lists, are in this folder (see
  [`README.md`](README.md)). The sealed inputs are private.
- **Running the check:** `python -m merlin.model_check check --submission S --model M --split visible
  --out R.json` (contract §2). `kernels-export` writes every kernel's `interface.mlir` and `abi.json`,
  which are exactly what a device implementation must compute.

## 7. Reproducing the same int8 model

To match Merlin bit for bit, an independent implementation needs the following.

1. **The same programs.** Use the captured `model.mlir`, `weights.safetensors` and
   `weights.safetensors.manifest.json` of each program, identified by the recipe digest (§3) and the
   program's `model.mlir` sha256 from its `capture_receipt.json`:

   | Program | `model.mlir` sha256 |
   |---|---|
   | resnet50 `model` | `3e54ab8a76afa581c9809ffe790c30bb5f90ebfcb6acb22afac3a4268bedc761` |
   | smolvla `prefix_encode` | `97e9d47758e5e51187f68f607a93700ab33f1e4aecbfa6826d009403a5d07cc2` |
   | smolvla `flow_denoise` | `0c4f5c9df81a8f6d924a348e85cf3670381d6beebf4af3a8ca44cc75dd311115` |
   | tiny_llama `prefill` | `4102afc50f3891ce3f4cb52e84686c205adf9b7754472f1fc2ae74e9d23ef428` |
   | tiny_llama `decode` | `43581ac03553366c753d11ec2738c99677a2cf4c9ccfc1d0654c6e7bf1567bcf` |

   These identify the r53b captures. A recapture under another recipe (§8) changes them.
2. **Scales and zero points**, read from the capture rather than recomputed:
   - For a quantize or dequantize, the scale is an f32 constant passed as a rank-0 tensor
     (`tensor.splat`). The zero point is an i64 0, and `quant_min`/`quant_max` are op attributes.
   - For a contraction integerized at capture, the scales are the two f32 constants multiplied right
     after `arith.sitofp` of the int32 accumulator, activation first. The same values are listed per
     contraction in `meta.json` → `integerization_receipt.qparams[]`, together with `kind`,
     `reduction_k` and the quant range.
   - The recipe, observers and calibration count are in `meta.json` → `quantization_stats`.
   - int8 weights are the `_frozen_param<N>` tensors in `weights.safetensors`. The manifest is keyed
     by `@forward` argument index. Biases and norms are f32 or bf16 parameters.
3. **The arithmetic in §2**: reciprocal-multiply quantize with round-half-to-even and clamp, the
   two-multiply epilogue in operand order, bf16 rounding where the program rounds, SmolVLA's dynamic
   patch-embedding scales, and `clamp_i8(round_half_even(acc · acc_scale))` for the average pool.
4. **The right ground truth**, which is the INT8 reference. It can be regenerated from a capture
   directory with `merlin.model_check.catalog.prepare(stage, out_dir, target="gemmini")`, which writes
   `ref/model.prepared.mlir` and `ref/quantization-policy.json` (`passes: [contraction_int8]`,
   `selection: captured_scope`); evaluate it with `merlin.runtime.linalg_numpy.evaluate`. Two other
   candidates are **not** the ground truth:
   - The capture's `integer-reference.json` (`m2m/capture/pt2e_integer_reference.py`) runs every
     non-contraction op in PyTorch eager and keeps bf16 Q/DQ contractions in float.
   - PyTorch fake-quant (dequantize, then float contraction) evaluates in a different order. The
     capture's own `portable_agreement` check (tolerance 1e-3) fails on ResNet50, TinyLlama and
     SmolVLA `flow_denoise`.
5. **Fusing requantization into the readout** is allowed only when it is exact. One integer
   accumulator seed plus one f32 `acc_scale` must reproduce `(acc·s_x)·s_w + bias → ReLU → quantize`
   over every reachable accumulator value. `group_exact_float.prove` checks exactly this. Otherwise,
   return int32 and apply the reference epilogue on the host.

## 8. Status and work in progress (2026-10-16; this section will change)

The r53b INT8 references do not yet meet their admission bars. Values below cover 16 inputs (8
visible, 8 sealed) from the development bundle's `reference_quality.json`:

| Model | r53b INT8 reference vs FP | Main causes found |
|---|---|---|
| ResNet50 | top-1 14/16; logits cosine 0.88–0.99 (mean 0.94) | synthetic calibration images; per-tensor weight scales |
| TinyLlama | mean KL 4.3–5.9 nats per input; top-1 6–21 % | one-sample synthetic calibration; one massive input channel of `mlp.down_proj` in layers 2, 7 and 21, 60 to 3000 times its 99.9th percentile |
| SmolVLA | worst action-dim relRMSE 2.0–3.4 | one-sample calibration; error concentrated in the prefix encoder, mostly the vision tower, then the text model |

Phase 1 runs against the r53b captures: numerics bundle v1 freezes them with the admission bars
waived, so the compiler work is not held up by recipe quality. Improved recipes ship as bundle v2.
The device-kernel forms stay plain int8 × int8 → int32 matmuls and convolutions throughout.

The following techniques are under evaluation. They are screening results, not commitments.

- **Calibration with real data**: 260 Imagenette train images, 128 WikiText-2 train segments and
  50 ALOHA frames, listed in [`calibration/*.json`](calibration/) and disjoint from every evaluation input.
- **Per-channel weight scales.** The device still computes int8 → int32; the per-column scale moves to
  the host epilogue.
- **Function-preserving outlier-channel splitting** in gated MLPs. In
  `out(act(gate(x)) · up(x))`, a channel j whose calibrated absmax exceeds a threshold t is split
  into k = ⌈m_j/t⌉ parts: row j of `gate` is repeated k times, row j of `up` is divided by k and
  repeated, and column j of `out` is repeated. Every matmul stays plain int8 per-tensor; the
  intermediate width grows by Σ(k − 1).
- **SmoothQuant-style migration** of per-channel activation range into the weights.
- **Hadamard rotations** (QuaRot and SpinQuant style) of linear inputs, compensated in the weights.
- **Per-token dynamic activation scales**, computed on the host.
- **Two-plane activations**: about 15 bits carried as two int8 matmuls, for sensitive layers.

Best results screened so far:

- **ResNet50**: real calibration, per-channel weights, and residual adds plus the average pool left in
  float (full capture and bundle). Top-1 16/16, mean cosine 0.991 (visible) and 0.993 (sealed).
  **Meets the bar.**
- **TinyLlama**: real calibration, per-channel weights, outlier-channel splitting and SmoothQuant-style
  migration (PyTorch screen, 8 visible inputs). Mean KL 0.042, top-1 92 %. **Meets the proposed bar.**
  Per-token dynamic activations alone screen at KL 0.009 and top-1 94.7 %.
- **SmolVLA**: GPTQ weights and static int8 activations, with 2 int8 parts (≈15-bit) on the prefix's
  linear layers, pass the fp32-based bar (§6) in PyTorch measurements on the 8 visible inputs, at about
  1.7× the linear MACs, all on Gemmini. Confirmation on the hidden inputs and in the capture pipeline is
  pending. Earlier screens, measured against the bf16 float noise, located the error: real calibration
  alone gives a worst-dim relRMSE of 1.2 (full capture); attention passes with the finer scales of §5;
  with static per-tensor activations and per-channel weights on every linear the error is about 25× the
  float noise, and per-token scales bring it to about 7×. Most of it comes from a few units: the text
  model's `down_proj` (text layer 3 alone is 44 % of the squared error) and the vision tower's `fc2`.

The sealed recapture will use whichever recipe is approved. Its recipe digest and capture hashes will
replace those in §3 and §7.

## 9. Open questions

- The average-pool kernel computes `round(sum · acc_scale)`, while the reference computes a float sum,
  a division and then a quantize. Merlin records the bound between them as one output step
  (`compute_groups._window_mean_of`, `exactness: float_reassociation`). The bundle's consistency gate
  checks bit equality only on the bundle's inputs. It is not proven for every input.
- The patch embedding's dynamic scales (§4) come from where static-scale reuse stops, not from the
  recipe. Whether the next capture keeps them is undecided.
- In the r53b `flow_denoise` capture, all 96 bf16 expert weight dequantizes carry the same scale
  constant. Whether this reflects the checkpoint or an observer effect has not been checked.
- TinyLlama's admission bar is still a proposal (contract §4.5).
- No residual add is routed to the device in the r53b bundle, although ResNet50's are quantized.
  Gemmini's load-time scaling rounds each operand, so it can be one step away from the reference
  (§2). This guide does not establish that this is the routing reason. Whether a later recipe moves
  them to the device is open.
