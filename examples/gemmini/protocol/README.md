# Shared Gemmini experiment protocol

This is the shared entry point for Merlin and independently implemented
baselines. Use the [baseline comparison workflow](baselines.md) to export a
hash-bound selection and check saved arrays with the common numerical code.

Use [`experiment.yaml`](experiment.yaml) to agree on workload shapes and inputs,
and [`numerics/models.yaml`](../numerics/models.yaml) as the single numerical
policy: common `numerics` semantics and each model's `judged.admission` thresholds.
The protocol links that policy instead of copying its rules. It records
the full-model profile selected on 2026-10-19. The exact, data-only
[`recipes/`](recipes/) are copied byte-for-byte from that selection. They contain
no kernels, checkpoints, tensor values or reference outputs.

This definition complements the [experiment catalog](../experiment.yaml), the
[Phase 1 launch config](../phase1/launch/phase1_el4_config.yaml) and the
[Phase 2 templates](../phase2/README.md). Those launchers still read their own
configuration; this document does not silently override them. Archive the resolved
launch config and a completed [run manifest](run-manifest.template.yaml) with each
run. A new recipe, input shape, camera count, math policy or calibration creates a
different experiment and needs a new profile and references. Preserve old receipts.

## Two profiles and three comparisons

`functional_v1` is the selected functional workload: batch 1, all full-network
layers, one bound input per model, and every declared output at every step.
The full-model check compares compiled execution directly with the **selected
quantized reference**, with independent exact accelerator-kernel checks. Float
values must satisfy `abs(actual-ref) <= 2e-6 + 1e-5*abs(ref)`; integer/bool values,
shapes, dtypes and output arity must match exactly. Nonfinite values fail. Carried
caches, positions and padded action channels are included, not just final logits
or the six judged robot channels. The selected external input is `visible/v00`.
Its derived-file hashes match both the selected evaluation freeze and public
`numerics/MANIFEST.json`: one Imagenette validation image, WikiText test article 0
and ALOHA episode 0/frame 100/noise seed 1000. Capture sample provenance is
recorded separately; those samples are not the external evaluation inputs.

The sealed quantized oracle evaluates prepared, ungrouped quantized IR through
`merlin.runtime.linalg_numpy` with the selected scalar math policy. The capture's
PyTorch fake-quant `integer-reference.json` and recipe-side `integer_reference`
helper do not substitute for that oracle.

`quality_v1` defines separate evaluation against the floating-point model on
held-out real inputs. Its rules do **not** add a Phase 1 exit gate: FP quality is
report-only for the selected direct quantized-reference execution check. Eight
observations per visible/sealed split and four noise draws for each SmolVLA
observation give 32 action chains per split. Missing measurements are unavailable,
not a pass. The offline section reader computes RMS/error and true continuation
perplexity from existing bound outputs; it does not need new input manifests or
model execution. Cohort quality admission still needs the full stated cohort.

Keep these comparisons distinct in every report:

| Comparison | Question | Evidence |
| --- | --- | --- |
| Compiled vs selected quantized reference | Does this compiler implement the selected numerical program? | Complete execution, all-array comparison, exact kernel checks, placement and dispatch receipts |
| Quantized reference vs FP32 | What error does this quantization introduce? | Paired inputs, selected math policies, per-model metrics and cohort coverage |
| Output vs labels/task outcomes | Does the model perform the task? | Classification labels, real-text NLL/PPL or robot rollout outcomes |

Capsule completion publishes its frozen compiler and capsule certificate
independently. Full-model validation runs outside that clock and has its own
pending/pass/fail state. Neither completion state substitutes for the other.

## Full networks and exact interfaces

Dimensions below include batch. Argument/output order is normative in the YAML.
`i1` denotes a boolean tensor; `f32` denotes IEEE float32. All selected runs use
batch **1**. Batch 4, 132-token prompts, multiple cameras or fewer denoise steps
are new profiles, not equivalent measurements.

| Model / stage | Invocations | Inputs, in order | Outputs, in order |
| --- | ---: | --- | --- |
| ResNet50 / model | 1 | image `f32[1,3,224,224]`, NCHW | logits `f32[1,1000]` |
| TinyLlama / prefill | 1 | tokens `i64[1,32]` | logits `f32[1,32000]`; KV `f32[2,22,1,4,64,64]`; positions `i64[22]` |
| TinyLlama / decode | 32 | token `i64[1,1]`; KV; positions | logits; KV; positions, same shapes as prefill |
| SmolVLA / prefix_encode | 1 | image `f32[1,3,512,512]`; image mask `i1[1]`; tokens `i64[1,48]`; token mask `i1[1,48]`; state `f32[1,32]` | prefix mask `i1[1,113]`; cache `f32[2,16,1,113,5,64]` |
| SmolVLA / flow_denoise | 10 | prefix mask; cache; actions `f32[1,50,32]`; time `f32[1]` | actions; cache; time, same shapes |
| SmolVLA / action_decode | 1 | actions `f32[1,50,32]` | actions `f32[1,50,32]` |

One complete check of all three networks contains **46 invocations and 133 output
arrays**: ResNet50 1/1, TinyLlama 33/99, SmolVLA 12/33. These counts describe
the selected chains; they are not measurements of completed execution.

**ResNet50.** Use the pretrained torchvision `IMAGENET1K_V2` checkpoint with the
SHA in the YAML. Its capture sample is synthetic, seed 20260830; the
weights are pretrained, not random. External validation uses the real Imagenette
`v00` image identified in the YAML. Calibration used 260 real Imagenette train
images, 26/class, with a 1024 sample limit. The captured session describes a
256-image stream; the selected external check executes **one** full-network
image per case. For the quality profile use the attributed Imagenette validation
selection in [`sources.yaml`](../numerics/sources.yaml): resize 232 with bilinear
antialiasing, center crop 224, then float32 normalization with torchvision V2
mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`. Do not substitute that
preprocessing for the current synthetic capture and reuse its old reference.

**TinyLlama.** Use the complete 22-layer
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at the pinned revision, hidden 2048, FFN 5632,
32 query heads, four KV heads, head dimension 64 and vocabulary 32000. The
checkpoint declares BF16; this loader loads FP32 before INT8 capture. Prefill
consumes 32 tokens; each of 32 decode invocations consumes the next fixed input
token. This is **teacher forcing**, not argmax-fed generation. KV axes are
`[K/V, layer, batch, KV head, capacity, head dimension]`; reset cache/positions
for each case and carry the preceding compiled outputs within a case. The
selected capacity is 64, KV dtype f32. Only the last position's logits are
returned per invocation. Capture sample tokens are synthetic seed 0; external
validation consumes the first 64 tokens of the stored 160-token WikiText `v00`
article. Calibration uses separate WikiText training data (128 prefill / 1024
decode samples).

**SmolVLA.** Use the pinned `lerobot/smolvla_base` and underlying
`SmolVLM2-500M-Video-Instruct` checkpoints, with the full selected policy's
16 VLM and 16 expert layers. The checkpoint's `num_expert_layers=0` means inherit
16 in this constructor. The underlying text config's 32 layers does not change
the policy selection. No new parameter recount is claimed. The wrapper supplies
**one camera** resized/padded to 512, although the checkpoint declares three
camera features. Apply the pinned LeRobot processors, tokenizer length 48,
MEAN_STD state/action normalization and VISUAL IDENTITY policy; bind processor
code and normalization-stat bytes in the run manifest. The capture sample is
ALOHA insertion episode 0/frame 0, seed-0 noise. External validation uses
episode 0/frame 100 with seed-1000 noise, matching public `v00` metadata.

Execute prefix once, then ten recurrent denoise steps from `t=1`, `dt=-0.1`,
carrying actions, cache and time from each compiled output. Decode retains
`[1,50,32]`: six semantic action features and 26 padding channels. The selected
decode is an identity/slice, not a physical-unit robot postprocessor. Report
action error in **normalized policy space**, not meters/radians or task success.

## Quantization and Gemmini placement

The three pinned recipes use symmetric static INT8, activation scales per tensor
and weight scales shared by **16 output channels**. Weights use `[-127,127]`,
activations `[-128,127]`, zero point 0, accumulation i32. Weight observers are
min/max; activation observers are histogram with epsilon `2^-12`. Reuse captured
scale bytes. Changing scale granularity, calibration, bias handling or attention
planes changes the numerical reference.

The selected `integer_epilogue_v1` preloads integer bias rounded half-even using
the bias qparam `f32(s_x*s_w)`. For a narrowing INT8 exit, selected ReLU acts in
accumulator space, then conversion to f32 and **one** f32 multiplier per
16-column command, followed by half-even rounding and saturation. Form that
multiplier as `float64(s_x)*float64(s_w)/float64(s_out)`, then cast once to f32.
Raw **INT32 readout applies neither scale nor ReLU**: it returns the accumulator;
the remaining dequantization/epilogue is host work. Its dequantization product is
formed in float64 then cast once to f32. This differs from the historical r53b
two-multiply order. The captured IR and exact recipe remain the authority; do not
reassociate arithmetic silently or assume the INT32 readout performs an FP exit.

| Model | Selected numerical variant | Deliberate details |
| --- | --- | --- |
| ResNet50 | `native-epilogue-contractions-calibration-1024` | 53 conv + 1 linear source contractions; 32 requant/ReLU and 22 raw-accumulator exits. Residual adds and average pool remain floating host work in this recipe. |
| TinyLlama | `static-heads-softmax2-bounded-partner2` | INT8 attention with static head scales: two softmax planes and two partner planes; ordinary QK uses one plane. SmoothQuant alpha 0.5 and legal outlier-channel splitting expand widths, not the layer count. |
| SmolVLA | `static-heads-softmax2-bounded-operands2` | INT8 attention with two softmax and two residual-operand planes. Prefix/flow have 187/116 source contraction decisions; plane/head expansion changes the emitted call count. |

Attention's non-softmax head scales are **calibration-time constants**, derived
from each head's calibration absmax as `max(absmax/127.5, 2^-12)` and stored as
f32. The bounded two-plane encoding uses steps `s` and `f32(s/128)`; each plane
rounds residual/step half-even, clamps to `[-127,127]`, then subtracts the
reconstructed plane. Softmax uses `e=exp(scores-rowmax)`, fixed plane steps
`1/127` and `1/(127*128)`, and the same residual construction. Plane products are
rescaled and added in the captured order, then divided by the exp-row sum floored
at `1.1754943508222875e-38`. Do not replace this with a differently rounded
quantized-probability matmul. See `_static_attention.py` and the exact recipe's
plane selection; these are INT8 products with explicit floating normalization.

Floating islands follow the FP32 policy; SmolVLA flow records two wider rotary
exceptions. Selected host math policies are ResNet50 `native`, TinyLlama
`portable_exp_trig`, SmolVLA `portable_neural_f32`. The default public
`numerics/models.yaml` does not supply those three overrides: the selected sealed
models config does. Record that exact config and reference identity.

Every operation eligible under the selected numerical program and independent
Gemmini contract must execute on Gemmini. Retain the region/work census, refusal
reasons, dispatch and protocol traces. An INT8 capture census is not a runtime
offload certificate. Merlin supplies host work outside eligibility; host-only
groups do not become accelerator-authoring capsules. Quantizing additional
floating work requires a new explicit reference, not a concealed compiler change.

## Numerical metrics and aggregation

Let `e = actual-reference`, over the stated nonempty array. Existing relative-L2
and KL reductions use float64; existing logits cosine keeps the input dtype
(currently float32). Preserve those semantics for the declared quality rules.
The offline reader uses float64 for additional RMS/MAE and PPL diagnostics.
Report the compared quantities and cohort, not just a scalar.

| Metric | Definition / interpretation |
| --- | --- |
| RMS(reference) | `sqrt(mean(reference²))`; report alongside RMSE to explain scale |
| RMSE / MAE / max absolute error | `sqrt(mean(e²))` / `mean(abs(e))` / `max(abs(e))` |
| Legacy `rel_rmse` | `norm2(e)/norm2(reference)`, equal to RMSE/RMS for nonzero denominator. If reference norm is zero, existing code returns **unnormalized norm2(e)**, without epsilon; both zero gives 0. Flag this fallback. |
| Legacy `worst` | Maximum absolute error divided by explicit scale when nonzero, otherwise maximum absolute reference value, or 1 when all reference values are zero |
| Cosine | `dot(actual,reference)/(norm2(actual)*norm2(reference))`; zero norm is unavailable and cannot certify quality |
| KL | `KL(softmax(FP) || softmax(quantized))`, stable log-softmax, nats |
| Top-k agreement | Fraction of FP argmax labels appearing in candidate top-k; this is agreement with FP, not dataset accuracy |

The offline reader reports every absolute diagnostic above. Its `relative_l2`
is unavailable for zero reference energy, while absolute errors remain reported.
This leaves the legacy gate's historical `rel_rmse` fallback unchanged.

Reference quality rules are evaluated **per split**:

| Model | Rule | Reduction |
| --- | --- | --- |
| ResNet50 | Mean cosine ≥ **0.98**; mean top-1 agreement ≥ **0.9375** | Each B1 image gives one logits comparison. With eight binary top-1 cases per split, 0.9375 requires **8/8 in each split**, not pooled 15/16. |
| TinyLlama | Mean KL ≤ **0.075 nats**; mean top-1 agreement ≥ **0.90** | Mean over 33 emitted distributions per case, then mean over equal-size cases. These distributions do not constitute a PPL calculation. |
| SmolVLA | Mean and p90 worst-channel relative RMSE ≤ **2×** the corresponding native-BF16-vs-FP32 baseline aggregate | Per noise draw, reduce across 50 steps per semantic channel, take worst of six channels, then aggregate 32 chains per split. Use linear-interpolated p90. Compare aggregates, not averages of per-case ratios. Maximum is report-only. |

SmolVLA noise draw 0 uses stored torch-generated noise. Draws 1–3 use the existing
deterministic NumPy policy keyed by split/id/noise/draw; pin that implementation
and seed roster. Compare candidate and BF16 baseline against the same FP32
reference, observation and noise. Retain per-channel results and absolute error
diagnostics, especially for channels with near-zero reference energy.

Ground-truth ImageNet top-1/top-5 is a separate measurement reported by the
offline reader when label metadata matches the evaluated input's manifest hash.
It does not change legacy `model_metrics`. Current logits quality code assumes
B1 and flattens logits; batch-4 quality needs explicit row semantics before use.

## True perplexity from retained outputs

The `judged.reporting.perplexity` policy reuses the existing selected WikiText
token inputs and 33 retained logits, without recapture or retokenization. Consume
the first **64** tokens of each stored 160-token input, no cross-document packing,
no added BOS/chat template, no padding. Keep existing source/tokenizer/input
identities; calibration remains training-only. The current report covers one
selected input; it is not a dataset-wide benchmark.

Run the same 32-token prefill and 32 teacher-forced decodes. Score target token
positions **32–63** using logits at positions **31–62**: the prefill result plus
the first 31 decode results. Exclude the last decode's logits, which predict
absent token 64. Prompt and padding are masked; score an existing EOS only when
it has a valid target, without implicit insertion. Reset KV between segments.

Compute `NLL = sum(logsumexp(logits)-logits[target])/total_scored_tokens` using
stable float64 reductions, then `PPL=exp(NLL)`. Aggregate by scored tokens,
not by averaging sequence perplexities. Pair FP32, selected quantized reference
and compiled results on identical tokens/masks; report delta NLL and
`PPL_quant/PPL_FP = exp(NLL_quant-NLL_FP)`.

PPL is **report-only**, without an invented acceptance threshold. Missing or
misaligned targets make it unavailable. In particular, KL 0.075 does not imply a measured
7.8% corpus-perplexity increase; KL averages under the FP distribution, not the
observed ground-truth token stream.

## Capsule inputs and performance reporting

The independent PyTorch workload roster is
[`target/phase0-workloads.yaml`](../target/phase0-workloads.yaml). It selects
small residual CNN, causal prefill/decode and multimodal mechanisms for Phase 1,
and larger CNN/decoder/ViT mechanisms for Phase 2. Loader defaults plus explicit
profile overrides define their shapes. Phase 0 also extracts/synthesizes MLIR
capsules where needed. Retain each capsule's source role, loader/profile hash,
shape, dtype, alignment, admission and required L0–L3 evidence. These mechanisms
are independent stand-ins; they are not reduced full-model success claims.

The maintained [capsule calibration requirements](capsule-calibration.yaml)
define the small external measurement batch: independent-column readout beyond
the observed training range, the raw output-tile control, and capacity-boundary
controls derived from the selected memories. Reopen compatible archived evidence
first and request only missing measurements for the selected frozen compiler and
runtime. Preserve the original source-group splits, including held tails.
Source coverage and logical memory footprints do not prove dynamic store counts,
buffer overlap or spilling. Each requires the recorded execution evidence.
Connected attention, GQA and KV reuse remain unadmitted until their complete
arithmetic, state and effect contracts are selected; batched dots cannot cover
them. These requirements are included in the resolved protocol export, without
changing an existing frozen compiler, corpus or qualification.
Phase 2 admission must bind this definition alongside its selected evidence;
this file does not choose its cohort, change its queue or launch measurements.

For reproducible latency, retain simulator cycle counters and simulator wall time
separately, plus exact clock, hardware facts and correspondence proof. Spike and
gSIM must bind the **same FireSim Gemmini Rocket configuration**. Phase 1 uses
untuned Spike. Phase 2 must version its fitted timing model, training and held-out
gSIM capsule labels, supported shape domain and prediction error. Surrogate
cycles do not become FPGA measurements by multiplying by a clock.

Report cold initialization, warm complete-chain latency, host/transfer scope and
accelerator cycles. For TinyLlama retain prefill, every decode step and total;
for SmolVLA retain prefix, all ten flow steps, decode and total. State whether
weights/cache setup is timed and how many warm-ups/repetitions were completed.
Bind MAC-count convention, peak throughput, bytes and memory level when reporting
roofline gaps. No ground-truth numerical or latency result is supplied here.

Select finite Phase 2 wall-time/token budgets and a plateau/minimum-improvement
rule before launch. The old whole-model objective is a template with its own
longer budgets, not an implicit budget for this profile. CPU reservations must
be scoped to the active experiment and released on terminal exit.

## Check the definitions without running an experiment

From a provisioned Merlin checkout:

```sh
python examples/gemmini/protocol/check.py
```

This checks recipe file/content hashes, recipe formats, numerical-policy links,
selected launch comparison tolerances, stage counts and PPL score selection.
It does not run capture, calibration, an agent or a
simulator; it does not certify a model or inspect private/sealed answers. Run it
when editing these definitions, outside the grading hot path. The per-run
manifest must bind the actual capture, references, host math, hardware and
resolved configuration before results can be reproduced.

Add `--export "$TMPDIR/gemmini-comparison-contract.json"` to write a portable
snapshot of the checked definitions, recipes and source hashes. The snapshot is
generated from these files and supplies no execution or numerical qualification.

For existing run numbers, use the [offline section report](../numerics/README.md#offline-section-reports).
