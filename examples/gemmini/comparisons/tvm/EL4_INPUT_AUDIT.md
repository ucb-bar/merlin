# Historical EL4 input audit

> Historical input audit. Current decisions and readiness are in
> [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md): protocol 8ec99308 only, TinyLlama
> 32+32 correction, pinned quality policy only, 4h/1h/1h tuning, and the approved
> median wall watchdog. The older quality/budget/active-cycle policy below is
> superseded; it is retained to explain the original findings.

This dated audit preserves earlier source/input findings. It is not the current
model protocol or preparation plan. All three models are now enabled; current
8ec99308 bindings, completed chunks and remaining E2E work are in
[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md). Earlier source identities and numerical
policies below are retained only to interpret their original evidence.

Audit date: 2026-10-26. The requested upstream refresh fetched
`el4-core-aten` at `f6f1e499565f7cb766da1b680f490cbc3dc0b54b`, the latest
published EL4 branch, and `feat/el4-experiment` at `0cc289de52a10bed256f32c90cfbb54a21b3c81f`.
The three requested files are identical on those branches and in this checkout:

| File | SHA256 |
| --- | --- |
| `protocol/experiment.yaml` | `6ebbc846ffdbc57608e9842ee1ac7466de0ef4444afeb11d9a58aa22d7f4b3ca` |
| `protocol/README.md` | `bc0e832e2050980d915c0c264889405bb3fbf688bf5b5f209d9928f0469690b8` |
| `numerics/models.yaml` | `7b9addd43b87e4ee1f72e78449d3a29760369699d304d8359be794382fcb9d13` |

Paths above are relative to `examples/gemmini/`. No merge of unrelated EL4
implementation changes is needed to obtain these versions. The separate
`feat/megaphase-feedback-integration` head `a2ff611c8eeb25fc611bfc13f7e919b69fa47e0d`
has the same protocol but a conflicting models file: 128-token TinyLlama prefill
and ResNet cosine 0.99. That combination is not silently substituted for the
selected EL4 32+32 profile.

The [machine-readable audit](../../../../out/artifacts/audits/gemmini-el4-input-conformance/v1/audits_gemmini-el4-input-conformance_v1_20261026T191206Z_2ac14a1/report.json)
retains the actual input comparison, checkpoint metadata and source hashes.

## What matches, and what remains unverified

| Prepared workload | Verified matches | Missing or different |
| --- | --- | --- |
| ResNet round-two release | The actual FP32 `[1,3,224,224]` input equals the processed selected v00 array element for element. Recovered input archive SHA is `aef47b171c1132f1aaec18a957499f85bc5c80bfbfde197121c8255709311263`. Capture records the selected checkpoint state hash, recipe digest, 260-image calibration and 53 conv + 1 linear integerization. | New source identity after 54 INT32 bias moves; frozen scales/weights retained and bounds prove equivalence to the former bias ordering. Agreement with the selected sealed prepared-IR oracle and fixed-set quality are not established by the current host/Spike controls. |
| Fresh synthetic TinyLlama | Selected checkpoint and 22/2048/5632/32-query/4-KV geometry; 32+32 calls, 99 outputs; static/group16 W8A8 arithmetic. | **Diagnostic only: fails selected protocol conformance.** Synthetic max-abs calibration replaces WikiText histogram calibration; SmoothQuant/outlier transforms and selected attention planes are absent. Cache ABI, positions/logit shapes, host math and reference producer differ. Do not replace pending selected-profile campaigns with it. |
| SmolVLA | Cached policy/backbone parameters; actual checkpoint has 16 VLM and 16 expert layers. Selected one-camera/48-language-slot/113-prefix/50-action-row contract is defined. | Final admitted processed observation is absent: all four recorded recovery attempts have the wrong hash. Selected prepared stage exports, frozen scales/integerized IR and all 33 admitted reference arrays are not available in the audited handoffs. No runnable selected-profile chunk is declared ready. |

ResNet's capture recipe content digest is
`b840ae04104798799670015e39d2a5be692a70bc2a6ada42eb75d6878dd2ffbd`;
its loader records checkpoint **state-dictionary** SHA
`8bea63dc89b0700f2acbc5b170ed0b4637daed4ab748949eefe50ebe95149ac0`.
The downloaded `.pth` container hash is different by design. These capture
receipts establish provenance, not a new independent model-quality verdict.

The synthetic TinyLlama cache is INT8 `[22,2,4,64,64]`, positions I64 `[1]`
and logits FP32 `[1,1,32000]`. Selected EL4 requires FP32 cache
`[2,22,1,4,64,64]`, positions I64 `[22]` and logits FP32 `[1,32000]`.
Its single-plane QK/PV and native/frozen-table host path also differ from the
selected static-heads/softmax2/partner2 and `portable_exp_trig` preparation.
Its NumPy integer/Torch host references are useful diagnostic controls but do
not replace `linalg_numpy` on the selected prepared ungrouped quantized IR.
Synthetic inputs were authorized; these additional recipe/ABI changes were
beyond that request. They are preserved explicitly as diagnostics and excluded
from the selected-profile handoff queue. No automatic numerical rewrite follows
from this audit. Earlier TinyLlama E2E evidence retains its original ELF scope.

## SmolVLA parameters and meaningful small chunks

Checkpoint metadata is available; finalized benchmark tensors are not.
The policy is `lerobot/smolvla_base` revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`, policy safetensors SHA
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
Backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` has safetensors SHA
`b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`.
These are local file identities; bind download/revision receipts before reuse.

The checkpoint's expert width is 720, MLP width 2048, query width 960
(15 heads × 64), KV width 320 (5 heads × 64), action channels 32 and horizon 50.
Config `num_expert_layers: 0` is an inheritance setting; the checkpoint contains
actual expert layers 0 through 15. Do not interpret it as an empty expert.
The checkpoint declares three cameras; the selected protocol explicitly uses
one, with its specified processing. Do not infer benchmark inputs from config defaults.

The following are **blocked extraction definitions**, also recorded in
[smolvla-chunks.pending.json](smolvla-chunks.pending.json). No SmolVLA campaign,
ELF, timing estimate or extra proposal allocation is added.

| Chunk | Real computation retained | E2E occurrence and boundary |
| --- | --- | --- |
| Action embedding | All 50 rows: `32→720`, timestep embedding/concatenation, `1440→720`, SiLU, `720→720`; actual biases, quantization and every selected operand plane. | `flow_denoise` suffix input, ten calls. Use real action/time values from an admitted chain. |
| One expert MLP | One complete residual/norm/gate/up/SiLU/product/down region: `720→2048` twice and `2048→720`, all 50 rows and full reductions. | One selected expert layer in `flow_denoise`; inventory other layers before schedule reuse. |
| One expert self-attention region | Full Q/K/V/O, RoPE, masks, QK/softmax/PV and residual; all 15 query/5 KV heads, 50 queries, 113 prefix + 50 action keys. Q/K/V project `720→960/320/320`; O is `960→720`. | Even-index expert layers under the source's every-two-layer policy. Preserve the 163-position context, cache, all selected planes and host reconstruction. |
| One expert cross-attention region | Full Q/K/V/O, masks and QK/softmax/PV over 50 queries and the actual 113-position prefix cache. Q is `720→960`; cached K/V projections are each `320→320`; O is `960→720`. | Odd-index expert layers. This is structurally different from the self-attention region, including its K/V input width. |
| Action projection and Euler update | All 50 rows: `720→32` plus bias, dequantization and the actual `actions + dt × velocity` update; retain all 32 output channels. | Inside each of ten `flow_denoise` calls. Compare projected velocity and updated action boundary as available. |

The final `action_decode` stage is an identity/slice retaining all 32 channels;
it is **not** the `720→32` projection and offers no contraction DSE task.
The older `smolvla-action-out` ready probe is explicitly a synthetic-shape
surrogate with legacy numerics. It is not an admitted selected-model chunk.
Prefix vision/VLM work remains uncovered by this initial definition roster;
the roster is not a complete SmolVLA DSE or an E2E coverage claim.

Extraction must use the actual selected prepared IR, including both bounded
operand planes and softmax planes. Parameter shapes alone do not specify those
integer programs. Retain full M/N/K, real live inputs/outputs, masks, cache,
norms/epilogues and layout costs. One layer and one denoise step keep the chunk
small without shrinking its token count, reductions, output width or heads.
First/middle/last-step boundaries are confirmation cases, not new search axes.
Actual exported cache/mask extents take precedence over the source-derived
definitions above; refuse mismatches instead of padding/truncating to fit.

SmolVLA expects observation archive SHA
`8c4c0dca184abf8da7af74c187531cecc172c3eb50f51aa7b6d6d0c61fef3daf`.
Attempts 1–4 instead produced
`e4863c12cb7f9ae9bdf9f20ec48d6788083a81f29bb680738f420daed5405414`.
Matching shapes does not waive that input gate. Raw frame/state and policy
weights exist; do not choose a different frame, noise draw or processor silently.

## How tuning decisions reach the E2E model

1. Slice a connected region from the frozen parent export. Record its parent
   hash, exact FX/Relax operation interval, live input/output edges, constants,
   fixture hashes and every source call occurrence. Boundary values come from
   that same numerical release. This is traceability, not model-name dispatch.
2. Extract tasks with the same existing TVM pipeline, graph fusion fixed at 2.
   TVM EvolutionarySearch/XGB explores legal contraction schedules and stores
   traces/costs in its database. Model dimensions, heads, scales, planes and
   calibration are fixed computation, not tunable parameters.
3. Re-extract the full parent graph with that pipeline and match records by
   **structural TIR workload equality**, target/capabilities and compatible
   measurement provenance. A short hash is an index, not proof of equality.
   Bind fixtures to actual operator call arguments and validate shapes/dtypes,
   constants and `complete_inputs`. Same matrix dimensions alone are insufficient.
4. Reapply the matching stock TVM schedule trace to every corresponding full-model
   task occurrence. Quantization/layout/tails/epilogues or operand-plane differences
   can prevent a match. Report unmatched tasks and their defaults explicitly.
5. Compile and check connected chunks, then the full session with actual carried
   state. Measure complete baseline/tuned guest windows. Summed task minima
   cannot predict E2E time or replace that measurement.

For example, ResNet task `47cf36513e812255` has owner `resnet-stage1-block1`
and matching occurrences in all three stage-1 blocks. The record may be reused
through structural matching at each occurrence. SmolVLA's action projection is
called ten times with different activations; reuse requires the same extracted
workload and fixed numerical identity. Self/cross-attention K/V tasks differ
despite having the same model name. A TinyLlama down panel with N=128 cannot
stand in for N=2048; synthetic untransformed records likewise do not qualify
selected SmoothQuant/outlier-expanded workloads.

Current evidence establishes ResNet's chunk inventory/ownership, not a completed
new tuned E2E replay. A full-parent match roster, selected record per occurrence,
unmatched defaults and connected/E2E confirmation receipts must still be
produced by the next campaign. SmolVLA has definitions only, so it has no actual
operator bindings or schedule records yet.

## Blockers and execution boundary

- **ResNet selected-reference admission:** provenance and functional controls are
  available, but the sealed prepared-IR reference comparison remains unverified.
  Bind an admitted oracle for the new bias-source identity before claiming
  selected-profile correctness. Fixed-set accuracy is separately NOT_RUN.
- **TinyLlama selected release:** the new synthetic release is nonconforming as
  detailed above. Obtain the selected prepared exports/scales/IR and admitted
  references, or agree on an explicit new profile before modifying numerics/ABI.
  The selected v00 token file itself is recoverable with the required hash.
- **SmolVLA inputs and numerics:** resolve the observation hash; obtain selected
  stage exports, frozen recipe realization and all 33 reference arrays. Only
  then extract/qualify the definitions above and allocate their DSE proposals.
- **Quality policy alignment:** upstream `models.yaml` still specifies ResNet
  cosine/top-1 agreement, TinyLlama KL/agreement and SmolVLA BF16-relative RMSE.
  The user's later [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md)
  requires ResNet ≤1 percentage point top-1 loss, TinyLlama test PPL ≤1.10×FP32
  and SmolVLA cosine ≥0.99. Retain both identities; do not label one metric as
  passing the other. The later user quality gates remain required for new runs.
- **Paid-run prerequisites:** qualify actual live active-window cutoff,
  tightening and cancellation on the remote adapter. Stop at active cycles
  greater than or equal to the incumbent for the same workload; startup is
  excluded. Bind actual FireSim image/loader/hardware identities. DDR3 is
  expected from the bitstream; this audit adds no hardware or memory settings.

Keep the existing shared four-hour maximum, cumulative proposal cap 96,
physical-slot locks and bounded parallel builds. No new compiler knobs, host
schedule/packing implementation, unreported CPU contraction fallback, paid
job launch or new numerical profile is authorized by these definitions.
