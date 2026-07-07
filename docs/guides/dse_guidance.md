---
title: DSE guidance
kind: guide
status: current
owner: dse
last_verified: 2026-07-07
related: [dse, design_pressure]
code_refs: [merlin/python/merlin/dse_guidance]
---

# DSE Guidance — Workload-Contract Analysis

## Framing

**Merlin is a compiler-based workload-contract analysis tool for accelerator DSE. It does not
choose hardware designs, and it does not calibrate against existing hardware to predict a future
one.** It recovers temporal, numerical, memory, and runtime-interface structure from real model
captures, identifies which HW/SW abstractions the workload appears to require, derives the
design requirements/theoretical pressure points that contract imposes, and emits a DSE-ready
**contract-analysis package** for a later DSE engine.

> Merlin tells a future DSE engine what the workload actually needs the HW/SW boundary to express,
> what any design must satisfy, and what facts are still missing — without picking a design or
> claiming a speedup.

### The contract-analysis package (per workload)

The DSE engine does not consume a flat MLIR graph + a list of knobs. It consumes
(`merlin-dse-guidance --case-study`, per `<workload>/`):

- `workload_contract_report.md` — the unified package (structure → numerical contract →
  requirements → abstraction candidates → measurement plan → readiness).
- `region_attribution.yaml` — region roles + real per-region IR facts (`prov.fqn`).
- `numerical_contract.yaml` — storage/compute/accumulator precision + what the capture lost.
- `design_envelope.yaml` + `requirements_table.csv` — hardware-independent requirements
  (compute/bandwidth/capacity/command-rate) + optional roofline feasibility vs a candidate design.
- `abstraction_candidates.yaml` — the HW/SW abstractions the workload implies (e.g.
  `resident_weight_object`, `bounded_loop_command`, `packed_lowbit_tensor + scale_object`), each
  with the compiler proof + runtime/HW support needed, the DSE knobs it exposes, and an explicit
  *what_is_not_claimed*.
- `measurement_plan.yaml` — what to measure next, split into measurable-now (accuracy + runtime
  **proxy**) vs needs-**target**-design.
- `dse_readiness.yaml` — is this workload ready for a DSE tool to rank designs, and what's missing.

For VLA accelerator DSE, Merlin recovers **two contracts that flat captures obscure**: the
**temporal contract** — which regions repeat, persist, or run at different rates (`topology.py`,
`attribution.py`) — and the **numerical contract** — which tensors are stored, computed,
accumulated, dequantized, and requantized in which precision (`numerical_contract.py`). It emits
structural DSE candidates only when those contracts justify them, and gates quantitative ranking
on measurement. Concretely: across the captured zoo, every int8/fp8 model stores weights low-bit
but runs **f32 matmuls** — native low-bit compute and the packed layout are absent from the
capture (a hidden DSE axis), reported by the numerical-contract audit with no speedup/accuracy
claim. See `merlin/benchmarks/dse_guidance/case_study/`.

## Design envelope vs calibration

We do **not** calibrate nonexistent hardware. For a not-yet-existent accelerator, a model fitted to
one existing instance does not transfer to a proposed design (and the P1 work showed whole-model
coefficients aren't even identifiable). Instead:

- **Requirements** are derived from the workload contract + the real-time deadline, hardware-
  independent: `required_compute_rate`, `required_bandwidth`, `resident_capacity` (per dtype),
  `required_command_rate`, `avoidable_weight_reload`. (`design_envelope.py`)
- **Feasibility / lower bounds** use **analytical roofline** against an *optional, hypothetical*
  candidate design (`compute_bound`, `memory_bound`, `latency_lower_bound`, capacity/dtype/command
  feasibility) — labelled `design_assumption`, not measured.
- **Measurement** grounds only (a) real system interactions that exist regardless of the future
  accelerator — CPU/runtime coupling (a `proxy_measured` host signal) — and (b) accuracy of a
  numerical contract (`accuracy_measurable_now`). Everything needing the proposed design is
  `needs_target_design` and stays gated.
- **Calibration is demoted** to a sanity-check / anchor for an *existing* target only
  (`cost_calibration.py`, `measured_cycles.yaml`) — never future-HW prediction.

So the precision↔capacity coupling becomes a clean, hardware-independent DSE insight, e.g. for RDT
the repeated head is **197 GMAC/replan**, **1.56 GB avoidable reload**, and a resident set of
**196 MB (bf16) / 98 MB (int8) / 49 MB (int4)** — datatype determines whether residency is even
feasible, with no speedup claimed.

## Pipeline order (structural first, quantitative last)

```
flat capture
  → VLA runtime topology         (topology.py — recover the contract)
  → capture fidelity report      (fidelity.py — what flattening lost, and the DSE risk)
  → structural DSE candidates    (candidates.py — which axes matter + what to measure)
  → measurement plan             (each candidate lists required_measurements)
  → quantitative axis triage     (triage.py — ONLY with a baseline; uncalibrated by default)
```

The structural layer (topology / fidelity / candidates) is valid **without** calibration and is
always emitted. The quantitative triage runs only when a baseline cost is supplied and is clearly
labelled uncalibrated unless its components are `measured`/`calibrated`. Two distinct outputs:

- `dse_candidate_axes.md` / `.yaml` — **structural, defensible now.**
- `axis_triage.csv` / `.md` — **quantitative, only valid when calibrated.**

### Workload classes and capture fidelity

`flow_matching_action_head` (Class A) and `autoregressive_decode` (Class C) have an inner
loop that flattening destroys → **high** DSE-risk severity. A `regression_parallel_head`
(Class B) has no inner loop → **low** severity. The capture-fidelity report names exactly which
structure was lost (`denoise_loop`, `prefix_kv_reuse`, `replan_deadline`, …) and which DSE axes
that loss hides.

### Recovering structure: three levels

- **Level 0 (implemented)** — sidecar `vla_runtime_topology` / temporal metadata (hand-authored
  K, H, control rate, region roles, loop-invariant/carried state). Every interesting fact is
  human-asserted.
- **Level 1 (partially implemented — `attribution.py`)** — attribute *real* per-matmul IR facts
  (count, MACs, weight/activation bytes, epilogue) from the captured `model.mlir` to topology
  phases, with **explicit provenance** for every assignment. What the captures carry: per-op
  `prov.*` provenance (`prov.region_id`, `prov.op`=matmul/addmm, shapes/dtypes). What they do
  **not** carry (in pre-`prov.fqn` captures): a backbone-vs-head marker. **Now fixed in
  `model2MLIR`:** the importer emits `prov.fqn` (the deepest `nn.Module` path, e.g.
  `vision_backbone.layers.3.attn` vs `action_expert.denoise.2`), so for freshly-captured models the
  **role is recovered automatically** from the module path (`attribution.role_from_fqn`,
  `source: prov_fqn`, verified end-to-end on a toy capture). For captures that predate `prov.fqn`,
  the role comes from an explicit operator mapping
  (`merlin/benchmarks/dse_guidance/region_maps/<model>.yaml`, by `region_ids`/`shape_signature`) or
  stays `unknown`. Either way per-region **facts are recovered exactly from IR**. Output:
  `region_attribution.yaml` (per role: status, source, confidence, IR facts; head facts ×K,
  backbone ×1; repeated-shape clusters; unresolved remainder). When a repeated head is attributed,
  its candidate axes carry the real facts and report `quantification_blocked_by: missing_calibration`;
  otherwise `missing_region_attribution`.

  **Demonstrated on a real capture (not the toy):** a fresh RDT denoise-step capture (the real
  `RoboticsDiffusionTransformer` architecture, small random config, via the `prov.fqn`-enabled
  `m2m`) — module paths `model.blocks.N.{attn,cross_attn,ffn}`, `model.{t,freq}_embedder` — has all
  20 matmuls **auto-recovered to `repeated_head` from `prov.fqn`** (no operator map), with real
  attributed facts: 20 matmuls, 39.4 GMAC/step, 391 MB weights, ×K. The
  `resident_action_head_weights` certificate carries those facts and stays
  `blocked_by: missing_calibration`. Fixture: `tests/fixtures/dse_guidance/rdt_recap_fqn/model.mlir`
  (+ `region_attribution.example.yaml`). Caveats: depth=2 (real 1B is 28) and random init — the
  *structure and provenance* are real, the magnitudes are a small-config instance.
- **Level 2 (long-term, in `model2MLIR` at `/scratch/agustin/projects/model2MLIR`)** — preserve
  `scf.for`/`scf.while` loops and region cadence in the capture IR itself, so roles need not be
  operator-supplied.

The concrete question that drives every ranking:

> For a workload W and target T, if DSE optimizes axis X, how much of the measured/
> trace-derived target gap can X actually close?

## The multi-rate timing budget

A flat capture collapses a VLA action head to a single pass. Reality is multi-rate:

```
backbone once
for k in K:            # denoise / action-head steps
    action_head_step(...)
emit H actions
execute actions at control_rate_hz
```

The budget the guidance reasons about:

```
t_backbone + K * t_head_step  <=  H / control_rate_hz
replan_deadline_ms = 1000 * H / control_rate_hz
```

A flat representation hides `K`, the loop-invariant reuse, and the deadline — which is exactly
why it leads DSE to the wrong axis. The tool builds **both** the flat and the multi-rate
representation and shows the recommendation flip.

## Scoring

```
target_gap            = baseline_total - target_total
intervention_benefit  = baseline_total - intervention_total      (= Σ component reductions)
gap_closure           = intervention_benefit / target_gap
priority_score        = gap_closure * confidence * legality / max(cost_tier, 1)
```

Edge cases are handled explicitly, never papered over: no target → `gap_closure` is null and we
report the benefit as a share of the baseline; `target_gap <= 0` (baseline already meets the
target) → no gap, no invented score; `gap_closure` is clamped to `[0, 1]` for scoring with the
raw value retained.

Each axis declares exactly which baseline cost components its intervention reduces, and a
benefit can never exceed the sum of the components it touches.

## Evidence tags

Every important number carries a source tag. Confidence is a **weight** (it scales the
priority score), not a performance measurement:

| evidence type     | confidence weight | meaning                                        |
|-------------------|-------------------|------------------------------------------------|
| `measured`        | 1.0               | observed on real hardware / runtime (via aet)  |
| `trace_derived`   | 0.8               | derived from an execution trace                |
| `calibrated`      | 0.7               | model calibrated against measurement           |
| `structural_bound`| 0.55              | provable structural bound (e.g. reuse removes repeated traffic) |
| `analytical`      | 0.4               | analytical cost model                          |
| `assumed`         | 0.2               | assumption / default                           |

An axis is only as trustworthy as its softest input: its tag is the **weakest** of its
intervention-model evidence and the evidence of every baseline component it touches. A
structural intervention therefore caps at `structural_bound` even when its inputs are measured.

The cost tier is an ordinal **build-cost proxy** (1 = software-only … 5 = major datapath/memory
redesign), not measured PPA.

## Instrumentation (aet)

Measured / trace-derived evidence comes from the `aet` harness
(`/scratch/agustin/projects/aet`), never hand-coded constants. The adapter
(`merlin.dse_guidance.aet_ingest`) reads an aet run's canonical metrics file
`runs/<suite>/<run_id>/metrics/summary_metrics.json` (flat, dotted keys
`cpu.<regime>.{host_submit_ns,command_encode_ns,sync_wait_ns}`), or an equivalent
`cpu_coupling` YAML. When neither is supplied, the tool reports

```
CPU coupling result unavailable: no measured overhead file provided.
```

and emits no calibration anchor — an anchor without a measurement would be a fabricated number.

## Backbone vs. action head (component-specific residency)

Real VLA timing is `t_backbone_once + K · t_action_head_step`, not `K · t_whole_model`. The
backbone (vision/LM) runs **once** per replan; only the action head repeats K times. Residency
must therefore be charged component-specifically:

- `resident_packed_weights` reduces only the **action-head** weights' repeated packing + weight
  DMA. It is quantified when the baseline supplies a `repeated_head` sub-breakdown (or, for a
  single-region workload, a region-derived reducible fraction). For a whole-model capture that
  does **not** separate head from backbone, it is reported **structurally legal but
  unquantified** — never a fabricated whole-model K-reuse number.
- `resident_prefix_kv` reduces the prefix/KV produced once and reused across the head; quantified
  only when a `loop_invariant.prefix_kv_memory` cost is given.
- `autonomous_K_loop` removes the per-step host launches **inside** the loop — the head's
  repeated dispatch/sync, with the backbone's once-per-replan dispatch excluded.

## Calibration finding (read this before trusting magnitudes)

Calibrating against the real FireSim FASED cycle sweep (6 models, `docs/results.md`) gives a
fitted **≈ 99 cycles/MAC** (median over the 4 parseable, consistent models — tiny_llama, rdt2,
openvla, small_llama), at **MAPE ≈ 32 %**. See
`artifacts/dse-guidance/study_models/cost_calibration.md`. Two honest takeaways:

- A single cycles/MAC constant is a *crude* whole-model predictor — usable as analytical ordering,
  not as a validated magnitude (32 % error, and small_llama is 88 % off because fixed overheads
  dominate a tiny model).
- **xr0 is a 1123× outlier** the matmul-only predictor cannot explain — its capture has 1.3 M MACs
  but the run measured 146 G cycles. This *explains* the earlier "10⁵× off" anomaly: it was an
  xr0 capture/run inconsistency (a partial capture or a non-matmul/repeated-body-dominated run),
  not a uniform model failure.

**Per-component calibration attempt (P1-a).** A multi-feature regression (MACs / activation-bytes /
matmul-count) over the consistent measured points, with leave-one-out CV, shows the features are
collinear (condition number ~10⁹) and multi-feature fits do **not** beat single cycles/MAC under
CV — **per-component coefficients are not identifiable from whole-model totals**. Cycle-exact
per-component calibration needs isolated microbenchmarks (compute-bound matmul, memory/repeated-RHS
matmul, dispatch-heavy tiny-kernel sequence, `matmul_bias_requant_relu`, `no_reuse_matmul`) measured
on the chipyard/spike or FireSim toolchain — which is **unavailable in this environment**
(`MERLIN_CHIPYARD` unset, `spike` not on PATH), so it is the precise scoped remaining measurement,
not a fabricated coefficient. See `cost_calibration.md`.

**Consequence:** cross-workload `gap_closure` *magnitudes* remain analytical (the per-component
cost model is still uncalibrated; the fits above are whole-model sanity anchors, not a per-component
model). What stands on its own is the **structural / legality** result, the *ordering* within a
single baseline, the **measured dispatch count + host-dispatch-bound finding**, and the numerical
contract. Evidence tags make the distinction visible.

## Measured dispatch coupling (the one measured runtime leg)

Running real small captures through the host reference executor
(`merlin.runtime.dispatch_runtime.run_model`, cos=1.0 vs the torch golden) **measures** the actual
dispatch count per forward: 183–213 dispatches vs a 15-matmul estimate — **the matmul proxy
under-counts real dispatch granularity by ~13×** (real dispatches include every
elementwise/norm/view/glue kernel). This grounds the `dispatches per replan` input to
`command_batching` / `autonomous_K_loop` in a *measured* number instead of an estimate; the
opportunity is larger than the matmul-only view implied. Data:
`merlin/benchmarks/dse_guidance/measured_dispatch.yaml`; report:
`benchmarks/dse_guidance/case_study/dispatch_coupling_report.md`. Honest scope: the dispatch
*count* is measured; per-dispatch *host cost* is host-interpreter timing (Python reference
executor), not the deployable C runtime — so no speedup is claimed.

## What is and isn't measured today (honest status)

| Quantity | Status |
|---|---|
| Which axes flat hides (legality flip) | structural — robust, definitional given the K-loop |
| FASED cycle totals (6 models) | **measured** (FireSim, `results.md`) |
| cycles/MAC fit (≈99, MAPE 32%) | **calibrated** — crude whole-model anchor; xr0 a 1123× outlier |
| Per-model cost components | analytical (placeholder constants, uncalibrated) |
| Region roles (backbone/head) | **recovered from `prov.fqn`** for freshly-captured models; operator-mapped or `unknown` for pre-`prov.fqn` captures |
| Loop counts K | assumed / reference (architecture values, overridable) |
| Dispatch count per forward | **measured** (host reference executor, cos=1.0) — ~13× the matmul estimate |
| Per-dispatch host submit/sync ns | host-interpreter proxy only; deployable-runtime coupling still **unavailable** |
| smolvla per-component head split | illustrative — exercises the component-attributed path; not measured |

## What not to claim

- Do **not** claim global optimality — this is guidance, not a solver.
- Do **not** claim a validated cost model unless the numbers are `measured`/`calibrated`.
- Do **not** use invented constants as measurements. Synthesized baselines are tagged
  `analytical` and weighted accordingly.

## Commands

Single workload (headline VLA action head, with measured coupling from an aet run):

```bash
merlin-dse-guidance \
  --temporal-metadata merlin/python/tests/fixtures/dse_guidance/smolvla_action_head_temporal.yaml \
  --baseline-cost     merlin/python/tests/fixtures/dse_guidance/smolvla_action_head_cost.yaml \
  --aet-run           merlin/python/tests/fixtures/dse_guidance/aet_run \
  --out artifacts/dse-guidance/smolvla_action_head/
```

Exhaustive study across the design-pressure `semantic_memory` regions (synthesizes an
analytical baseline + temporal view from each region's reuse):

```bash
merlin-dse-guidance --study --out artifacts/dse-guidance/study/
```

Cross-workload provenance case study over the **real `prov.fqn` recaptures** (the breadth
result — rdt, openvla, small_llama, tiny_llama; not overfit to one):

```bash
merlin-dse-guidance --case-study --out artifacts/dse-guidance/case_study/
```

Reads `merlin/benchmarks/dse_guidance/recaptures/<workload>/model.mlir` (real architectures via
the `prov.fqn`-enabled `m2m`, small/random configs — weights not committed), auto-recovers roles
from `prov.fqn`, and emits `case_study.md` + `cross_workload_provenance.csv` (per-item
flat-vs-recovered with explicit evidence labels: `recovered_from_ir` / `recovered_from_prov_fqn` /
`assumed_reference` / `calibrated` / `uncalibrated` / `unavailable`). OpenVLA's vision-backbone vs
LM-decode split is recovered; every candidate stays `blocked_by: missing_calibration`. A committed
copy lives at `merlin/benchmarks/dse_guidance/case_study/`.

Exhaustive study across the **real model zoo** — every captured workload under
`artifacts/recaptures/<model>_<dtype>_consistent/model.mlir` (smolvla, openvla, pi05, rdt/rdt2, groot,
molmoact, bitvla, xr0, the llama LMs):

```bash
merlin-dse-guidance --study --models --out artifacts/dse-guidance/study_models/
```

This is the headline demonstration. `docs/results.md` records that **whole-model captures use
each weight once — they emit 0 contract facts**: the capture is flat and hides the host-side
decode/denoise loop. The model study reads each `model.mlir` for aggregate structural facts
(matmul count, MACs, weight vs activation bytes — `analytical`), applies the architecture's
loop count K (a reference value, tagged `assumed`, overridable), anchors xr0's total to its
measured FireSim cycles (`146.2 G`, `measured`/`calibrated`), and shows that residency and the
autonomous K-loop **become legal only under the multi-rate view** — for every model, including
the autoregressive ones (they reuse weights across action-token decode). Captures that do not
parse with stock xDSL still show the structural legality flip (magnitudes reported as `n/a`,
parse status surfaced — never fabricated).

Residency benefit is grounded: it always claims the repeated **packing** it can prove, and
claims a slice of **DMA** only when there is a region-derived reducible fraction or an explicit
`weight_memory` component. It never assumes all DMA is reducible weight traffic.

Structural-only run (no baseline needed — recover contract + candidates for any workload):

```bash
merlin-dse-guidance --temporal-metadata <topology.yaml> --structural-only --out <dir>/
```

### Output

```
artifacts/dse-guidance/<workload>/
  # structural front-end (always)
  vla_runtime_topology.yaml
  capture_fidelity_report.md / capture_fidelity.yaml
  dse_candidate_axes.md / dse_candidate_axes.yaml
  flat_report.yaml
  multirate_report.yaml
  flat_vs_multirate_diff.csv
  # quantitative back-end (only with a baseline cost; uncalibrated by default)
  axis_triage.csv          # multi-rate
  axis_triage_flat.csv     # flat
  axis_triage.md           # ranking + why the representation matters
  bottleneck_breakdown.csv
  deadline_feasibility.yaml
  cpu_coupling_result.yaml | cpu_coupling_result.txt   # measured, or the unavailable message
  calibration_anchor.csv   # only when a measurement was supplied
  negative_control_report.md   # for no-reuse (K=1) workloads
  figures/                 # optional (matplotlib): axis_triage / bottleneck / flat_vs_multirate
artifacts/dse-guidance/study/         # --study (semantic_memory regions)
artifacts/dse-guidance/study_models/  # --study --models (real model zoo)
  study_summary.csv
  study_summary.md         # per-workload top axis + cross-workload axis ranking + representation flips
  <workload>/...
```
