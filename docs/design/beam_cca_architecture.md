---
title: "Design: the CCA beam — two CCAs, cross-framework analysis, and autonomous whole-model improvement"
kind: design
status: current
owner: rvvgen
last_verified: 2026-07-20
related: [beam_search, expert_gap_attribution, whole_model_op_profile, transpose_fusion, kernel_mining]
code_refs:
  - merlin/python/merlin/kernels/cca.py
  - merlin/python/merlin/kernels/cca_contract.py
  - merlin/python/merlin/kernels/action_catalog.py
  - merlin/python/merlin/kernels/trace.py
  - merlin/python/merlin/mining/beam.py
  - merlin/python/merlin/mining/wholemodel_proposer.py
  - merlin/python/merlin/llvmlower/op_profile.py
  - build_tools/scripts/run_autonomous_beam_experiment.py
---

# The CCA beam: learning whole-model compiler improvements from experts, autonomously

This is the deep reference for four things the project keeps coming back to: **how the beam works,
how the CCA works (both of them), how a proposed change becomes a real compiler edit, and how we
analyze other frameworks / expert kernels**. The step-by-step how-to lives in
[beam_search.md](../guides/beam_search.md); the reproduction recipe in
[reproducibility.md](../guides/reproducibility.md). This doc is the *why* and the *architecture*.

## The one-paragraph model

The compiler is improved by a search that **learns from experts and is judged on real hardware**.
An expert artifact (an XNNPACK RVV kernel, an ExecuTorch graph, our own hand baseline) is reduced to
a target-agnostic **Common Compute Abstraction (CCA)**. The compiler's *own* emitted code is reduced
to a CCA the same way. Where the two disagree is a **divergence**; each divergence routes through the
**action catalog** to a concrete, default-OFF compiler edit; the beam enables edits, measures the
result on the K1 board, gates correctness fail-closed, and keeps what is measurably faster —
**end-to-end on the whole model**, not on an isolated kernel. Frozen `hand_v0` is the control it forks
from and never mutates.

## There are TWO CCAs (this is the crux)

The CCA (`kernels/cca.py`) is one vocabulary — target-neutral facets naming *what a computation
decides*, not how a backend spells it: `ComputeFacet` (contraction_form, register_block,
accumulator_resident, reduction_form, widening, epilogue, activation_vectorization), `VectorFacet`
(sew, lmul, vl_strategy, tail), `MemoryFacet` (access_pattern, panel_reuse), `EnvelopeFacet` (the code
*around* the loop — calls_in_loop, runtime_calls), plus `Spatial`/`Dataflow` stubs for other targets.

The same vocabulary is lifted from two sources, and keeping them distinct is what makes the loop
meaningful:

- **Expert / kernel CCA** — lifted from an *expert's* assembly (`cca.lift_asm` over
  `decode.rvv.decode_text(objdump)`). This is the *target shape*: what a good kernel decided. It is
  deterministic — **no LLM authors a CCA**. Per-dtype expert fixtures live in
  `merlin/tests/data/cca_asm/` (f32 / int8-qd8 / f16, the last carrying a native-accumulate caveat).
- **Compiler CCA** — lifted the same way from *our own* emitted object (`generated/objdump.txt`,
  disassembled from the fork's `model.o`). This is what our compiler *actually* produced — read from
  the emitted code, never assumed from the schedule text. (Why "actually": two levers, `KC` and
  `MR`-under-`unroll_m`, were once wired end-to-end yet emitted identical code — see
  [expert_gap_attribution.md](expert_gap_attribution.md). A CCA lifted from emitted code catches that;
  a schedule-text diff does not.)

A **divergence** is a facet where expert-CCA ≠ compiler-CCA (`cca_compare.compare`, which reflects the
facet list from the dataclass so a new facet is never silently skipped).

### The bijection contract

`cca_contract.check_bijection('rvv')` enforces that **every LEVER facet has a route** and every route
maps to a real facet — a ratchet (`FIELD_REGISTRY` classifies each facet.field IDENTITY / LEVER /
METRIC / BACKEND_STUB; `KNOWN_OPEN` allowlists the not-yet-closed gaps). This is what stops the CCA
from growing a vocabulary the compiler can't act on, or a lever the search can't reach. Closing an
orphan (e.g. `compute.reduction_form`) means: add the facet inferer, add a default-OFF
feature that emits it, add the route, remove it from `KNOWN_OPEN`.

## How we analyze other frameworks / expert kernels

Two complementary axes, both deterministic:

1. **Kernel axis (asm → CCA).** Decode the expert kernel's objdump (`decode.rvv`, which tracks the
   effective vtype so `sew`/`lmul`/`vl_strategy` are read, not guessed), lift its CCA, and diff. The
   XNNPACK RVV surface is enumerated by `xnnpack_kernel_catalog.py` (637 kernels / 115 families) and
   ranked for relevance by `model_op_census.py` — **by BYTE-TRAFFIC, not flops**, because the models
   are memory-bound (transpose is 38% of traffic vs matmul's 26%, while matmul is 82% of *flops*).
   `kernel_coverage_matrix.py` joins census-relevance × catalog-status × measured board ratio and
   records the honest gaps (softmax-fused, gather, transcendentals) where no expert primitive or no
   Merlin lever exists.

2. **Whole-model axis (graph → IR → asm → cycles).** A model is more than its kernels. The
   `LoweringTrace` (`kernels/trace.py`) threads *flattened graph → transform steps → emitted code*,
   and the per-op profiler (`llvmlower/op_profile.py`, driver `k1_op_profile.py`) attributes measured
   board ticks to each op, **keyed on `prov.fqn`** — the cross-compiler join that also aligns a Merlin
   region with an ExecuTorch node (`baselines/_et_export.py`). That join is what makes the analysis
   *framework-agnostic*: a graph node from any frontend (ExecuTorch/GGUF/ONNX/PT) maps to the IR that
   generates its compute, to the emitted asm, to the cycles it costs. It is how we found that a scalar
   `linalg.transpose` was 57% of openvla's runtime — invisible to any kernel-level view.

## How a change becomes a real compiler edit

`kernels/action_catalog.py` routes each divergence to a `CompilerAction` on an escalation ladder
**FLAG → KNOB → HEURISTIC → PASS → CODEGEN** (cheapest realization first; `route_escalated` walks up
when a fork's emitted CCA did not achieve the promised facet). Every routed edit is a **default-OFF
compiler feature** (`llvmlower/impr_features.py`) so frozen `hand_v0` stays byte-identical; the beam
enables features per fork via `compiler_features`. Current forkable whole-model levers and their
routes:

| divergence axis | action | feature | measured whole-model effect |
|---|---|---|---|
| `layout.transpose_materialized` | PASS | `fuse_transpose_b` | fold transpose into matmul indexing map; −6.5% openvla |
| `compute.register_block` | KNOB→PASS | `accumulator_resident_wholemodel_vf_mrpad` | per-matmul MR + M-tail pad; 1.49× rdt2 matmul bucket |
| `envelope.runtime_calls` | PASS | `erase_self_copy` | drop the per-tile `memref.copy %x,%x`; 1.88× f32 GEMM |
| `compute.reduction_form` | PASS | `vectorize_reduction` | vectorize reduce → `vredsum`/`vfredusum` |
| `compute.activation_vectorization` | PASS | `vectorized_transcendental_activation` | poly-vectorize gelu/sigmoid/silu |

Composition rule: at most ONE full-schedule-replacement feature (the register-block recipe) plus any
number of additive passes (transpose fusion, self-copy erase, reduction, activation). The
`CompositionError` guard enforces it; the whole-model proposer respects it when stacking.

## The beam engine

`mining/beam.run_beam` — generation by generation: propose → mint fork (`fork_from_action` /
`wholemodel_proposer`) → certify (`runner.certify_rvv`) → rank → keep top-k → next depth (which
*stacks* another lever onto a survivor).

- **Objective = whole-model.** `model_dir` is a whole-model bundle, so the ranked `speedup` is the
  model's real K1 wall vs the seed's, and `attainment_vs_expert = xnn_wall / fork_wall` (≥1 beats
  XNNPACK). Kernel-only ranking is *anti-correlated* with e2e (the `ours_v3` trap: best on a 128³
  GEMM, 12× worse whole-model), which is why the objective is the model.
- **Two proposers.** `propose_forks_from_cca` (kernel-CCA-divergence router — GEMM-level levers) and
  `wholemodel_proposer.propose_wholemodel_levers` (byte-traffic-ranked whole-model levers — the one
  that surfaces graph-level moves like transpose fusion a single-kernel diff is blind to). Select with
  `beam_cli --proposer {wholemodel,cca}` (default `wholemodel`).
- **Two-phase (optional).** Explore on a fast bundle (bitvla), re-certify survivors on a full/slow
  bundle (`--validate-model-dir`) before promotion, so exploration is cheap and promotion is honest.
- **Correctness gate** (`zephyr_model._gate`, the single K3/K5 chokepoint). Tiers: T1 int8/w8a8
  (cos>0.999 + rel + per-element), T2 classification (cos + argmax + per-element), legacy bit-close
  (cos>0.9999 + rel<1e-3 + per-element), **T3 whole-model fp32 regression (cos≥0.9999, cosine-only)**.
  T3 is cosine-only *by evidence*: a whole-model regression output legitimately carries high
  per-element relative error on its many small elements (bitvla measured per-element max-rel 1.1 at
  cos 0.9999945), so a per-element ceiling false-rejects it; the four-way authority gates these on
  cosine alone. The per-element veto is retained where outputs are well-scaled (int8/classification/
  bit-close) and in the dtype drivers.
- **Noise + inert discipline.** The K1 noise floor is ≥1.9% (≥4.3% contended). A fork counts as a win
  only if its speedup exceeds the parent by more than a margin (`--noise-margin`, default 0.02); a
  fork whose emitted mnemonic digest equals its parent's is flagged `inert` and cannot be promoted.

## Autonomous experiment

`build_tools/scripts/run_autonomous_beam_experiment.py` runs the whole thing with no hand-feeding:
for each `(dtype, model)` cell it seeds from frozen `hand_v0`, runs the beam (whole-model proposer,
depth ≥2 to discover a *stack*), and writes what the beam **discovered** (which levers, final speedup,
attainment vs XNNPACK) beside the **manual** `ours_best` and the experts — the fully-autonomous
reproduction of the campaign's e2e gains. Validated fp32 result (K1, depth-2): the beam independently
reached bitvla 1.244× / openvla 0.985× / rdt2 1.385× of XNNPACK, discovering per-matmul MR + self-copy
erase (+ transpose fusion on rdt2) on its own — matching or exceeding the hand-tuned stack.

## What the loop still cannot do (honest gaps)

- Whole-model int8/fp16 have no XNNPACK e2e column (the kernel-backend router is an f32 GEMM path), so
  those cells compare beam-vs-baseline-vs-manual, not vs an XNNPACK whole-model wall.
- `vectorize_reduction` gate-failed on bitvla in the autonomous run (a real gap under diagnosis).
- Graph-level levers only exist for what the census surfaces + the action catalog routes; the honest
  gaps (softmax-fused, gather/embedding, sin/cos/RoPE) have no lever yet — tracked in the coverage
  matrix, never silently treated as covered.
