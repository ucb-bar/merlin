# AGENT.md — src/merlin/perf

## Purpose

The performance layer: what a target's legal choices *cost*. Derives archetypes and traits, emits a
performance contract whose terms carry provenance and a validity domain, composes a predicted cycle
count with gap attribution, and classifies workloads by which lever has headroom.

`component_applicability` holds pure frozen joint semantic cells for measured
payload/resource, tile/tail, streaming, dependency, reuse and composition scope.
Exact membership cannot imply range or repetition transfer. These declarations
and complete-cost arithmetic do not issue hardware/timer authority; experiments
owns source/runtime provenance, actual controls and held qualification.

`component_measurement_plan` retains the complete development roster and selects
bounded existing distinguishing measurements for overlapping intervals. Disjoint
intervals may defer development measurement only; unsupported or new joint cells
require independent requalification. Arithmetic owner/status inputs do not issue
qualification. Correctness, held and final rosters cannot select this pruning.

`component_source_demand` derives bounded logical payload, work, dependency and
last-needed intervals from the original checked functionalized tensor program
under an explicit complete topological schedule. Alias relations share logical
values; updates retain fresh SSA epochs and complete publication. No shaped
data or references are allocated. Eager logical payload and operand extents are
not physical allocations, bus traffic, capacity bounds or timing/ranking facts.
Unknown grammar and arithmetic limits remain explicit before analysis expansion.

`compiled_static_features` reuses the complete typed emitted LLVM observer and
structural ELF reader for bounded site/access-width histograms and linked code
extents. Counts are before native optimization. They are neither executed
instruction counts nor physical traffic, staging, capacity or cycle features.
Unsupported bodies refuse; experiments owns actual compile-consumption joins.

`component_measurement_stream` checks one bounded, explicitly ordered raw event
frame containing every original complete-cost stage. Counter wrap and control
changes are retained. Event labels do not establish actual stage boundaries;
there is no elapsed-cost subtraction, clock unit or cold/warm inference. Stage,
timer, observer, resource, runtime and held qualification remain independent.

The namespace is shared without duplicate implementations: experiments owns
the isolated/controlled/paired probe providers, host-region/physical-transition/lane-migration
qualifiers, source contraction/convolution preparation, source-program-pair binding and execution,
initializer-elision admission, and `analysis_worker` (bounded experiment workers that load
the installed Phase 2 emission analyzer). These owners consume concrete experiment revisions
and execution capabilities; they are not core primitives. Analysis owns
`recovery` (known-answer benchmark scoring and `merlin-recovery`). Keep pure cost/evidence
primitives here; sandbox capability enforcement stays with experiments. Audit classification
is shared policy in `merlin.common.access`, independent of either optional distribution.
Compiler edit-surface vocabulary is read from the selected manifest schema during
`inspect_compiler_package`, never during import. Its explicit `contract` argument
must not fall back to checkout data when a selected schema is missing or invalid.

## The rule that governs everything here

Every module is a **generic, trait-gated analysis**. Gating is on **derived traits**, never on an
archetype name and never on a target name — an archetype is only a prior (it decides *which questions
to ask*); the RTL-derived traits decide *which of them apply*. The target is a parameter (`target=`)
threaded from the descriptor/manifest.

A tool that only works where it was written is manual overfitting with extra steps. The bar: the same
code runs on two targets of different archetypes and produces **different, correct** answers.

## What does not belong here

- Target-name literals of any kind. `check_no_target_name.py` scans this tree, and its advisory
  `--coupling` pass also flags imports and substring symbol hits in any file whose own path does not
  name a target — which `perf/` never will. **Do not add allowlist entries**; that list only shrinks.
- `import re`. Parse structurally.
- Assumed geometry, capacities, opcodes or latencies. Derive them, or record `UNKNOWN` and fail
  closed. `UNKNOWN` is a distinct inhabited state and must never be readable as `0.0`.
- Generated output. Products go to `out/artifacts/` via `merlin.common.paths` / `artifacts` helpers.

## Invariants

- **Never default a composition operator.** Textbook roofline takes `max` of compute and memory time,
  which assumes perfect overlap; a target that does not overlap sums them instead. Deriving `max` where
  the truth is `sum` understates runtime badly, and in the flattering direction.
- **Prefer moved bytes to algorithmic bytes.** Transfer amplification is real and large; a bound built
  on the bytes an algorithm needs rather than the bytes a program moves is optimistic by that factor.
- **Fixed terms are first-class.** Pipeline fill and drain are intercepts, not rates. A rate-only model
  mispredicts every small workload.
- **At least two points per fitted parameter.** A single rate cannot price a unit whose cost is a rate
  plus a fixed overhead.
- **A partial whole-model build is never a whole model.** `only_groups` (`whole_model_partial`) marks
  the record, the oracle and the expectations. The verdict, the grade, the gate and the measured
  service each refuse the marker; a new whole-model reader must call `whole_model_partial.refuse`.
- Build stages are compile-trace stages (`whole_model_build.BUILD_STAGES`, the stage clock's own
  vocabulary); the builder CLI lives in `whole_model_build_cli` so the builder module holds the build.
- Tests go in an existing bucket — there is no `perf` bucket and the list is an enum. Contract, record
  and profile tests live in `merlin/tests/targetgen/`; envelope, attribution and analysis tests in
  `merlin/tests/dse/`. Resolve paths via `merlin.common.paths.repo_root()`.

## Where the task register lives

`merlin/experiments/performance_contract/TASKS.md` — every task, its state, and its blocker.
Rationale for the cost and oracle decisions: `docs/design/performance_budget_unit.md`.

`debug_companion` admits only byte/address/attribute-identical allocated ELF
sections and normalized relocations before mapping an existing PC census.
Target tools, compiler recipes, source identities and numerical/ISA gates remain
caller-owned. Every address is retained once, including missing source metadata;
inline frames preserve context and their counts overlap parent call-site totals.
Instruction counts never imply hardware cycles. The ELF reader explicitly
refuses unsupported formats rather than guessing them. Tests live in the DSE
bucket and exercise actual compiler/symbolizer twins plus changed-byte refusals.
Its production caller is `merlin experiment inspect --trace` (the group build's `debug_companion`).

`address_locality` counts first touches and exact distinct-intervening-region
recurrence distances from explicitly ordered requested addresses. Granule and
resource budget are caller inputs; capacity thresholds exclude first touches.
These are logical locality features, not physical traffic or timing estimates.
Its production callers are `merlin experiment inspect --group gN --trace`, which records the
group program's committed memory requests from the functional model's commit log
(`requested_addresses.json`) and censuses them at each `--locality-granule`, and
`merlin experiment census locality` (`census_cli`), which re-takes the census from that file and
refuses a trace longer than the stated budget.
`fast_estimate_validation` is called by `whole_model_screen.fit_calibration`, which validates every
refit of the structure screen's calibration against held-out board-measured groups under the board's
derived noise margin and records the screen's ranking as `unvalidated` whenever that does not hold.


`execution_boundaries` summarizes explicit provider-decoded instruction extents,
execution counts, call classifications, stack access widths and frame facts.
Every instruction is covered once; unknown facts remain unknown. Entry-normalized
counts and repeated stack sites are descriptive features, not physical traffic,
peak stack usage, interprocedural dependencies or cycle prices. Its training-only
boundary envelope checks an unpriced subdomain and never approves a ranking.
The provider owns decoding/ABI and exact artifact verification; the existing
fitter and held-out ordering gate remain separate. Independent expanded event
traces and multiple provider geometries test the summary and refusal behavior.
Its production caller is `merlin experiment census boundaries` / `boundary-domain`
(`census_cli`), which reads the provider's records and digests from disk. No provider writes those
records yet: call kinds, callees and stack widths are host-ABI decoding a target's support owns, and
no vendored provider declares a decoder for them. A provider that adds one is the producer.

`schedule_proxy` prices what a schedule asks the device for in array-tile transactions (compute tiles
plus moved bytes, never one without the other). It is validated as an ORDER against 71 FireSim-measured
groups (`merlin/tests/dse/test_schedule_proxy.py`, points in `examples/<target>/phase2/`) and is the
`cost_proxy` rung of `merlin/contract/measurement_ladder.yaml`: ranking only, never a cycle count.
Its production caller is `group_headroom.group_rank`, reported beside each stated group's bound.

`load_state_residency`/`load_state_claim`, `stationary_residency`/`stationary_claim` and
`movein_placement`/`movein_claim` decide the template's EMITS families (PD, PA, PJ) from a
candidate's own decoded stream. Every bound is derived through the selected support (the
load-configuration layout and retain sentinel via `targetgen.rocc.decode`) or the target's RTL
facts, and an underivable one REFUSES. Callers: `trace_check`'s declared residency modes and
`merlin_experiments.phase2.claims.dispatch`. The measured-claims coordinator refuses to seal them:
they read no cycles.
