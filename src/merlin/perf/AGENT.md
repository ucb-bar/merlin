# AGENT.md — src/merlin/perf

## Purpose

The performance layer: what a target's legal choices *cost*. Derives archetypes and traits, emits a
performance contract whose terms carry provenance and a validity domain, composes a predicted cycle
count with gap attribution, and classifies workloads by which lever has headroom.

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
