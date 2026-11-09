---
title: Independent source-only coverage rosters
kind: design
status: current
owner: experiments
last_verified: 2026-11-09
related: [component_scale_generalization, component_execution_budget, component_graph_variants]
code_refs:
  - src/merlin_experiments/phase0/component_compile_sources.py
  - src/merlin_experiments/phase0/component_compile_plan.py
  - src/merlin/targetgen/component_program.py
  - src/merlin/targetgen/contract/compile_only.py
  - src/merlin_experiments/phase0/component_compile_graphs.py
---

# Independent source-only coverage rosters

Large type and footprint cases require a separate source-only roster. They do
not enter the numerical v2 execution roster, its data/golden builders, or its
bounded execution acceptance. The Phase 0 producer emits original standard
MLIR and reopens its complete ordered tensor signature. The ordinary Phase 1
transport separately lowers, builds objects, links and admits instructions;
neither source readiness nor successful linkage discharges target static or
numerical obligations.

## Protected selection and generation

`issue_independent_compile_only_roster` requires the exact live independently
issued hardware and minimal software authorities, their identical shared
hardware origin, the original pinned descriptor, and an explicit external plan.
`merlin.component_compile_only_plan.v1` is closed. It admits independent
preauthor provenance, exact intake hashes, an explicit metadata budget, and
complete public guard/private transfer members. Model names, validation inputs,
reference compiler code, schedules, timings and optimization history have no
field. Selection occurs at the protected coordinator boundary; provenance labels
and saved JSON do not independently establish authority or historical origin.

Each member declares its immutable name, cohort, separate `compile_only` or
`static_refusal` expectation, reviewed operation owner, original dimensions and
required static obligations. The version-1 fixed original frontend supports
positive rank-two integer copy and contraction sources. It admits explicit
`integer_reference` bounded-exact or modular numerical semantics; values and
numerical input domains are not chosen or proved by this path. Saturating,
floating point, zero-sized, other-rank and other-operation source frontends
remain unsupported here. Missing required cases block complete issuance.

The target-neutral `component_program.render` emits the original source using
only the checked generic DAG and selected scalar types. It does not construct a
corpus binding or invent tile geometry, instruction classes, runtime tiers or
target placement. Existing ordinary numerical `build` delegates to this same
source emitter and retains its original capsule and numerical behavior.
Contraction initialization stays symbolic scalar constant plus `linalg.fill`;
large tensor types never expand dense zero data just to read an ABI.

## Extents and budgets

The external plan chooses explicit positive integer extents or boundaries from
fresh structural memory facts. `memory_volume` derives
`floor(memory_bytes * 8 / (selected_dtype_bits * fixed_elements)) + offset`.
`memory_depth` derives `observed_depth + offset`. Memory ordinal and each
expression operand are explicit; absent or nonpositive facts refuse.

These formulas select source type stress cases. A memory depth is not an
observed CPU address width, and a tensor volume is not an allocation schedule.
No layout, padding, simultaneous residence, streaming, ISA field width or
hardware arithmetic is inferred. Tests around the boundary must still acquire
their independent indexing, resource and streaming evidence.

The policy explicitly bounds member count, aggregate actual UTF-8 source bytes,
extent bit length and scalar bit width. Unknown source costs or semantics
refuse. Source byte and metadata budgets do not establish compiler time, Python
heap, device memory or device cycles. Source construction work is independent
of logical tensor volume for these fixed primitives.

## Live admission and missing obligations

The complete private report retains every original mandatory member and each
declared obligation. Supported sources are `source_ready`; failed sources are
`source_unavailable`. All static obligations are `unknown`. Over-budget cases
are never relabeled numerical or compile passes. A failure writes the original
denominator and raises `CompileOnlyRosterRefusal`; it issues no live partial
roster. `static_refusal` means a separately selected candidate grading
expectation, not a source producer's refusal proof.

The live `IndependentCompileOnlyRoster` pins the actual selected plan,
descriptor, original emitted source files, report and reader sources. Verification
reopens all pins and re-renders every source, re-parses every original ABI,
and rechecks complete membership and aggregate budget. The complete source
must match, including ordered input/output type slots. Reconstructed dataclass
objects and saved or re-signed records cannot mint this live authority.

`public_summary` exposes only identities and cohort counts. Withheld member
names, dimensions and source paths stay in the private roster. Public source
grants and private grading access are selected and frozen by the ordinary phase
boundary, not by a candidate or this report reader.

The remaining mandatory proof owners must establish semantic coverage, original
input numeric domains, target index/encoded fields, resource and streaming
legality, full output iteration/store coverage, and physical ownership and
completion. Compilation, correctness, timing and whole-domain generalization
stay separate. A finite source roster cannot guarantee arbitrary whole-model
composition or numerical error behavior.

Version 2 of the source-only plan additionally admits explicit
`component_program` graph families. `component_compile_graphs.py` expands each
selected family into its original paired representations, requires independently
selected `matmul`, `copy` and `add` operation owners, and bounds node/output
metadata before constructing topology. The resulting sources still establish
no numerical, physical-reuse, ordering or performance claim. Version 1 retains
its original primitive-only meaning.
