---
title: Inspect ordinary per-capsule performance feedback
kind: guide
status: current
owner: experiments
last_verified: 2026-10-19
related: [phase2_corpus_search, normal_search_feedback, spike_timing]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_capsule_feedback.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/candidate_execution_products.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/compiler_profile_emission.py
  - packages/merlin-experiments/tests/test_ordinary_capsule_feedback.py
  - src/merlin/perf/capsule_roofline.py
  - src/merlin/perf/candidate_diagnostics.py
  - packages/merlin-timing/src/merlin_timing/observations.py
---

# Ordinary per-capsule feedback

`merlin_experiments.phase2.ordinary_capsule_feedback.join_capsule_feedback` joins
already observed public data for exploration. It runs no compiler or simulator,
reads no files and issues no qualification. The ordinary owner must reopen the
original source, candidate, numerical and count products before calling it.

```python
feedback = join_capsule_feedback(
    member.descriptor,
    compiled_observation,
    raw_report=complete_original_count_report,  # None retains UNKNOWN
    prediction=existing_empirical_comparison_case,
    target=selected_target,
    facts=selected_facts,
)
```

The result retains original useful MAC demand from
`compiled.candidate_diagnostics.candidate_features.features.source_macs`, typed
source inputs, static emitted diagnostics, placement and representation
declarations, host/packing scope, complete dynamic activity and the existing
screen's correlated terms. Count-v2 coverage and the original observed ELF must
match. Native timing taint stays separate from count completeness. A missing,
malformed or incomplete count report remains UNKNOWN. Counts do not supply
target cycles, physical traffic, array utilization or completion semantics.

The join reuses `capsule_roofline.machine_bounds` and `capsule_roofline.roofline`.
Useful source MACs remain distinct from padded issued MAC requests. Source
logical payload and requested DMA bytes never become compulsory physical
traffic. The structural compute bound is conditional on the selected arithmetic
inventory describing the executed source work. Complete-call host costs,
packing/layout materialization, physical ports/transfer granularity, placement
and resource overlap remain unresolved. No floor from this join can establish
convergence.

An empirical estimate or interval never supplies measured cycles, gap or
attainment. Coefficients remain correlated screening estimates rather than
physical latency or stall prices. `estimated_cycles` in the raw native event
report is ignored, even when present.

An optional original `measurement` permits arithmetic over independently
admitted measured rows. The original caller must first reopen its actual
measurement receipt. It supplies `kind="measured_target_cycles"`,
`measured_now=True`, `functional_status="PASS"`, the exact compiled
`elf_sha256`, `complete_source_scope=True` and finite positive `cycles`.
Its `scope_sha256` and `device_sha256` must match the explicit selected `machine`
document. These data checks issue no measurement or hardware authority.
A comparable measurement below the conditional floor refutes that floor;
negative gaps are retained and attainment is unavailable. Other comparable
measured rows may expose partial-floor attainment, with convergence still
disabled. Paired arms and every repeat retain their separate observations.

Supplying `facts` prevents an explicit `machine` argument from replacing the
arithmetic rate derived from those facts. Geometry without explicit arithmetic
counts remains UNKNOWN. No ambient target facts are loaded when inputs are
absent. The output reuses the existing bounded public diagnostic projection;
raw artifact payloads and private paths are not added.

For an original public or development layer grade with retained execution-v2
products, the ordinary observation adapter reopens the exact original roles.
It joins the compiler snapshot, source, invocation, completed command buffer,
LLVM, object, ELF and the selected functional engine. Source MACs therefore come
from the source actually consumed by that execution. A repeated basename cannot
select another command buffer. The bounded reader refuses ambiguous producers,
changed bytes, unsupported namespaces and incomplete original functional roles;
feedback then retains UNKNOWN. Private grades cannot fall back to public
emission-only discovery. Older public emission-only grades retain their reader.

The separate host-only `compiler_profile_emission.emit_analysis` diagnostic can
observe the original compiler-profile execution while its temporary input and
output still exist. Its issued live receipt binds those bytes to the original
executor, package, selected library and invocation. It can rejoin native inputs
only through that live owner; saved JSON cannot restore the receipt. This is a
producer check. Source arithmetic, instruction effects, hardware execution,
timing, final guards and mandatory grading tiers keep their independent owners.
Neither this diagnostic nor the product reader adds a candidate import grant or
qualifies an ordinary search queue by itself.
