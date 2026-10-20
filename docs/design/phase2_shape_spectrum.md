---
title: Development shape spectra and reduced-depth compilation tests
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-18
related: [phase2_capacity_cohort, ../guides/phase2_test_justification, ../guides/shared_pass_phase2]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/shape_spectrum.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_coverage_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/public_boundaries.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/capacity_families.py
---

# Development shape spectra and reduced-depth compilation tests

Phase 2 needs a compact distribution of development computations, hardware
boundary probes and connected compilation tests. Requiring an exact copy of
every headline-model layer is neither necessary nor a generalization test.
Matching one M/K/N tuple alone does not establish layout, arithmetic, placement,
state or performance coverage.

## Data boundary

Select public development source metadata before search. Validation executions,
reference outputs, hidden inputs and candidate timings cannot select training
members. Protected validation metadata stays in an operator-only coverage audit;
it must not become an author-visible corpus or input manifest. Existing private
exclusion and ordinary Phase 0 admission still apply to every generated member.

If architecture distributions are explicitly disclosed for development, record
that choice as distribution-informed development. Those distributions and known
geometries cannot also be described as unseen transfer evidence. A newly reviewed
aggregate input requires a separate explicit selection; the public-summary scope
string is a declaration, not proof of disclosure or admission. Current planning
accepts only explicitly selected public development summaries. It refuses
validation, protected and unspecified aggregate scopes.

## A bounded shape plan

`shape_spectrum.plan` consumes `merlin.phase0.public_shape_summary.v1`:

```json
{
  "schema": "merlin.phase0.public_shape_summary.v1",
  "scope": "public_development",
  "source_sha256": ["<original source metadata SHA256>"],
  "members": [{
    "family": "contraction",
    "contract": "<semantic, layout and readout identity>",
    "shape": {"M": 7, "K": 60, "N": 240},
    "occurrences": 1
  }]
}
```

Each shape holds independent source parameters with their equality constraints
already resolved. Batch/head relations, convolution stride/padding and readout
contracts must remain distinct semantic families; flattening them to a common
matrix volume cannot grant equivalence. Resource granules and their selected
identity are explicit inputs, never assumed target constants.

The new versioned planner:

1. Partitions semantic/layout/readout contracts and axis signatures.
2. Keeps GEMV/short-M, tall, wide, balanced and deep-K morphology distinct.
3. Forms coarse logarithmic logical-size bins. Default width is three octaves
   in the product of independent dimensions; this is sampling policy, not a
   latency or capacity model.
4. Chooses an actual joint tuple nearest the occurrence-weighted log center of
   each occupied bin. Independent axis quantiles never form invented combinations.
5. Retains joint witnesses for each dimension's smallest and largest extent,
   and expands one independent coordinate at a time to an upper envelope of
   1.25 times its maximum, rounded to the selected granule where applicable.
6. Adds at most eight additional observed tuples, favoring frequent shapes.
   Occurrences distribute sampling effort; they do not weight expected latency.

The default spectrum budget is 64 members globally. Every deferred candidate,
unrepresented bin and unmet downstream requirement remains in the result. Too
small a budget yields `budget_incomplete`; it does not shrink the declared
coverage denominator. Sampling parameters and original metadata/resource hashes
are bound into the plan identity. This does not change a frozen existing cohort,
its objective, numerical tolerances or plateau calculation.

`planned_complete` means every requested point, including optional density
samples, fits this plan. Product-size bins and coarse morphology do not prove a
maximum nearest-neighbor distance along each individual axis; inspect per-axis
gaps before claiming a sufficiently dense intended workload spectrum.

The output is a plan, not source, an admitted cohort or full-model coverage.
The existing `capacity_families` must additionally supply store-fit/spill knees,
independent M/N/K tails, queue, packet, readout, residency and supported-dataflow
boundaries. The plan retains those requirements explicitly. Target software
support and a legal source realization must be checked before generation.

Translate admitted tuples into ordinary reviewed component obligations; an
obligation with no axes already produces one member. For coarse categorical
interactions, reuse `component_coverage_inputs.covering_points`, which bounds
pairwise coverage and reports missing cells without a full Cartesian product.
Do not change the original automatic component policy's shape-free examples or
silently reinterpret older versions.

## Connected and reduced-depth tiers

Keep three complementary development tiers:

| Tier | Required evidence |
| --- | --- |
| Kernel spectrum | Correct accelerator work over the shape/contract strata and hardware boundaries |
| Connected patterns | Packing/layout, resident/reloaded weights, configuration/fences, compatible operation chains, attention QK/PV and KV or prefix reuse |
| Reduced-depth models | Ordinary full-graph compilation and complete outputs/state, preserving stage changes and stateful invocations while removing redundant repeated blocks |

Complete reduced-depth public model factories still need implementation. Suitable families
include a stem, one bottleneck at every spatial/channel transition and a readout;
an attention/MLP block with actual grouped-head layout, prefill and at least two
decode updates; and vision/prefix/action blocks with at least two denoising
updates and final action decoding. Use fresh independent parameters, inputs and
references. Removing repeated depth must not silently shrink dimensions until
the intended working-set or layout mechanism disappears. Record which pressure,
topology and state properties each reduced model preserves and omits.

Existing generic matmul/copy/fork graphs are useful integration tests, but do not
implement those model families. Existing `public_boundaries` renders fresh
PyTorch connected cuts through normal sealed capture and the ordinary writer;
its capacity-axis expansion does not currently consume this spectrum plan.
The separate `development_envelopes` route now consumes this spectrum into
fresh plain/transposed-RHS PyTorch source factories and joins existing capacity
families. Its `development_cohort` producer freezes every request, source, ABI,
independent reference where budget permits, and exact coverage/refusal matrix.
Additional factories cover strided/padded spatial-to-pointwise transitions,
grouped-head QK/PV with growing KV state, shared RHS, recurrent prefix/action
state, and a connected vision/prefix/action pattern. The ordinary capture/writer
diagnostic checks complete fresh outputs. These patterns do not establish full
reduced-model coverage. Numerical software correspondence, private exclusion,
accelerator admission, baseline execution and qualified timing remain required.
No validation workload was executed to construct or test this route. Public
architecture configs are now disclosed development inputs, not held-out models.
