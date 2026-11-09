---
title: Bounded independent graph topology coverage
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-11-09
related: [component_execution_budget, component_integer_bounds, component_scale_generalization]
code_refs:
  - src/merlin_experiments/phase0/component_graph_variants.py
  - src/merlin_experiments/phase0/component_graph_relations.py
  - src/merlin_experiments/phase0/component_coverage_inputs.py
---

# Graph interaction coverage

The existing component DAG source supports shared producers, aliases, updates
and multiple outputs. Parameter sweeps of a fixed authored DAG did not vary its
depth, number of consumers or repeated logical epochs. A fixed, target-neutral
source family now generates these interactions through ordinary MLIR generation
and the original complete independent integer reference.

The family is selected explicitly in an externally frozen v2 coverage plan:

```yaml
base:
  op: component_program
  kind: model_slice
  M: {axis: M}
  K: 3
  N: 2
  depth: {axis: depth}
  fanout: {axis: fanout}
  graph_family:
    schema: merlin.component_graph_family.v1
    representations: [logical_epochs, fresh_values]
```

The extents and topology parameters are independent selected inputs, with no
validation graph, workload dimensions, performance history or reference
compiler selection. Depth is positive and fanout is at least two. Every
generated computation needs a selected reviewed software operation owner.
Where an independent public example basis is selected, the existing coverage
owner also requires its explicit source-to-owner semantic correspondences.
This fixed family defines generic interactions; it does not copy a public
example topology or use its shapes. Concrete source screening still decides
operation, dtype, placement and layout admission.

## Two representations and all observable values

One initial contraction feeds every subsequent stage. Each stage copies an old
snapshot, forks the current value into distinct copies, joins the copies in
their original left-to-right addition order, and adds that join to the current
value. It publishes the old snapshot, the first fork (also consumed by the
join), and a copy of the new epoch. The final value is separately published.
There are exactly `3 * depth + 1` outputs, and all are independently evaluated.

The `logical_epochs` representation aliases one source storage owner and
updates that owner repeatedly. The `fresh_values` representation explicitly
uses fresh SSA values. Both retain the same arithmetic order and leaves.
Snapshots and escaped forks use distinct copies so later logical updates cannot
rewrite their published values. The existing source analyzer resolves alias
epochs; the ordinary builder functionalizes them into standard SSA MLIR.

Each selected point requests both representations in the same public or
private cohort. The member budget counts both; it cannot issue half a pair.
Source effects are derived from the actual typed DAG. Logical alias and mutation
facts apply to the logical representation; shared producer, multiple consumer,
escaped use and output publication facts apply to both. An effect declared for
an obligation must still be present in every requested member of that obligation.

## Admission before topology or data allocation

A very large depth or fanout cannot first expand millions of node dictionaries.
The fixed source factory derives exact logical cost from constant-size checked
source prototypes before constructing the requested DAG. Repeated stages and
extra forks have affine work and payload costs; an extra fork adds a copy and
an ordered addition. Palettes charge their complete leaf realization once,
independently of the number of stages. No shaped inputs or numerical evaluator
are called to derive these costs.

The selected per-member v2 budget must admit this cost before topology unrolling.
After construction, the ordinary budget owner rederives the full actual typed
DAG cost and requires equality with the prototype derivation. Whole-generation
budget admission then charges development sweeps and both requested variants.
All existing integer interval proofs and numerical choices apply unchanged.
Unknown arithmetic and any potentially overflowing bounded-exact node refuse
before data/reference allocation. Equivalent wrapped final outputs do not
establish bounded-exact safety.

## Private original-output relation witnesses

The private `merlin.component_graph_relations.v1` ledger binds the full requested
pair membership, source parameters, typed source hashes and complete output
rosters. It reopens the original ordinary source and complete golden, replays
the original independent evaluator for every output, and then checks exact
full-output equality between the representations on identical leaf semantics.
No weakened tolerance, final-output shortcut or candidate witness establishes
this relation. Witness replays are additional verification passes over the same
admitted bounded sources; reference admission counts are not a measurement of
total Python work or process heap.

Missing sources, missing numerical paths or one missing representation retain
the obligation and both requested members as unavailable. Receipt verification
replays these source and numerical checks, and joins the actual entire issued
budget roster so an otherwise re-signed report cannot drop a family member.
Public summaries contain only the existing commitments and cohort counts;
private topology, pair membership and output witnesses remain private.
Historical v1 coverage receipts and fixed-DAG generation retain their behavior.

## Evidence limits

This family adds finite bounded source interaction coverage. It proves neither
whole-domain compiler completeness nor generalization to untested large shapes.
Logical repeated buffers and functionalized aliases do not establish physical
allocation reuse, lifetime, asynchronous ordering, synchronization or completion.
Those mechanisms require independent execution controls. Large compile-only
legality, device arithmetic correspondence and performance remain separate
obligations; an over-budget numerical execution is never relabeled a pass.

Regressions use ordinary generation and an additional scalar oracle over every
output, with multiple depths, fanouts, extents and a separate private cohort.
They inspect actual operation counts in emitted MLIR, logical epoch/source
effects, oversized preallocation refusals, missing pair denominators, exact
overflow refusals, immutable roster replay, and identical corruption of escaped
outputs in both variants while final outputs remain correct.

`component_graph_variants.program` constructs the selected representation only
after budget admission. `component_graph_relations` reopens both complete source
and golden rosters. The separate source-only graph plan can reuse the topology
owner, but cannot convert its compile/link results into these numerical witnesses.
