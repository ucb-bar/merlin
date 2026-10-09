---
title: "Independent compiler tests and calibrated performance feedback"
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-09
related: [component_compiler_convergence, component_phase2_workflow, beam_cca_architecture]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/sweeps.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_workflow.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_cca.py
  - src/merlin/perf/phase2_analytical_provider.py
  - src/merlin/perf/phase2_calibration_bundle.py
  - src/merlin/perf/fast_estimate_validation.py
  - src/merlin/xdsl_dialects/lowering/global_plan.py
---

# Independent tests and performance feedback

This design specifies information each phase needs to build and optimize a
general compiler. Existing generators, objective bindings, CCA and calibrated
provider APIs implement parts of it. Semantic scenario generation, automatic
experiment execution and a qualified fresh campaign remain incomplete. No
reference implementation or held-out model is admitted by this design.

## Phase responsibilities

| Stage | Required inputs and tools | Required result |
| --- | --- | --- |
| Phase0 | Reviewed supported operation/numerical/effect domain; byte-bound public hardware/ABI facts; independent generator rules and golden engines | Functional obligations, performance alternatives, resource-boundary sizes, independent transfer selections and explicit missing coverage |
| Phase1 | Declared compiler edit surface; upstream graph/IR inspect and lowering witnesses; functional tests, native execution and selected functional simulator | A frozen functional compiler with supported composition, layouts, tails, effects, fallback, host/device linking and full output checks |
| Phase2 | Complete-cost component feedback; CCA use/effect/resource witnesses; calibrated cycle intervals; native/functional features; selected RTL evaluator; bounded experiment workers | A general policy that improves independent components and transfers to withheld compositions without weakening correctness |
| Final evaluator | Frozen compiler; protected model/input/output budget; selected target/hardware/timer and actual executable | Whole-output correctness, instruction restrictions and measured whole cost on the held-out workload |

Expose enough public contract information to make legal choices: semantics,
numeric order and permissions, shape/layout, consumers/effects, resource intervals,
ownership/lifetimes, cost scope, available evidence, refusal reasons and editable
compiler owners. Exposing a historical winning implementation, held-out graph,
golden values, capture-derived sizes or measured validation labels is a different
action and remains forbidden. Every prior generic library admitted to an
experiment needs explicit disclosure; a tool cannot silently grant new authority.

## Components and small composed networks

Isolated kernels establish local legality and service costs. They cannot establish
that packing, preparation, dispatch, allocation and publication compose profitably.
Include independently generated small programs with reusable structures such as:

- convolution, virtual padding, residual observation and pooling;
- projection, attention products, normalization and nonlinear observation;
- matrix products, feed-forward activation and an output consumer;
- repeated producers, shared immutable inputs, multiple consumers and epoch mutation;
- mixed host/device paths, readout/requantization and output publication.

These are semantic structures, not copies of validation networks. Declare their
rule classes from the supported domain before inspecting validation data. Generate
weights, numerical edge inputs and private answers independently. Withhold depths,
combinations, layouts, rectangles, reuse counts and policy regimes, not only random
seeds of a single graph. A separate training network family is useful when it
exercises these contracts; a lookalike assembled from protected graph fragments
or exact validation shapes is not independent.

Exact intermediate agreement is one source-proof route. A separately admitted
approximation may change intermediates while preserving the unchanged final
output budget. Local permission does not authorize an entire graph replacement.

Exercise upstream capture/lowering as well as backend IR. Verify source semantics
and ordered numerical observations through FX/prepared IR, host/device partition,
emission, linked execution and all outputs. Keep normal builders, independent
golden engines and program admission authoritative. The current component-only
generator refuses model/capture inputs; this design does not add a parallel
capture route or claim a general network composer is implemented.

## Sizes that distinguish strategies

Tile-relative extents and residency ladders already provide useful independent
sizes. Add simultaneous live-buffer footprints around selected resource boundaries,
aligned and partial tiles, transfer-width/segment boundaries, accumulator output
blocks and reserved intervals. Consider physical banks separately from total bytes.
Cache/table cases require selected host cache facts; unknown cache geometry remains
unknown rather than inherited from a simulator.

For a strategy needing reuse, include one use and multiple uses. For a fusion
needing closed users, include an escaped user. For a certificate needing immutable
epochs, mutate its dependency. For an observer needing a numeric preimage, include
ties, cancellation, saturation and fallback. A large matrix alone does not exercise
these requirements. Record requested, emitted, refused and unavailable scenarios.

## Complete cost and feedback

Every comparison must bind the same work, outputs, numeric budget and cost
boundary for baseline and candidate. Include setup, allocation/init, packing,
proof/certificate construction, all device work, readout, reconstruction,
refinement/replay, dispatch, publication and cleanup when they belong to that
boundary. Report warm reusable setup separately without discarding cold or
one-shot costs. Never add contained region times to an already inclusive callback.

The feedback record should contain:

1. source pattern, users/effects, applicable policy and concrete refusal;
2. effective options and actual emitted/linked entries, plus byte identities;
3. typed layout/resource/ownership proof and requested versus observed traffic;
4. full functional outputs and original fallback/accuracy obligations;
5. exclusive counter scope, units, dependencies, branch/code/frame/table footprint,
   preparation, replay and publication activity;
6. complete measured cost or calibrated interval, domain and missing prices;
7. controlled alternative, independent transfer result and unresolved interaction;
8. the compiler owner to edit and next experiment that distinguishes hypotheses.

Do not equate retirement counts, native wall time, requested bytes, nominal MACs,
partial RTL runs and hardware cycles. CCA explains correspondence and mechanisms;
an analytical feature without a price does not become zero-cost work.

Resolved cycle intervals require finite nonnegative ordered endpoints. Boolean
point estimates, NaN and infinity refuse at construction; an arithmetic overflow
in the empirical screen returns an explicit unknown estimate. Zero remains a
valid numeric endpoint and is distinct from missing work. Broker admission still
checks finite bounds and qualified provenance independently, including malformed
callback objects that bypass the value constructor. These arithmetic checks do
not replace calibration, complete-cost coverage or held-evidence thresholds.

## Ranking and choosing the next experiment

First gate correctness and legality. Then compare complete-cost intervals inside
qualified domains. Rank a supported improvement by critical-path opportunity,
confidence, evidence cost and complementary coverage. Derive importance from the
selected independent corpus's own observations or reviewed domain weights; do not
use protected model frequencies to tune a training score.

If intervals overlap, a cost is unpriced, or a case leaves the calibrated regime,
the ranking is unresolved. Prefer the cheapest controlled experiment that can
resolve that uncertainty: a native fallback-cost comparison, dependency feature
screen, a complete RTL component or a new mechanism calibration. Keep diversity
across resource regimes so that one family of small kernels does not consume the
search budget. Preserve ties, refusals and negative results to avoid repeating
already rejected hypotheses. Comparing against an obsolete baseline can make an
improvement look competitive when it loses to the current legal baseline.

Existing analytical providers require selected evidence and retain UNKNOWN.
The ordinary component broker now exposes the analytical provider's existing
conservative experiment order. It replays the same complete costs and gate
observations, using equal generated-family shares and equal member shares within
each family. Saved feedback rechecks that roster, ordering and arithmetic. Failed
or unknown gates retain refusal or unresolved status. This orders observed
opportunities; automatic selection and execution of new distinguishing
experiments remains incomplete.

## Roofline and calibration

A roofline gives compute and movement limits; it is insufficient to select host
instruction order, certificates, tables, packing, replay or synchronization.
Compose resource bounds using dependencies and declared concurrency. Serial
stages cannot collapse into one maximum; overlap cannot be invented. Distinguish
optimistic unique-byte demand, requested transfers and measured physical traffic.
Nominal padded issue work needs a target-state proof before being called a
mandatory hardware floor.

Fit host dependencies, memory/cache service, command issue/fences, readout,
preparation and replay using independent controlled mechanisms. Bind each fit to
the target, engine/configuration, ABI, numeric policy, cost boundary and source
bytes. Freeze parameters before opening held labels. Withhold alternative
schedules, sizes and crossed regimes; repeats measure variability. Report absolute
error, ranking mistakes and calibrated uncertainty. A model matching one reference
label or fitting the same points it scores has not established transfer.

## Parallel CPU search and hardware use

Use bounded independent native and functional-simulator workers for correctness
and features, deduplicate identical emitted objects, apply qualified analytical
intervals, then run a diverse shortlist of complete RTL components. Respect
selected engine admission, shared leases and memory/cycle/wall budgets. Immutable
candidate snapshots and private result paths prevent cross-worker mutation.
Timeouts/cancellation provide incomplete evidence, never partial winning scores.

Final-only hardware evaluation becomes defensible after the selected simulator
and model have independently qualified ranking across all relevant mechanisms
and compositions. Configuration mismatch, unknown memory/cache prices or RTL/
hardware ranking disagreements require development calibration. The final held-
out whole hardware test remains necessary even when the preceding search uses
CPU tools. Existing fresh-author transport/isolation and authenticated final
evaluation must close before claiming an automatic convergence or elapsed-time gain.

Generic generation, workers, evidence, calibration, CCA, ranking, cost composition
and host transforms belong in Merlin. Target layouts, hardware facts, ISA models,
device schedules and simulator adapters belong in the OOT compiler provider.
