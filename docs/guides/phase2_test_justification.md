---
title: "Justifying Phase 2 tests from Phase 0 evidence"
kind: guide
status: current
last_verified: 2026-10-19
owner: experiments
related: [phase0_specification, generating_capsules, normal_search_feedback]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/test_justification.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/bottleneck_priority.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/corpus.py
  - experiments/templates/phase0/performance.yaml
---

# Justifying Phase 2 tests from Phase 0 evidence

Phase 0 generates two distinct cohorts. The public functional cohort tests Phase 1 compiler
behavior. The `_perf` development cohort asks Phase 2 performance questions. A Phase 1 pass is
not a speedup, and a generated `_perf` capsule is not a measurement. The phase selections in the
corpus manifest make this boundary explicit.

For current generated corpora, Phase 2 discovery derives `merlin.phase2.test_justification.v1`
over **every** generated performance member before applying a requested subset selection. Freeze
rechecks the original members and evidence bytes, saves the receipt with the selected capsule
snapshot, and replay verifies its hashes. An incomplete corpus without `MANIFEST.yaml` cannot be
admitted. The receipt is generated under the run's output directory; it is not an authored test
definition.

## Choose tests before compiling a headline model

Phase 0 first counts typed operation forms in explicitly selected development
captures. `performance-basis.json` keeps their M/K/N, precision, static occurrence
count, recognized MAC mass, unknown mass and unresolved placement. Selected RTL
facts constrain which experiments are legal and which machine regimes, such as
tile depth or operand residency, are worth probing. Neither static MAC mass nor
an RTL capacity proves a runtime bottleneck.

The shared performance template defines a falsifiable machine law or a matched
comparison with a negative control. A capture-shaped member is generated only
when its observed point fits that family's unchanged cohort rules; other forms
remain explicit uncovered or deferred demand. Such a member tests shape
relevance, not equivalence to the source operation. Phase 2 must check emitted
code and correctness, measure with the selected timing oracle and replicates,
then compare cycles only after the analyzer confirms equal demand. A small
graph-island witness is useful when standalone timing depends on surrounding
placement or reuse. Full-model runs remain an independent validation of the
extrapolation, never a source for Phase 0 test selection.

Use a staged cost ladder: isolated primitive for a machine law; a connected
development-workload slice for placement, transfer and reuse; then a reduced
iteration model for composed effects. Generate dimensions from the selected
development capture and RTL capacity boundaries, including points on both sides
of a tile, accumulator or storage limit. A small model is representative only
for the named constraint it actually crosses and measures. Its operation count
and static MAC mass cannot weight the expected benefit in a headline model.
Rank levers using identity-matched RTL cycle measurements for the *measured
tuning corpus*; whole-model priority additionally needs a calibrated connected
slice or host-private validation measurement. Preserve unknown and excluded
source chains rather than replacing them with a convenient synthetic graph.

For Gemmini, the current development captures contain mixed-lane
movement→contraction→map chains whose FP32 cast/map and fusion placement are
not yet admitted by the selected software spec. The generated contraction-depth
chain is therefore not a substitute for those source chains. First resolve the
software/placement obligation or produce an independently justified host/accel
boundary witness; only then can its timing support a connected-slice claim.

## Review whether the tests distinguish compiler decisions

A shape inventory is insufficient for freezing a new development cohort. For
each intended policy, identify its legal alternatives, a nearby case where it
must refuse or fall back, and an independent transfer case. Match the alternatives
on the same complete source, inputs, required accelerator work, numerical policy
and measurement boundary. Keep controls and failures in the frozen roster.

| Policy question | Required distinguishing cases |
| --- | --- |
| Retain operands or reload them? | Reuse once, twice and many times; immutable versus mutated storage; alias and intervening clobber controls. Account for all simultaneously live A/B, output, reserved and ping-pong intervals. Total tensor bytes alone do not prove fit. |
| Change blocking, placement or overlap? | Last legal and first overflowing allocations for each storage/bank limit; zero, one and two available next-panel slots; independently vary packet-width boundaries and bank conflicts. Derive these sizes from current hardware and software contracts. |
| Reorder tiles or retain command loops? | Independent axis tails, including one element and one less than a tile; one/two/many loop iterations and code-size limits. Verify complete issued work, reduction order, accumulator state and final writes. |
| Eliminate packing or a layout conversion? | Contiguous and segmented/strided sources; an escaping user that needs the original materialization; ownership and lifetime controls. Inspect emitted objects, since an apparent source copy may already have disappeared. |
| Clean up configurations or fences? | Identical and changed configurations, intervening consumers, pending asynchronous work and drains that remain necessary. A large contraction does not by itself establish queue pressure. |
| Fuse readouts or attention? | Exactly selected scalar/grouped readout, bias/activation order, rounding ties, saturation and output tails; complete QK-mask-softmax-PV with GQA/KV state. Escaping scores and changed state must block invalid elimination. Integer QK/PV witnesses do not establish complete attention equivalence. |

Apply this review at three scales: primitive decision witnesses, connected
source-faithful graphs, and complete public reduced-depth models. Reduced depth
may remove repeated layers; it must preserve the widths, layouts, nonlinear
operations and carried state responsible for the constraint being tested.
Declare any omitted component and unsupported numerical contract explicitly.

An operator may inspect a separately authorized reference implementation for
general mechanisms and counterexamples. Record exact source revisions and
translate findings into independently generated semantic/resource tests. This
does not admit that implementation, its old hardware facts, private captures,
answers or historical timing labels to an experimental author, nor replace
the inherited Phase 1 compiler. Evaluate a learned policy against independent
shapes and compositions after freezing it. Development models already inspected
are development evidence, not unseen validation.

For every requirement, retain the exact candidate member identities and separate
source/reference, capture, compiler emission, full numerical execution,
instruction/effect and timing evidence. Mark an unsupported or unexecuted
requirement as such even if a similarly named family exists. Coarse logical
capacity bands, declarations of residency and source operation counts cannot
stand in for an actual emitted schedule and its checks.

## Turn a measured gap into a development capsule

Use the same identity-bound program and measurement method for the candidate
and its reference. First separate *placement* (an eligible operation stayed on
the host or in a vendor fallback), *algorithm* (tile, reuse, fusion or transfer
choice), *machine scheduling* (the same work takes more target cycles), and
*build throughput* (compilation is too slow to iterate). A host fallback is a
coverage or Phase 1 obligation before it is a Phase 2 speedup opportunity;
do not count a faster host routine as improved accelerator placement.

For an optimization question, derive the smallest source-faithful unit from an
allowed development capture and retain its operand values, layout, neighbors,
transfer boundary and output oracle. An isolated operation tests a machine
law. A connected slice tests whether placement, fusion and reuse survive the
surrounding graph. A reduced model tests composition only if it preserves the
named resource constraint, such as working-set size, launch count or live
buffer pressure. Record which property was preserved and which was not.
Then compare standalone and in-program cycles for the *same* unit on both
candidate and reference paths. If cache warmth, setup or neighboring work
causes a material offset, the standalone capsule is diagnostic until the
offset is explained or calibrated. Exact numerical output and equal-work
checks precede timing; a cheap pre-screen cannot replace them.

Keep compiler-build latency and peak memory as separate reproducible receipts
for large generated programs: per-stage timings, unique-kernel counts and
compiled-function sizes expose caching and host-code scaling problems that a
device-cycle capsule cannot see. None of these measurements should reveal a
sealed validation model to the Phase 1 or Phase 2 authoring agent. Periodic
held-out whole-model runs test transfer of the selected optimization, not the
rule used to select or tune its capsules.

Each row answers a concrete question:

| Question | Receipt field | What establishes it |
| --- | --- | --- |
| Why this target? | `hardware_basis` | Required traits and execution capabilities from the frozen Phase 0 performance facts; an unsatisfied gate refuses a generated member. |
| Why this workload shape? | `workload_need` | Frozen Phase 0 performance-basis rows are compared with the capsule's exact shape and typed arithmetic form. A geometry-class stamp alone never establishes a match. Structural and shape-only candidates remain unverified. |
| What could be false? | `hypothesis` | The declared observation and falsifier from the shared performance family. |
| What is compared? | `matched_comparator` | Declared equal-demand fields and, when present, the candidate group members. Matching remains `planned_unverified` until the analyzer checks emitted programs and results. |
| What detects a spurious effect? | `negative_control` | The family's declared control. Its presence is a plan; the control is not established by generation. |
| How is it measured? | `measurement`, `correctness`, `repeat_dispersion` | Instrument, the concrete timing and correctness engines with their tiers, replicate plan and dispersion rule. A missing policy stays missing rather than becoming zero noise. |
| Can this member run? | `support` | The software screen; unknown support stays unverified and an explicit refusal stays unsupported. |

The receipt also reports `workload_accounting`: captured static MAC mass, known and unknown
denominators, exact arithmetic-form matches, and each still-uncovered contraction demand with
its candidate capsules. These are counts from selected development captures, not model runtime
frequencies or cycle estimates. A matching arithmetic form does not establish a complete
operation, accelerator placement, functional correctness or speedup.

The receipt binds the shared template digest, raw RTL digest, derived performance facts,
`coverage/operation-accounting.json`, `coverage/performance-basis.json`, selected software and
hardware views, evidence manifest, provenance manifest and each capsule's byte tree. Freeze
copies the basis into `test_justification_inputs/` for replay. When the selected
Phase 0 evidence contains a conformance-spec source snapshot, freeze also copies
a small canonical `performance-scope.json`: the required and excluded connected
chain identities and reasons, not the complete software requirement or private
captures. Its digest is bound to the original requirement snapshot digest and
the frozen corpus manifest. The post-GO priority report replays this scope; for
an ineligible source chain it records the `no_eligible_chain` refusal rather than
silently crediting standalone matmuls with connected-chain coverage. Older frozen
corpora without this projection remain `unknown` and need a newly frozen run for
verified connected-scope reporting. Freeze refuses a
changed comparator, control or evidence byte between discovery and freeze. Its status is always
`diagnostic_unmeasured`: it does not certify simulator fidelity, functional correctness, a
negative control, statistical significance or a speedup. Those claims require completed,
identity-matched correctness and timing cells and the family's declared analyzer. In particular,
instruction counts from a functional simulator do not substitute for cycle measurements from the
selected timing engine.

The shared performance template names oracle *tiers*, not simulators: its evidence fields carry
placeholders such as `$target_oracle:L2` and `$target_oracle:L3`. Phase 0 resolves them per target
from the capability contract or the recipe's explicit oracle selection and freezes the concrete
engine names, with the placeholders they came from, into each generated member. The receipt's
`measurement.timing_simulator` and `correctness.simulator` therefore name that target's engines. A
placeholder that does not resolve to a concrete simulator is a generation error for that member
(recorded under `families_not_generated.errors`); it never falls back to whichever simulator is
installed. Its form-performance `PW` members derive form classes and source-convolution windows
from the declared iteration workloads, then plan paired candidate and target-support reference
measurements. A generated pair, even with a declared acceptance analyzer and replicate band, is
still an unmeasured hypothesis; separate measured evidence must establish both arms on identical
demand and the selected timing oracle.

Read `test_justification.json` next to a frozen `performance_corpus_manifest.json`. First inspect
`workload_accounting.uncovered_demands` and `families_not_generated`, then each member's workload
and support status. A missing family may be
inapplicable, unimplemented or failed; it is not evidence that the compiler handles that lever.
Families admitted only by the selected frozen requirement (a captured multi-region scope chain or
a derived model form class) are listed as `skipped_inapplicable` when that requirement selects
nothing, and as `blocked_unimplemented` when the requirement lacks the scope they need. The
emitted-stream families (PD load-configuration residency, PA stationary residency, PJ move-in
placement) are generated: each is decided from the candidate's own decoded instruction stream by the
analyzer its acceptance block names (`merlin.perf.{load_state,stationary,movein}_claim`, dispatched by
`merlin_experiments.phase2.claims.dispatch`). They read no cycles, cap at the correctness tier and
say only that redundant or exposed work is absent, never that anything got faster. A family the
template declares but no materializer can derive yet (PT, whose starting depth is a measured property
of the machine) stays `blocked_unimplemented`.
For a performance conclusion, inspect the later measured result and its analyzer verdict as a
separate artifact. Do not promote a generated test or a diagnostic receipt to a measured claim.

After an admitted paired tuning campaign, `bottleneck_priority.json` requires every selected
member's baseline and candidate results at both the correctness and RTL timing tiers, in both
replicates. It reports the candidate's share of **this selected tuning corpus**, the matching
development-capture demand, and paired cycle savings or regressions. These numbers help decide
which measured capsule to investigate next. They do not establish statistical significance,
source-connected placement, a family-level optimization claim, or whole-model time share;
`whole_model_priority` stays insufficient until separately calibrated evidence exists. A
candidate-only or correctness-incomplete run is not a paired priority report.

For a cheaper **development iteration subset**, pass `--capsules representative`
to Phase 2 authoring. This option is separate from the default full cohort and
explicit capsule-name selection. It derives a deterministic greedy cover of
exact selected-capture arithmetic forms, exact generated operation/input
tensor forms, selected RTL-trait axes, and each
declared optimization lever/level/claim from the
complete `test_justification.json`, with each performance family indivisible.
Fit grids, paired arms and their declared controls therefore stay together.
The frozen corpus binds `representative_selection.json`; inspect its omitted
families/levers, `uncovered_static_axes`, `uncovered_source_demands`, and
connected-slice refusals before interpreting results. This is not a
mathematically minimum corpus or a replacement for full-family confirmation.
It neither weights model runtime with static MACs nor turns standalone
capsules into source-equivalent connected slices. If no complete family covers
a selected evidence axis, selection refuses instead of producing an empty run.

An original host may explicitly pass a live `CheckedSourceModelDiagnostic` and
its same original compiler-pass selection to `prepare_ordinary_search_policy`.
This is an optional connection for selected compile/Spike witnesses: the queue
forwards that original selector to the existing sealed model-child route, keeps
L3 requirements, reports `DIAGNOSTIC_ONLY`, and leaves timing `UNKNOWN`. It does
not infer public-program placement, score a complete cohort, or accept a saved
selector record as authority. The ordinary policy still requires its original
formal Phase 1 admission. A provisional Phase 1 checkpoint cannot instantiate
this policy; neutral retained-revision and fresh-child transport tests do not
establish a qualified target queue run. Actual boxed source/compiler/effect
diagnostics remain separate evidence until those original gates are met and
the selected target queue is independently exercised.
