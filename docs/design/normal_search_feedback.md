---
title: Normal compiler search and diagnostic feedback
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-12
related: [phase2_corpus_search, phase2_cpu_timing_study, independent_performance_feedback]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/selfcheck.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/candidate_observations.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/candidate_census.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_fast_feedback.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_calibration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_screen_features.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_queue_calibration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_measurement_selection.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/provisional_search.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/queued_search_actions.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_costs.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_priorities.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_program_measurement.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/stage_prompt.py
  - src/merlin/perf/candidate_diagnostics.py
  - src/merlin/perf/candidate_cost_graph.py
  - src/merlin/perf/host_cfg_activity.py
  - src/merlin/perf/dynamic_census.py
  - src/merlin/runtime/complete_call_timing.py
---

# Normal compiler search and diagnostic feedback

The normal route reuses Phase 1's selected public target,
specification, ISA, toolchain, neutral runtime and observers. Candidate diagnostics
are available during Phase 1 without a successful baseline. Normal Phase 2 search
still requires the original successful functional fork and timing certificate.
There is no additional component qualification ledger prerequisite.

Diagnostics describe the candidate's own artifacts and executions. They supply
no compiler implementation, preferred schedule or historical answer, and do not
replace functional admission or measured evaluation.

## Capabilities and evidence limits

| Item | Capability | Evidence limit |
| --- | --- | --- |
| Phase 1 selfcheck | `--candidate-diagnostics` observes own public emission; `--dynamic-census` follows original passing numerical execution | Neither probe grants admission; missing own executable or sandbox joins remain UNKNOWN |
| Normal cheap feedback | Complete public compile and Spike roster, rich source/emission diagnostics, original scalar features and bounded census | Required RTL tiers remain pending; counts cannot price physical traffic or cycles |
| Normal search | Frozen revisions, bounded queue, provisional shortlist, measured portfolio, repeat ranges, priorities and parent restoration | A provisional candidate has no winning score or promotion authority |
| Model producer | `search-calibrate` joins completed own cheap and full RTL jobs and installs a live screening owner | Original held gates must qualify; absent or out-of-domain evidence stays UNKNOWN |
| Public graph generation | Existing complete public programs and explicit typed connected cuts including supported casts and reductions | Unsupported grammar or source/effect joins refuse; logical pressure does not establish physical residency |
| Performance objective | Complete correct comparable development roster, using each original measured call boundary | Loader/startup and external parent costs remain outside a selected complete-call timer |

The prompt must describe this actual objective and available tools. A request to
optimize an unmeasured complete model, paired with a prohibition on measuring
any complete public program, cannot establish an end-to-end improvement.

## One candidate observation record

The copied Phase 1 selfcheck client forwards `--candidate-diagnostics` through
the original broker to public emission, integrity and sandbox entrypoints. It
collects own source, emitted IR, command buffer, build products and declared
options without tensor/oracle execution or a certified comparison compiler.
`--dynamic-census` instead follows the original public numerical selfcheck;
these are separate requests. Normal Phase 2 cheap workers reuse the observer
for both issued revisions on identical frozen public members.

Eligible normal public model members select the original native candidate route
through `public_program_measurement.selection(..., simulator="spike")` with an
explicit L2 ceiling and `complete_call_timing=False`. The same source/ABI/ISA and
full numerical gates run on Spike. A passing `candidate_numerical_screen` records
L0/L1/L2 only; original required tiers remain intact, L3 stays pending and the
overall grade remains incomplete. This produces no cycle measurement.

| Question | Observation | Evidence and limit |
| --- | --- | --- |
| What reaches each execution lane? | Source operations, shapes, dtypes, work extents, emitted placement and explicit refusal reason | Join source nodes to actual emitted entries; declared placement alone is insufficient |
| Where does representation change? | Casts, quantization, materialization, layouts, copies and publication boundaries | Source/IR observations; logical bytes are not physical traffic |
| What does the host emit? | Arithmetic/conversion families, loads/stores, constant CFG activity, allocations, code/data extents and available frame records | Unsupported control flow, peak stack and constant-table semantics stay UNKNOWN |
| What actually repeats? | Lexical entry instruction/command activity and requested addresses | Bounded functional trace; outside-entry work is separate without complete callee/harness attribution |
| How much extra work is requested? | Source work, declared device geometry and repeated operand requests | Decode only supported current ISA requests; tails or useful issued work need their own joins, and counts do not establish cycles |
| Which resources might constrain it? | Declared allocations and graph tensor dependencies | Physical residency, resource intervals, bank conflicts and overlap remain UNKNOWN without additional evidence |
| Did the change reach the executable? | Source/IR/object/ELF identities and structural differences | Bind the artifacts actually built, including composed transform selections |
| What is still unknown? | Missing joins, unsupported terms, incomplete windows and unavailable prices | Never remove an unknown cost or turn it into zero |

Records retain available artifact, compiler/member, invocation, tool and
configuration identities, exact scope, completeness and collection cost. A
missing consumption or ownership join stays explicit. Literal tensor payloads,
private host paths and expected/reference outputs are withheld. State whether a
quantity is declared, derived, functionally executed, modelled or RTL measured.
Report the candidate's own observations and differences, without prescribing a transform.
CCA may join source, representation, resource and dependency facts to editable
compiler owners; it is not a comparison against an expert implementation.

For an original native model grade, `observe_generated_artifacts(...,
original_grade=...)` selects explicit source/emission/build/ELF roles from the
original receipt. It reopens the original LLVM-to-object-to-ELF invocation chain;
a matching basename or a copied saved status cannot replace that join. Normal
FAST retains `artifact_join.kind=original_native_model_products` with source,
command-buffer and producer hashes. Calibration rejoins those artifacts to the
selected same-ELF RTL process, engine and full console. Missing or changed roles
remain UNKNOWN or refuse through the original integrity checks. Raw receipts,
products and private paths are not agent feedback.

Offload reporting keeps different denominators separate: operation count,
arithmetic work and movement volume. A timed host/device percentage requires
attributed timing. Full-trace instruction totals do not provide that percentage.

For bottleneck attribution, collect counters and timing over the same bracket,
inspect the complete emitted artifact, and compare controlled alternatives over
identical work. A factorial comparison can separate a kernel change from its
host or surrounding graph effects. Repeat unchanged executable controls to
estimate measurement variability. Static instruction counts and functional
retirement counts cannot substitute for measured or qualified predicted cycles.

## Evaluation ladder and escalation

The normal queue supports an explicit selected compile/Spike diagnostic request
as well as selected RTL measurement. Both use the original frozen public member
identities and evaluators. Unselected obligations stay pending in the complete
roster; neither partial kind can enter complete empirical calibration, measured
promotion or plateau decisions. This avoids paying for an entire cheap sweep
when the author has a specific development question. It does not infer source
dependencies or certify that unselected inputs are unaffected.

### Observe the cost of an evaluation

The parent queue measures each original action from CPU assignment, before
verification and setup, through result validation, descendant cleanup, durable
ownership release and artifact publication. Admission and queue wait have
separate durations; request latency includes them. Subsequent cost/status export
and client transport are outside this boundary. Worker-only elapsed time cannot
replace it.

`evaluation-status` returns this observation as `evaluation_cost`. Successful,
failed and cancelled outcomes remain distinct; incomplete cleanup or unavailable
cost publication gives UNKNOWN. Original job, source pair, corpus, engine,
environment, resource and saved-product identities must still match. Cache state,
CPU consumption and separate arm costs remain unknown.

`search-status.evaluation_priorities` joins passing selected singleton costs to
the measured incumbent's existing potential savings. It retains every repeat,
separates evaluator and resource contexts, and reports conditional savings per
observed pair-action second. It cannot predict another compiler's cost, divide a
batch across members, or sum singleton costs into a parallel sweep prediction.
Missing bounds keep observed costs visible with unpriced savings. This feedback
does not change compiler ranking, calibration, promotion, plateau or final gates.

| Level | Use | Escalate when |
| --- | --- | --- |
| Compile and structural inspection | All sizes: legality, coverage, interfaces, effects, resource feasibility and emitted differences | A result depends on execution or an unsupported source/IR construct |
| Original functional execution | Required output checks on selected public cases | Numerical checks fail or required RTL tiers remain pending |
| Bounded functional census | Selected cases: lexical entry instruction activity, command multiplicity and requested addresses | Cycle price, cache, physical traffic, contention or overlap decides the choice |
| Derived bounds and priced model | Large public shapes/graphs: feasibility, headroom and conditional ordering | Prices are absent, the regime is new, intervals overlap or a prediction contradicts a bound |
| Selected CPU RTL | Complete decisive public components/programs with the original runtime and timer | Resolve a specific uncertain ranking or extend independent model validation |
| Protected final execution | Frozen compiler, original full output and instruction gates | Final hardware performance comparison after freeze |

The implemented census streams the original selected Spike trace into bounded
aggregates. Its default limits are 30 seconds, two million retired instructions,
one GiB process memory, 32 MiB raw output, 8 KiB lines and 4096 histogram bins;
the enclosing deadline can shorten the window. The original sandbox is narrowed
to one actual linked ELF and selected tool inputs. Missing policy or producer,
truncation, unsupported decoding or attribution gives UNKNOWN after proven
cleanup. Unresolved descendant cleanup aborts the enclosing job and retains its
owner. Raw streams and process receipts stay private. Outside-entry/callee work
is retained separately and unpriced. Collection overhead is a tool cost.

Spike establishes functional execution and instruction activity; it supplies no
Rocket timing, cache hits, physical bytes, array utilization or overlap. Native host
timing can reject pathological construction costs, but does not price another
host ISA. Neither replaces selected RTL timing.

## A model richer than roofline

A roofline bounds arithmetic and movement service. It cannot price a program
that performs substantial preparation, conversion, reduction, replay or dispatch.
`candidate_cost_graph` observes verified original task/source dependencies,
per-task host CFG work, source types/reductions and declared representation and
device requests. Opaque calls/assembly, unowned work, dynamic multiplicity and
device/transfer/synchronization service costs remain UNKNOWN. Its pure
composition accepts externally established `CycleInterval` stage prices: serial
addition is an explicit conditional premise, and overlap needs an independently
observed complete `ActivityTimeline` matching stage IDs, prices and dependencies.
The observer issues no prices or hardware overlap authority. Keep inclusive
timers separate from contained costs; missing effects keep the total UNKNOWN.

Derive geometry, capacities, legal extents, ports, queue limits and available
latencies from the selected RTL/ISA facts. The minimal software specification
supplies mathematical semantics, numerical/effect permissions and interfaces.
It need not encode a scheduling recipe. Host dependency/cache costs and external
memory service that cannot be extracted require public controlled measurements.

Start with existing observed feature and cost-composition utilities. Price only
terms backed by current evidence; add a detailed event model only when it can
resolve an observed ambiguity. Do not begin by constructing another full RTL
simulator. Target instruction/resource interpretation belongs in target data or
the independent OOT provider; generic composition and host observers belong in
Merlin.

Normal empirical screening also accepts complete bounded activity counts when
the census matches the compiled ELF. Missing, truncated or foreign-executable
observations remain unknown. These counts describe instructions, commands and
requested addresses; coefficients fitted to selected member timing are empirical
predictors, never physical service prices or timing obtained from Spike.

The normal empirical producer selects identifiable columns from public training
observations only. Calibration and held public member partitions are frozen
before observations; alternatives compare the same member and configuration.
It retains the original 95% ranking agreement, maximum 10% error, 95% interval
coverage, 100 decided pairs, 20 per qualifying slice, three slices and 20 held
predictions. Refit or abstain outside the domain. A qualified empirical interval
does not turn a coefficient into a physical service price. Historical compiler
labels, protected workloads and coefficients fitted to them are excluded.
Predictions are screening evidence, never independent timing replicates.

## Search protocol

1. Preserve the successful Phase 1 compiler as the immutable initial baseline.
   Establish original numerical/ISA gates and the actual measured objective on
   admitted public programs. Until then, prepare observers and corpus purposes;
   do not start an optimization run under a fabricated admission.
2. Use the incumbent's measured costs and independently derived headroom to
   choose high-value questions. Prefer potential absolute savings across the
   public objective over the largest isolated utilization ratio. Unknown bounds
   keep measured expensive cases visible.
3. Let up to three authors propose distinct hypotheses from a recorded parent.
   Each states scope, expected emitted difference, conditions, regression risks,
   numerical obligations and an observation that would refute the hypothesis.
   The coordinator integrates coherent edits and freezes each revision. Shared
   toolchains and corpus are readonly; retain source once per content identity.
4. Inspect all contenders cheaply. Reject independently established legality or
   numerical failures. Record unchanged emission; deduplicate equivalent artifact
   requests only under exact input/build identity. A source edit is not evidence
   of a performance change.
5. Run affected public functional/census cases and qualified model screening.
   Keep provisional candidates and unresolved alternatives separate from the
   measured incumbent. A partial result cannot become a winning score.
6. Spend RTL on a small diverse shortlist and the cases that distinguish their
   predicted behavior. Include a complete connected member when an inter-operation
   claim depends on host preparation, movement or consumers. Record collection
   costs and revise the shortlist when screening has poor ranking fidelity.
7. Measure the complete required public roster before promoting a new incumbent.
   The measured portfolio uses the fixed Phase 1 baseline, equal family shares
   and equal member shares within each family, preserving every repeat range.
   Recorded exploration parents establish lineage, not a different objective.
   Only then compose independently useful changes and measure their interaction.
8. Stop under the existing budgets/reserve. Count a plateau only from completed
   diverse valid comparisons; failed or unresolved observations are not evidence
   that performance saturated. Stop edits, drain work, select measured bytes and
   run the fresh complete final gates.

The normal broker implements these controls; arguments refer only to issued
revisions or the frozen public roster:

| Action | Arguments | Effect |
| --- | --- | --- |
| `revision-freeze` | `parent_revision`, `hypothesis`, `edit_family` | Freeze the integrated source under its original edit authority |
| `evaluation-submit` | `revision_id`, `kind=corpus-compile-spike-feedback` | Observe complete public compile and Spike results with diagnostics and bounded activity |
| `search-status` | None | Reopen issued job statuses, measured incumbent, provisional shortlist and retained history |
| `search-resume-parent` | `revision_id` | Restore an issued exploration parent after joining authors; leave the measured incumbent unchanged |
| `evaluation-submit-selected` | `revision_id`, `members=family/capsule,...` | Measure diagnostic public members through the original RTL evaluator; other rows stay unmeasured |
| `evaluation-submit` | `revision_id`, `kind=corpus-rtl-feedback` | Measure the complete roster eligible for development promotion |
| `search-calibrate` | `revision_ids=issued_a,issued_b,...` | Join completed cheap and full RTL jobs for two or more issued alternatives, fit and validate a screening owner |
| `search-finalize`, then `search-select` | None | Drain and prove cleanup, then restore complete measured selected bytes |

`search-calibrate` starts no simulator. All jobs must be terminal; selected-only
timing cannot supply its complete labels. Run it before `search-finalize` and
the final queue drain, while exploration admission and its budget remain open.
The partition is fixed before measurements, and there are at most three
installation attempts. Saved reports cannot reconstruct a live owner. Failed
qualification retains UNKNOWN; qualified intervals only order experiments.

The provisional shortlist filters incomplete functional screens, records
unchanged/duplicate own emission, and retains declared edit-family diversity.
Missing predictions leave unresolved alternatives visible. Neither that list,
parent restoration nor diagnostic partial timing promotes a compiler or counts
as complete measured plateau evidence. Selection also requires complete measured
feedback for the unchanged starting compiler, so the baseline fallback is real.

Changed-code scope is a scheduling aid, not a sound regression proof. Re-emission
must reopen actual products and dependencies. Global/shared changes affect all
uncertain members. Simulator result reuse remains disabled under the current
reset/non-file closure; existing verified object reuse does not change that.
Full final evaluation stays fresh.

## Public cases that make feedback useful

Every member declares a performance question and a complete output/cost boundary.
Use independent public semantics and facts to vary tails, strides, reuse counts,
escaped users, simultaneous live values and resource transitions. Preserve the
property under investigation when reducing a witness; smaller extents alone
do not preserve capacity, overlap, reuse or fixed-cost balance.

Pair isolated operation witnesses with connected producer/consumer programs,
host transforms, reductions and repeated/shared-input cases. Keep compile-only
large shapes even when their execution exceeds the budget. Admit bounded public
complete programs for timing; protected validation programs remain inaccessible.
Source membership, interfaces, emitted placement and original full-output gates
must be checked before transferring a result between cases.

## Measured boundary and execution requirements

Eligible complete public programs use the original candidate native whole-program
grade and its full output comparison. Selected gSIM timing requires the optional
reviewed `logical_harness.complete_call_timing` declaration and original counter,
ISA and clock source pins; absent data gives no complete-call cycle boundary.
The [search guide](../guides/phase2_corpus_search.md#complete-public-call-timing)
defines its exact data shape. The timer includes main preparation through
completed whole-program outputs and selected output publication. Immutable input
staging, loader/startup, metric formatting, DONE/exit and external parent
transport/readback/decoding remain excluded. Cold/warm state stays UNKNOWN.
Reduced members keep their original timer scopes; mixing observations does not
expand either boundary.

Physical stage prices, cache service, traffic, residency and overlap require
independent public evidence. Neither diagnostics, typed source cuts, a complete
call delta nor an empirical fit supplies those facts. The normal AWS cost study
executes fixed cheap/RTL/calibration producers on explicitly allocated resources
and preserves costs even when qualification remains UNKNOWN; its default plan
launches nothing. Actual Phase 1 completion, worker qualification, costs,
convergence and protected final performance require the original executions.
