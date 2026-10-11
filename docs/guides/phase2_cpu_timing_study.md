---
title: Phase 2 CPU fidelity and timing study
kind: guide
status: current
owner: experiments
last_verified: 2026-10-11
related: [component_phase2_workflow, phase2_test_justification]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_fidelity.py, packages/merlin-experiments/src/merlin_experiments/phase2/component_compiled_arms.py, packages/merlin-experiments/src/merlin_experiments/phase2/timing_study.py]
---

# Phase 2 CPU fidelity and timing study

## Normal Phase 1 corpus feedback

`prepare_ordinary_search_policy` selects the original normal functional run,
frozen development corpus, target descriptor, RTL facts and gSIM certificate.
The returned `OrdinarySearchPolicy` retains `CorpusFeedbackPolicy` actions and
receipt verification. Its queue entry evaluates issued immutable candidate and
baseline revisions through the existing ordinary package executor, with fresh
job-local measurement state and the complete original member roster. Structural
analysis and executed-command counts remain separate observations. Rank only a
complete correct comparable measurement roster; partial or deferred cells
cannot establish a winner. Fresh final functional and formal measurement gates
still apply after selecting and freezing the compiler.

This normal route uses Phase 1's selected runtime and does not require the
separate strict component qualification ledger. Its ranking scope is simulator
cycles for the frozen development corpus. It does not establish complete
whole-model host cost, physical traffic, FPGA timing or a new engine
qualification. Actual selected tools and certificate/build artifacts remain
external run prerequisites. Retained member fanout is serial pending execution
context propagation into the original member worker threads.

Public program numerical execution is a separate route from the legacy reduced
probe policy. `evaluate_public_program_cpu` requires a live
`ComponentIndependentExecution`, an exact selected CPU engine, and an immutable
host-selected registry containing disjoint public and protected source hashes.
It invokes the original ordinary evaluator and checks its actual native process
against `execution_engine_identity`. Public labels, callbacks, import strings
and saved PASS records cannot supply the evaluator. Protected programs stay
outside this route and retain the original final evaluator after compiler freeze.

The current ordinary independent evaluator covers checked component programs.
A connected generated DAG can use that scope when its original source/runtime
qualification covers it. Full model coverage does not follow from component
qualification. A full model numerical or timing request remains UNKNOWN while
the ordinary whole-program issuer is absent. The legacy
`perf.execution_policy.require_probe_execution` continues to refuse model
descriptors; its default and the existing final gates are unchanged.

Spike output and ISA observations establish no physical traffic or timing. Native
fixed-memory FIR execution remains conditional functional evidence. GSIM and
Verilator engine source/output equivalence alone supplies no cycle equivalence
or timing ranking. FireSim Verilator metasimulation executes on a CPU; it is
not an FPGA measurement. Timing requires the exact final public memory, cache
and clock configuration and independently evaluated physical runtime and timer
semantics. Matching source hashes are necessary and do not issue those roles.

## Optional compiled screening features

`CompiledStaticArmFeatures` uses the original scoped compiler transport,
`compile_source_only`, and `observe_component_compiled_features`. It reopens the
original checked source/ABI and actual LLVM/object/link/symbol observations.
An issued queued revision uses its shared immutable source directly through the
original read-only compiler mount, bound to the same frozen edit authority.
Legacy sources retain their private copy. Extraction reopens the compiler path
actually consumed; output stays within the original qualified execution owner.
Extraction runs no engine and allocates no tensor values. Unsupported source or
emitted bodies refuse. Emitted site/access-width counts and linked bytes leave
dynamic counts, physical traffic, cold/warm costs, numerical correctness and
legality UNKNOWN.

Selecting this owner through `build_component_analytical_provider(compiled_arms=...)`
still requires the original live independent runtime, measurement and held
applicability qualifications. The original complete-cost model, conservative
intervals and screening policy remain the pricing and ordering owners. This
adapter creates no physical issuer and no final promotion. Missing complete
features therefore leave feedback UNKNOWN, including when static extraction
succeeds. Existing source/artifact checks govern cache reuse.

`build_independent_component_rtl_provider` selects the existing
`prepare_development_feedback` evaluator, the exact fresh baseline/cohort/target,
and `runtime.services.rtl_executor`. The live measurement issuer must already
grant that role. Certificate, facts, elaborated CIRCT and validated engine/model
artifact pins must belong to its original source membership; the unchanged
certificate gate admits every exact development workload. This constructor
does not extend the certificate envelope or grant a new timing scope.

The queued search ranks paired **measured simulator cycles** over the exact
generated component roster. Host preparation, allocation, reconstruction,
replay, dispatch, publication and cleanup costs and whole-model timing remain
outside that objective until independently measured complete-cost evidence
exists. Queued complete-roster ranking selects
`DevelopmentGsimFeedback.evaluate(complete_roster=True)` so losses on a cheap
prefix cannot hide later savings. Ordinary feedback keeps its default prefix
rejection. This option changes no workload/certificate/correctness admission.

## Nonexecuting study command

Run the installed command, or its fixed module equivalent:

```sh
merlin-phase2-timing-study --request /assigned/study/request.json --output /assigned/study/plan.json
python -m merlin_experiments.phase2.timing_study --request /assigned/study/request.json --output /assigned/study/plan.json
```

Both commands only validate and write a fresh plan. They do not load a backend,
query AWS, provision resources, execute a provider, compile or simulate. Public
requests have a closed data schema: exact target/registry/configuration pins,
public source membership, selected engine identities, paired replicate count,
and an explicit stage deadline of at most 600 seconds. Supply allocated AWS
instance identity, CPU IDs, RAM, disk, canonical private work root, and complete
compile/extraction/engine environment mappings. Commands and import strings
are rejected. Keep generated plans in the configured `perf-studies` artifact
concern; the request and its source/configuration inputs are immutable selections.

The plan retains every program, engine, replicate, cold/warm stratum and original
setup/compile/extraction/execution/calibration/held/complete-cost stage. It records
wall, CPU, sampled process-tree RSS and output disk metrics. Requested cold/warm
labels do not prove cache state. Setup, calibration, held validation and original
eleven-stage complete cost cannot disappear from the denominator.

Private controllers may supply an `OrdinaryCpuStudyAdapter` with the original
live compiled/runtime owners. Its fixed stage profiler verifies the actual
allocated AWS instance with IMDSv2, CPU affinity, available RAM/disk and exact
public roster before using the existing supervised worker. Compiler/static
observations use the original fixed functions. Execution uses the ordinary full
compile and numerical evaluator, including complete outputs and actual engine
and environment joins. It does not subtract compilation from an elapsed total.
Unimplemented or unqualified full-model/setup/calibration/held/complete-cost
producers remain missing. Sampled RSS may miss short peaks; disk covers retained
assigned output, with external services/writes UNKNOWN. These are diagnostic
cost observations, not scientific qualification or OS resource isolation.

Charge one-time setup, calibration and held validation as `M`, complete ordinary
RTL evaluation per proposal as `S`, complete cheap compile/extraction/Spike cost
as `E`, and the retained RTL fraction as `q`. Break-even requires
`N*((1-q)*S-E)>M`. Equality does not establish savings. An unknown term leaves
the result UNKNOWN. Screening still requires 95% ranking agreement, at most 10%
relative error, 95% interval coverage, and every original evidence minimum:
100 decided pairs, 20 pairs per qualifying slice, three slices and 20 predictions.
Final acceptance retains its original full roster and independent physical gates.
