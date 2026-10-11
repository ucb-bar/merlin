---
title: Phase 2 CPU fidelity and timing study
kind: guide
status: current
owner: experiments
last_verified: 2026-10-11
related: [component_phase2_workflow, phase2_test_justification]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_fidelity.py, packages/merlin-experiments/src/merlin_experiments/phase2/component_compiled_arms.py, packages/merlin-experiments/src/merlin_experiments/phase2/timing_study.py, packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_timing_study.py, packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_calibration.py]
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
external run prerequisites. Retained member fanout must match the queue grant
and carry the original execution contexts into each member worker thread.

### Cheap queued observations

Submit `kind=corpus-compile-spike-feedback` for retained revisions to run the
original boxed compiler and Spike ladder through L2. Original required RTL tiers
remain pending. Both arms cover the exact frozen public roster. Fresh LLVM,
linked ELF and link-invocation products feed the existing static-site observer;
unsupported bodies remain UNKNOWN. Static features never establish physical
traffic or timing. Submit `kind=corpus-rtl-feedback` for the original complete
certified timing roster. Only measured timing can select the source.

The live `produce_calibration` producer executes both routes for multiple
immutable public compiler alternatives on allocated AWS CPU. Its explicit
`merlin.ordinary_public_screen_partition.v1` partition contains `calibration`
and `held_validation` lists of `[family, capsule]`, disjoint and covering the
original public roster. These held members are public screening-validation
inputs, separate from the protected final cohort. The fit reads calibration
labels only. Ranking pairs compare alternatives for the same member and
configuration; different shapes cannot supply that denominator. Original gates
require 95% ranking agreement, at most 10% error, 95% interval coverage, 100
decided pairs, 20 per qualifying slice, three slices and 20 held predictions.
Intervals are empirical screening evidence, not physical confidence claims.

### Run the normal AWS CPU cost study

Use the completed Phase 1 run, its admitted readonly baseline, existing target
descriptor/toolchain/ISA/runtime, frozen development corpus, compiler edit
contract, RTL facts and selected gSIM certificate/build. The selected runtime
must already be available through the normal Phase 1 readiness/preflight path.
Allocate explicit CPU IDs, RAM and disk on an AWS instance; the command verifies
the instance through IMDSv2 and checks available resources. It provisions nothing.
Keep `MERLIN_PERF_SWEEP_WORKERS` equal to the number of assigned CPU IDs before
creating the request, and preserve the same effective environment for execution.

`ordinary_timing_study.request_from_policy` exports a closed JSON request from
the actual live normal policy. Supply `resources` with `provider="aws"`,
`instance_id`, `cpu_ids`, `memory_bytes`, `disk_bytes` and a fresh private
`work_root`; supply at least two public compiler `alternatives`, the exact
`public_partition`, and the original `edit_contract_path`. The request retains
all original source identities and an environment digest. It accepts no command,
callback, import string or saved qualification.

```bash
merlin-phase2-normal-timing-study --request normal-study.json --output plan.json
merlin-phase2-normal-timing-study --request normal-study.json --output measured.json --execute
```

The first command performs no execution or AWS request. Explicit execution
reopens original admission and freezes the selected public compiler revisions.
It records repeated complete compile/Spike/feature and certified RTL costs,
admission/source-freeze costs, and actual calibration/held-validation costs.
Each reduced evaluation retains its 600-second ceiling and owned descendant
cleanup. Wall/CPU time, process-tree RSS and allocated output disk are retained
with exact request, source-revision, partition and domain identities. Repeated
fresh processes do not establish cold/warm cache state; that stays UNKNOWN.
Prior external engine-build costs and whole-model host costs also stay UNKNOWN.

Costs support the existing arithmetic `N*((1-q)*S-E)>M`: `S` is complete ordinary
RTL cost, `E` complete cheap cost, `q` the retained RTL fraction, and `M` setup,
calibration and held validation. Missing denominators keep break-even UNKNOWN.
The study returns the original held validation verdict; it creates no final
acceptance. An unavailable original selected-engine/process or compiled-artifact
join leaves qualification UNKNOWN while preserving the complete cost summary
and every retained profile. Resolve that source prerequisite before exposing a
fit. Changed source, selected products or successful process receipts still
refuse through their original integrity checks. No live costs or screening
qualification have been established by the synthetic routing tests.

Trusted in-process orchestration can install the live returned calibration
through `policy.install_screen(owner)` before creating its queue. Saved reports
cannot reconstruct that owner. Optional original native runtime/index owners
may be installed separately before queued work; absent or unsupported owners
keep fresh execution. Calibration and final/protected evaluation require fresh
observations and do not use historical timing as independent replicates.

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

Queued normal corpus feedback rebuilds the complete kernel and readback harness
before observing artifact equality. Its job-local observer binds the complete
ELF, object, harness, numerical inputs, readback selection, known build tools,
selected engine/process source and effective environment, then reopens the
original successful Python dispatch. Changed or uncertain artifacts execute.
The queued route bypasses the historical prebuild LLVM measurement cache;
the original PME final phases also execute every roster member freshly.

**Unchanged artifacts currently execute too.** The existing frozen package
sandbox inventories selected tool trees, but records system mounts only as a
description. Moreover, its boxed entrypoints govern compiler package execution;
the normal native simulator launches outside those mounts. Sound execution
reuse needs the original native owner to adopt verified frozen loader/library
and system runtime trees. Adding a cache input cannot establish that adoption.
The current observation reports this exact closure gap as UNKNOWN and creates
no reused cycle count, numerical acceptance or extra independent replicate.
There is no parent-owned cross-job/worker observation transport. Selective
execution reuse remains incomplete until that runtime mechanism exists.

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
