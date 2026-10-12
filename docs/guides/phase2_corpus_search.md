---
title: Run bounded Phase 2 search from the Phase 1 compiler
kind: guide
status: current
owner: experiments
last_verified: 2026-10-12
related: [phase0_phase1_setup, phase2_cpu_timing_study, normal_search_feedback, neutral_runtime_tooling]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/candidate_observations.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/candidate_census.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_policy.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_fast_feedback.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_queue_calibration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/queued_search_actions.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_program_measurement.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_configuration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_resource_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_pool.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/functional_guard_execution.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/public_programs.py
  - src/merlin/runtime/complete_call_timing.py
  - src/merlin/targetgen/public_program_timing.py
---

# Run bounded Phase 2 search from the Phase 1 compiler

Phase 2 uses the original Phase 1 target, descriptor, public contract and ISA,
toolchain, neutral runtime, corpus and functional compiler fork. Original
functional admission and final correctness gates remain in force. The separate
component-runtime qualification ledger is not a prerequisite.

Follow [Phase 0 and Phase 1 preparation](phase0_phase1_setup.md) first. Record
selected Git revisions, sealed release digest, descriptor digest, functional
run ID and compiler digest, selected RTL facts and timing certificate. Reopen
those same files on the execution worker and qualify its runtime.
Protected validation programs and reference compiler answers are not author inputs.

The [search and feedback design](../design/normal_search_feedback.md) describes
the implemented diagnostic, provisional exploration and measured selection
routes and their qualification requirements.

## Public performance inputs

Normal Phase 0 can retain complete existing public programs through
`performance_source_programs`; the [selection example](../../examples/gemmini/phase0/recipe-public-programs.yaml)
preserves their original loaders and profiles. Generation retains graph, host
operations, references, source pins and member purposes. An explicit
`public_joint_boundaries` selection also constructs connected source cuts with
joint dimensions derived from public typed shapes and selected resource facts.
The supported source subset includes contractions, movement and reductions;
only live outputs leave a cut. Fresh normal capture supplies its references.
Logical tensor pressure does not prove simultaneous physical placement or
lifetimes. Unsupported source semantics or unavailable boundaries remain
UNKNOWN. Review purposes, semantic coverage and exclusion guards before sealing.
A source packaging probe does not qualify a release, compilation or timing.

Phase 1 owns functional completion. Phase 2 may include larger public members
and connected graphs from that same sealed closure. Keep complete public programs
separate from protected validation. Member counts do not establish coverage.

For a focused development check, the queue exposes
`evaluation-submit-functional-selected` with `revision_id` and `members`.
Members are comma-separated original public `family/capsule` identities.
The fixed evaluator compiles and runs Spike for the baseline and candidate on
those members only. Feedback retains the complete original roster, with null
observations and a pending reason for each unselected member. It runs no RTL.
This is an explicit diagnostic selection, not evidence that a source edit
cannot affect other members. Complete compile/Spike observations still precede
empirical calibration and complete exploration ordering; full measured and
final correctness gates remain required. Unknown dependency impact must broaden
checks. Partial results cannot establish promotion, convergence or a model fit.

## Candidate inspection during Phase 1

In the original author workspace, use its staged broker client:

```sh
python agent_selfcheck.py --candidate-diagnostics --capsules all
python agent_selfcheck.py --sim spike --capsules all --dynamic-census
```

These are separate public requests. The first follows original integrity and
emission entrypoints and needs no successful Phase 1 baseline. It observes own
shapes/dtypes, numeric declarations, source demands, placement/refusal joins,
representations, host CFG work, allocations, artifact identities and available
build options. It runs no tensor/oracle evaluation and returns no numerical or
timing verdict. Unsupported products and source/emission joins stay UNKNOWN.
Literal tensor payloads, expected/reference values and private paths are withheld.

The second keeps the original numerical gates and adds bounded descriptive
activity only after a passing numerical screen. It requires one actual linked
own executable and the original narrowed sandbox; missing joins remain UNKNOWN.
Trace limits and the parent deadline bound collection. Truncation is UNKNOWN,
and unresolved cleanup aborts the job while retaining ownership. Linked-entry
instruction counts, accelerator multiplicity and operand/local-address requests
are not physical traffic, cache behavior, cycle prices or array utilization.
Spike supplies no Rocket timing; a conditional model requires independently
established prices and coverage in its selected domain.
Outside-entry/callee work is separate and unpriced. Neither request completes
Phase 1 or grants permission to enter Phase 2.

## Worker and configuration

Allocate resources before preparing search. For example, 32 assigned CPUs with
four author CPUs can run three roster jobs with seven member workers each:
21 concurrent member simulations. This is an explicit allocation, not spare
capacity inferred on a shared host. Include setup, builds and final checks.

Construct `prepare_ordinary_search_policy` from the original functional run,
frozen corpus, functional fork, edit contract, facts, timing certificate and
ordinary feedback services. Prepare selection data from that live policy:

```python
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.search_configuration import prepare_search_configuration
from merlin_experiments.phase2.search_coordinator import SearchSettings

# policy and boundary are live original owners on the allocated worker.
selection = prepare_search_configuration(
    policy, configuration_path,
    resources=AssignedResources(
        cpu_ids=assigned_cpu_ids, ram_bytes=128 << 30,
        disk_bytes=512 << 30, engine_slots=21,
        author_cores=4, reserve_fraction=0.2, require_hard_limits=True,
    ),
    job_resources=JobResources(
        cores=7, ram_bytes=24 << 30, disk_bytes=32 << 30,
        engine_slots=7, files=100_000,
    ),
    settings=SearchSettings(
        beam_width=3, soft_seconds=14_400, hard_seconds=28_800,
        final_reserve_seconds=2_700, minimum_improvement=0.01,
        plateau_batches=3,
    ),
    source_bytes=256 << 20, source_files=20_000,
    retained_source_bytes=2 << 30, max_pending=6, max_jobs=96,
    author_agents=3, external_boundary=boundary,
)
```

`prepare_external_resource_boundary` verifies actual externally provisioned
cgroup and filesystem controls. JSON cannot reconstruct that owner. The normal
CLI verifies hard selections against the live enclosing worker before authoring.
Readonly root/device metadata, finite cgroup CPU/RAM limits, disabled swap,
unprivileged membership and one capped backing filesystem are required. Writable
mounts, output handles and tool scratch must satisfy the same boundary. Storage
limits cover allocated backing capacity; sparse logical byte quotas are unknown.
Unbound external native services and container supervisors are unsupported.

Use the original normal authoring CLI/controller selections, adding
`--search-configuration` and `--compiler-edit-contract`. Select one round and stage
and round timeouts covering exploration and final reserve. Use the same Codex
binary/configuration and original sandbox canary qualified on that worker. The
configured author count does not replace this check. Preparation allocates no
machine, cgroup, quota, agent or simulator.

For object reuse, add optional `build_runtime` when preparing the same search
configuration. Select the original Phase 1 compiler executables, bootstrap and
complete library/sysroot lookup roots. These operator inputs remain outside
author grants; exclude reference compilers, protected programs and grader trees.

```python
build_runtime = {
    "tools": original_compiler_paths,
    "readonly_roots": original_lookup_roots,
    "bwrap": original_bwrap_path,
    "snapshot_files": 100_000, "snapshot_bytes": 8 << 30,
    "cache_entries": 512, "cache_bytes": 16 << 30,
}
# Pass build_runtime=build_runtime to prepare_search_configuration above.
```

Paths are absolute strings. A lookup root is a string or a
`{"source": absolute_source, "guest": absolute_lookup_path}` mapping.
The toolchain is copied once, without mutable hard links or per-job copies.
Snapshot, cache and retained source ceilings fit the original reserved disk
fraction; output ceilings come from the job grant. The normal round constructs
live owners before queue admission. Declarations cannot authenticate a compiler
process. Unsupported commands stay fresh; missing libraries report the original
build failure. Final verification precedes owner release; unknown job cleanup
retains ownership.

For the example eight-hour hard budget above, launch the normal module on the
allocated AWS worker:

```sh
python -m merlin_experiments.phase2.authoring_cli \
  --functional-runs-root "$P1_RUNS" \
  --functional-run-id "$P1_RUN_ID" \
  --functional-submission-sha256 "$P1_COMPILER_SHA256" \
  --source-root "$MERLIN_SOURCE_ROOT" --stage-root "$P2_STAGE" \
  --contract-root "$P2_CONTRACT_ROOT" --descriptor "$P1_DESCRIPTOR" \
  --gsim-certificate "$TIMING_CERTIFICATE" \
  --gsim-certificate-sha256 "$TIMING_CERTIFICATE_SHA256" \
  --rtl-facts "$P1_RTL_FACTS" \
  --telemetry-price-table "$TELEMETRY_PRICE_TABLE" \
  --search-configuration "$SEARCH_CONFIGURATION" \
  --compiler-edit-contract "$COMPILER_EDIT_CONTRACT" \
  --suite "$SUITE" --model "$CODEX_MODEL" --effort "$CODEX_EFFORT" \
  --codex-binary "$CODEX_BINARY" \
  --rounds 1 --wall-budget-seconds 28800 \
  --round-timeout-seconds 28800 --tool-timeout-seconds 600 \
  --max-tool-calls 192 --families all --capsules all
```

`P2_STAGE` is a fresh output directory under the configured `out` root.
`P2_CONTRACT_ROOT` selects the reviewed performance guidance contract and frozen
public development corpus; it is not the target capability contract. Every
Phase 1 identity and selected file comes from that original admitted run.
Keep its staged target contract and facts environment selections consistent
with `P1_DESCRIPTOR`. The CLI rechecks configuration identities when installing
the queue, before compiler authoring.

The inline whole-model grader is selected when `MERLIN_MODEL_BUDGET_S` is
unset. A positive budget uses the original sealed child transport when the
exact original compiler-profile executor is active: it reconstructs the frozen
profile, package sandbox and any selected kernel resource boundary in the
child, then rechecks them after completion. Foreign or incomplete owners refuse;
an alternative component executor is not reconstructed by this route. See
[ordinary compiler execution](../design/public_compile_only_feedback.md#ordinary-compiler-execution-with-the-common-profile)
for the supported scope. The budget changes process supervision, not numerical
or certificate requirements.

## Iteration and final selection

One coordinator integrates up to three authors' disjoint compiler changes.
Freeze each integrated source with its hypothesis and edit family, then submit
it. Source bytes are retained once per content identity; inputs are shared
readonly. CPU and engine grants cap member fanout. Authors may edit live source
while issued jobs evaluate retained bytes.

Compilation and Spike supply static and numerical observations. Derived
roofline/headroom reports identify potential time savings. Search status orders
the incumbent's members by potential savings using the same weights as compiler
selection and preserves all observed repeat ranges. Matched empirical throughput
targets and hardware lower bounds have distinct meanings; neither guarantees
reachable improvement. Members without a bound remain visible with their
complete measured cost. Evaluation costs and savings per evaluation second stay
unknown until complete matched tool costs are observed.
Submit `corpus-compile-spike-feedback` for the cheaper original compile/Spike
route; each arm returns fresh compiled diagnostics and attempts the bounded
own-executable census after its original L0/L1/L2 gates. Mandatory RTL tiers
remain pending. `search-status` keeps a provisional shortlist and negative,
unchanged, duplicate or failed history separate from the measured incumbent.
For eligible normal public model members, the fixed cheap producer calls
`public_program_measurement.selection(capsule, policy.feedback_evaluator.rtl_identity,
simulator="spike")`. This requires the explicit L2 ceiling and sets
`complete_call_timing=False`; it reopens the original public source basis,
facts and board configuration. The native candidate runs Spike numerical-only
through the same original source/ABI/ISA/full-output gates. Its
`candidate_numerical_screen` may report observed L0/L1/L2 with null timing, while
the complete grade remains incomplete and required L3 remains pending.
Native compiled observation uses the original receipt's explicit source,
candidate emission, build products and ELF, reopening the actual
LLVM/object/link producer chain. `artifact_join.kind=original_native_model_products`
returns source/command-buffer/producer hashes; raw products and paths stay private.
Calibration matches those exact fresh artifacts to the original selected RTL
invocation, engine and full console. Missing/changed joins never become labels.
Qualified models screen only inside their measured domain. Unknown predictions
and larger regimes require selected CPU RTL observations. Physical stage costs
stay UNKNOWN without independent public evidence. Use the normal command in the
[timing study](phase2_cpu_timing_study.md) to measure complete route, setup and
calibration costs on AWS. Its default is a nonexecuting plan; execution is
explicit. Repeated processes alone leave cold/warm cache state UNKNOWN.

The queue ranks complete correct comparable measured feedback with original
certificate and repeat ranges. Predictions and partial rosters cannot select a
winner. Global and host improvements require connected public programs and
measured complete costs; contraction timing alone cannot prove whole-model gains.

Invoke these named actions through the existing staged feedback broker; values
are issued identifiers and exact frozen public names, not executable selections:

| Action | Arguments and use |
| --- | --- |
| `revision-freeze` | `parent_revision=<issued-or-none> hypothesis=<description> edit_family=<description>`; freeze integrated source |
| `evaluation-submit` | `revision_id=<issued> kind=corpus-compile-spike-feedback`; complete cheap roster |
| `evaluation-status`, `evaluation-cancel` | `job_id=<issued>`; poll or cancel an owned job |
| `search-resume-parent` | `revision_id=<issued>`; join authors first, restore the recorded parent, then branch edits from those bytes |
| `evaluation-submit-selected` | `revision_id=<issued> members=family/capsule,...`; decisive public RTL only, with other rows unmeasured |
| `evaluation-submit` | `revision_id=<issued> kind=corpus-rtl-feedback`; full measured promotion evidence |
| `search-calibrate` | `revision_ids=<issued_a>,<issued_b>,...`; install screening from completed own observations |
| `search-status` | Inspect both portfolios, current original ranking, priorities, budgets and stopping reason |

For calibration, collect complete cheap and complete RTL jobs for at least two
distinct issued alternatives. Finish or cancel every outstanding job so all are
terminal. Then invoke `search-calibrate` **before `search-finalize` and its final
drain**, while the exploration budget still admits fitting. Selected-only RTL
does not provide complete calibration labels. The public calibration/held
partition was frozen before measurements; the action launches no simulator,
reopens original artifacts/receipts and retains the original ranking, error,
coverage and minimum-count gates. UNKNOWN qualification leaves fresh measured
fallback and does not block ordinary search. Up to three installation attempts
are allowed. Saved qualification JSON supplies no live owner.

Parent restoration changes exploration source only. Selected-member timing and
the provisional shortlist cannot promote, replace the incumbent objective or
establish a plateau. Measure the unchanged starting compiler through the full
RTL kind as well, so final selection has a measured baseline fallback.

Stop at the exploration budget, job limit or three completed diverse batches
without a one-percent improvement. These decisions do not prove optimality.
Join authors, stop edits, run `search-finalize`, then `search-select`. Run original
required feedback and analysis on selected bytes. Final functional qualification
and the complete final roster execute freshly. FPGA validation follows compiler
freeze and separate authorization.

## Complete public call timing

An eligible original public whole-program member uses the same native candidate
grade, source/facts selection, complete numerical comparison and selected gSIM
engine receipt. Its `merlin.public_program_execution.v1` data binds the original
facts file and board configuration. This selection does not admit protected
programs or issue a new engine certificate. The normal projector reopens actual
source/emission, caller/build/ELF, full output console and original engine bytes.
Missing or changed products refuse; output equality alone supplies no timing.

The original target contract may select a reviewed complete-call counter before
sealing. It has no default instruction, clock or cycle unit. The following Python
shows the exact data shape; every named value must come from the reviewed original
counter accessor, ISA and clock selection:

```python
contract["logical_harness"].update(
    cycle_counter_asm=reviewed_counter_asm,
    complete_call_timing={
        "schema": "merlin.complete_call_timing.v1",
        "counter_semantics": selected_counter_name,
        "max_delta_cycles": reviewed_max_delta,
    },
)
contract.setdefault("counter_semantics", {})[selected_counter_name] = {
    "unit": "cycles",
    "width_bits": 64,
    "wrap": "modulo",
    "clock": selected_clock_id,
    "source_pins": [
        {"path": original_counter_source_path, "sha256": original_counter_source_sha256},
        {"path": original_clock_source_path, "sha256": original_clock_source_sha256},
    ],
}
```

`counter_semantics` names an existing reviewed record. `clock` is a nonempty
selected clock identity. Each source pin contains exactly `path` and `sha256`;
paths are absolute canonical regular files without symlink parents. The roster
is nonempty, at most 64 files, each at most 16 MiB, and every hash is rechecked.
Pin the actual counter/accessor, ISA/unit and clock sources needed to review the
declaration. `max_delta_cycles` is an explicit positive integer below `2**63`;
the measured modulo-64 delta must be positive and no larger than it. The plain
counter assembly is bounded and cannot contain quotes, backslashes, newlines or
semicolons. The selected backend includes these clock sources and timer/harness
owners in `build_source_paths()`; retain them in the original sealed build/runtime
source inventory. These are operator-reviewed target data, never an author knob.
An absent selection supplies no complete-call timing; do not infer it from Spike
counts or change the admitted target after Phase 1.

The timer covers **main preparation through completed whole-program outputs and
selected output publication**. It excludes static immutable input staging,
loader/startup, metric formatting, DONE/exit and external parent
transport/readback/decoding. The result retains these exclusions and cold/warm
UNKNOWN. Reduced members retain their original timer boundaries. A complete-call
cycle delta does not establish host/device stage prices, physical traffic,
residency, cache effects or overlap; those require separate independent public
observations. Unsupported coherent-output replay does not qualify this projector.

## Re-emission and remaining reuse limit

Queued contenders re-emit through the original build path and bypass the legacy
LLVM memo. The observer compares fresh complete artifacts, inputs and known
dependencies. A selected immutable build view can reuse verified LLVM object
and stack-sidecar bytes after comparing fresh LLVM, executed flags, ABI,
environment, tools and lookup identities. Publication follows parent cleanup.
Reuse has a distinct receipt and claims no new compiler invocation or timing
replicate. Unknown inputs, native CPU discovery, plugins, profiles and response
files stay fresh. Harness/link and simulator/numerical gates remain fresh;
calibration and final evaluation omit the product index.

Optional original native owners can freeze one bounded readonly
runtime filesystem view, shared across jobs without per-job copies. The live
parent publishes original observations only after actual guardian cleanup and
durable release; failed cleaned jobs discard pending pins. Unresolved cleanup
retains ownership. Saved cleanup reports cannot publish archive entries.

The original engine selection still does not establish reset/seed and non-file
input closure: **simulator execution reuse is not enabled**. Changed, unchanged and
uncertain observations execute. Filesystem closure supplies no timing or replay
qualification. No measured speedup is claimed, and these optional owners create no
extra Phase 1 or Phase 2 launch prerequisite. Calibration and final evaluation
keep fresh observations.

Before launching, verify the sealed inputs, functional baseline, selected engine,
sandbox canary and allocated resource boundary. Measure tool costs on that worker
before using them to plan evaluations. Compiler performance and search convergence
require completed experiment observations.
