---
title: Run bounded Phase 2 search from the Phase 1 compiler
kind: guide
status: current
owner: experiments
last_verified: 2026-10-11
related: [aws_phase0_phase1, phase2_cpu_timing_study, phase2_component_search, neutral_runtime_tooling]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_policy.py, packages/merlin-experiments/src/merlin_experiments/phase2/search_configuration.py, packages/merlin-experiments/src/merlin_experiments/phase2/search_resource_launch.py, packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_pool.py, packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py, packages/merlin-experiments/src/merlin_experiments/phase2/functional_guard_execution.py, packages/merlin-experiments/src/merlin_experiments/phase0/public_programs.py]
---

# Run bounded Phase 2 search from the Phase 1 compiler

Phase 2 uses the original Phase 1 target, descriptor, public contract and ISA,
toolchain, neutral runtime, corpus and functional compiler fork. Original
functional admission and final correctness gates remain in force. The separate
component-runtime qualification ledger is not a prerequisite.

Follow [Phase 0 and Phase 1 preparation](aws_phase0_phase1.md) first. Record
delivered Git revisions, sealed release digest, descriptor digest, functional
run ID and compiler digest, selected RTL facts and timing certificate. Reopen
those same files on AWS. Source publication is not worker qualification.
Protected validation programs and reference compiler answers are not author inputs.

## Public performance inputs

Normal Phase 0 can retain complete existing public programs through
`performance_source_programs`; the [selection example](../../examples/gemmini/phase0/recipe-public-programs.yaml)
preserves their original loaders and profiles. Generation retains graph, host
operations, references, source pins and member purposes. Boundary annotations
come from public shapes and selected facts; unknown physical mappings remain
unknown. Review purposes, semantic coverage and exclusion guards before sealing.
A source packaging probe does not qualify a release, compilation or timing.

Phase 1 owns functional completion. Phase 2 may include larger public members
and connected graphs from that same sealed closure. Keep complete public programs
separate from protected validation. Member counts do not establish coverage.

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

For bounded final functional grading, leave `MERLIN_MODEL_BUDGET_S` unset.
The original inline whole-model grader carries the live package and resource
owners through its worker threads. The exec-based model budget child cannot
carry those owners and is refused in this bounded final guard.

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
Qualified models screen only inside their measured domain. Unknown predictions
and larger regimes require selected CPU RTL observations. Measure cold/warm costs
and throughput in AWS with the [timing study](phase2_cpu_timing_study.md), including
setup and final checks.

The queue ranks complete correct comparable measured feedback with original
certificate and repeat ranges. Predictions and partial rosters cannot select a
winner. Global and host improvements require connected public programs and
measured complete costs; contraction timing alone cannot prove whole-model gains.

Stop at the exploration budget, job limit or three completed diverse batches
without a one-percent improvement. These decisions do not prove optimality.
Join authors, stop edits, run `search-finalize`, then `search-select`. Run original
required feedback and analysis on selected bytes. Final functional qualification
and the complete final roster execute freshly. FPGA validation follows compiler
freeze and separate authorization.

## Re-emission and remaining reuse limit

Queued contenders re-emit through the original build path and bypass the legacy
LLVM memo. The observer compares fresh complete artifacts, inputs and known
dependencies. Native simulation lacks a complete immutable loader/library
identity: **execution reuse is not enabled**. Changed, unchanged and uncertain
observations execute. No cache speedup is claimed, and this creates no extra
Phase 1 or Phase 2 launch prerequisite.

Repository acceptance covers ordinary-route integration, source gates and focused
revision/context/resource/final-roster tests. AWS acceptance also requires actual
sealed inputs, functional baseline, native engine readiness, sandbox canary,
allocated resource boundary and measured tool costs. Synthetic routing tests
establish wiring; target performance and convergence require the actual experiment.
