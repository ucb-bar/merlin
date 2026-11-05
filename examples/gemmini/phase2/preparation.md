# Prepare a bounded performance study

Preparation does not start an author, submit a board job, or qualify a timing
model. Preserve the functional compiler and its original qualification limits.
Use the [shared workload protocol](../protocol/README.md) and a new private copy
of [the run manifest](../protocol/run-manifest.template.yaml). Do not insert
held-out model dimensions into iteration-capsule generation.

## Reuse observations before collecting more

Keep only fresh public gSIM labels with the original source, executable,
simulator and kernel-window identities. Certification-cache hits are not new
timing observations. Join the original Spike and gSIM execution records: sharing
a kernel object does not mean that their wrapper ELFs are identical. Preserve
both ELF hashes and the selected timing boundary.

Use `merlin.perf.fast_estimate_validation` and `component_screen` for diagnostic
fits and validation. Aggregate repeated observations of the same executed
program/workload before fitting. Hold all versions of a capsule together, and
merge groups that share the exact source interface. Otherwise, renamed copies of
one source can leak into both training and validation. Keep out-of-range
predictions UNKNOWN and preserve refused, negative or unidentified fits. Do not
choose features from held cycle labels or relax validation thresholds afterward.

A fit over sanitized scalar projections is an offline diagnostic. The ordinary
live calibration owner still reopens original execution evidence before issuing
an empirical screen. Its fitted coefficients are not physical service prices.
Static instruction sites cannot substitute for dynamic counts, occupancy,
dependency stalls or memory traffic. Missing annotated Spike observations remain
missing even when ordinary Spike executions succeeded.

Audit measured coverage separately from planned coverage. A deferred performance
ledger is not a prepared or measured corpus. Compare M, K and N jointly against
one actual same-kind member; three independent marginal maxima are insufficient.
This is a performance-domain test, not evidence that model lowering is wrong,
and a gap does not imply that every model group needs a new probe. Keep the
model-specific audit private. Extend independent PyTorch iteration workloads
from hardware mechanisms and capacities; use typed MLIR when the framework
cannot express the required interface. Keep accelerator-containing work only.

## Propose at most eight fresh probe measurements

First use the existing selector without running a simulator:

```sh
python -m merlin_experiments.timing_probes select \
  --target-hardware gemmini --profile "$PROFILE" \
  --facts "$FACTS" --contract "$CONTRACT" \
  --labels "$ORIGINAL_PUBLIC_TIMING_ARCHIVE" \
  --domain "$INDEPENDENT_PHASE2_LEDGER" \
  --max-probes 4 --out "$PREP/selection.json"
```

The archive argument is the original protected public-label archive, not a
repackaged public JSON projection. Pin the selected facts, timing-space/backend,
profile, source and hardware identity. `--max-probes` limits distinct requests;
four requests with two deterministic repeats permit eight fresh gSIM
measurements. An imported selection above that limit is refused before measuring.

Review this small request set before granting execution:

| Priority | Distinct requests | Purpose |
| --- | ---: | --- |
| 1 | At most two full/partial requests chosen for unresolved required classes or rank | Separate setup, preload/reuse, transfer and compute terms that currently move together. Require the selector's actual rank gain. |
| 2 | At most two independent capacity-regime requests | Exercise joint contraction extents near a selected operand-store boundary, rather than another single-axis sweep. Inspect the realized geometry and bounds. |

The existing quantity-range selector can choose MAC/read/write extrema; those
scalar ranges alone do not prove joint M/K/N coverage. If its backend cannot
realize the required independent geometry, leave that request unmet and use the
ordinary iteration-workload/corpus producer. Do not relabel a small probe as a
model-scale witness. If only four requests cannot identify the selected profile,
retain the missing requirements; the cap is a budget, not a waiver.

An operator may approve a **separate one-hour preparation envelope**, including
builds, paired functional execution, gSIM and cleanup. Use serial probe work with the entire owned subtree limited to one CPU,
two repeats, and an owned absolute deadline with a cleanup reserve. This also
bounds aggregate CPU time; wall time alone would not bound a parallel subtree. Estimate
feasibility from retained build and active-simulator times, not target clock or
total cycle count alone: startup dominates small jobs, and a large outlier can
consume most of the envelope. Clip each attempt to the remaining allowance and
stop scheduling when its conservative estimate no longer fits. Per-replicate
`--timeout` is not a campaign-wide deadline. Cached labels remain cached.

Only after this review may the existing `timing_probes run --selection ...`
route execute the admitted set. Retain both deterministic replicates, actual
ownership/cleanup records and `recheck.json`. Missing features or remaining gaps
keep the profile unqualified. No command in this document authorizes execution.

## Use measured comparisons when the surrogate is unqualified

Use existing ordinary paired candidate/baseline gSIM feedback for the independent
performance members. Preserve exact-kernel correctness, routing, instruction,
supplier and final regression gates. A cheap unqualified estimate can guide a
diagnostic but cannot select a winner, stop the search, or publish a speedup.
Use occasional bounded full-chain measurements only under their own admitted
owner; do not turn every candidate into another full-model run.

Prioritize conservative measured cycles saved divided by evaluation cost, with
explicit independent-corpus shares and family diversity (`component_screen` or
the ordinary search coordinator). A large measured saving beats a large ratio
on a negligible member. A compute-floor gap is optimistic headroom, not a
measured saving. Do not use held-out model call frequencies as tuning weights.

Compute floors require explicit arithmetic sites and multiplier counts from the
selected facts, not mesh geometry alone. Movement floors additionally require
the relevant physical interface width and compulsory traffic across it. Logical
tensor bytes are not measured DMA bytes. Missing traffic, overlap or host costs
remain UNKNOWN; a compute-only floor cannot establish whole-program attainment.
Use `capsule_roofline`, `roofline_priority`, and the existing matched achievable
rate selection without substituting a structural peak for an empirical ceiling.

The retained simulator uses a 30 MHz architectural denominator. The stored FPGA
smoke's 110 MHz host clock and approximately 109.7 MHz effective emulation
throughput describe the host/emulator rate; they do not select a new 110 MHz
target clock. Keep the architectural denominator at 30 MHz pending exact
AGFI, FIRRTL, DTS, memory-system and timing-scope correspondence. Host runtime,
emulated target cycles and target-equivalent seconds are separate quantities.
Do not scale old target cycles or target-equivalent seconds by 110/30. Any
simulator-to-board transfer needs matched observations and held validation.

## Declare finite stopping before authoring

Suggested values for a new, explicitly approved study are **three hours of
authoring**, a **four-hour total owner deadline** including one hour reserved
for drain/final verification, and a separate preparation envelope above. Existing
`SearchSettings` can express this as `soft_seconds=14400`, `hard_seconds=14400`,
`final_reserve_seconds=3600`, `minimum_improvement=0.01`, `plateau_batches=5`.
The deadline starts once; checkpoints do not reset it.

The plateau means five completed, diversified, fully measured batches with no
conservative improvement of at least 1%. Incomplete, timed-out or incomparable
rosters do not count. Keep the incumbent and reopen repeats through the existing
coordinator. This scoped heuristic does not prove global optimality or authorize
a final performance claim. Missing floor evidence cannot trigger attainment.

Record an explicit proposed cap of **100,000,000 provider input-plus-output
tokens**, including cached input, all descendant actors and compaction traffic.
Use existing AET/Phase 2 telemetry and deduplicate actual response records;
reasoning is already in output and cached input already in input. Do not price
subscription tokens as money spent. The existing CLI's wall/tool-call limits do
not by themselves enforce an all-actor token ceiling. Bind a monitored token
stop before launch, or leave token-budget readiness blocked; an unenforced
manifest field is not an operative cap. Interrupted/unexposed usage is unknown.

These are proposal values, not inherited launch authorization. Freeze the chosen
limits, objective roster, weighting, timing policy and final evidence requirements
in the resolved definition and ordinary run manifest before any paid authoring.
