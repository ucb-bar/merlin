# Original per-record count intake

`spike-timing archive-count-plan` is an explicit v3-archive planning operation.
It does not execute Spike, fit prices, install a model, issue qualification, or
change the frozen483-wrapper/473-review intake or dc413 model. All old archive
commands and schemas retain their original behavior.

The input must be the original producer's immutable **training archive**, with
its externally recorded SHA256. A sanitized JSONL projection or saved inventory
alone is not accepted by this CLI. If the original public archive has not yet
been exported, its original producer provides the host-only operation:

```sh
python -m merlin_experiments.phase1.capsule_timing_archive \
  "$ORIGINAL_RUN" --training-out "$TASK_DIR/original-training.tar"
```

The producer reopens original records and artifact identities. Preserve its
archive SHA and original run/source provenance. Exporting does not remeasure
anything; no guest execution or formal certificate follows.

```sh
spike-timing archive-count-plan \
  --archive "$TASK_DIR/original-training.tar" \
  --archive-sha256 "$ORIGINAL_TRAINING_SHA256" \
  --output "$TASK_DIR/fresh-count-intake-plan.json"
```

The CLI freshly streams/hashes every tar member and checks original source,
caller, result, console, ELF and training eligibility through archive-inspect.
It then emits all original record IDs, including opaque non-public refusals and
ineligible public records. It selects the original gSIM ELF for future counts.
A separately wrapped functional sibling remains provenance and cannot silently
substitute for that ELF, its input context or counter-PC review. Existing
per-record counter-review pins are retained as evidence references; they are not
interpreted as new count reports. Every absent count or timing value is null,
and every eligible request remains UNKNOWN until separately checked execution.

Without an externally reviewed archive binding, hardware/window/domain and source
split fields remain missing. The plan preserves complete original timing-selection
and simulator pins, source-interface, command-buffer, original caller and console
identities. It never invents a scope review from a measured cycle value.

Use `--binding "$REVIEWED_BINDING"` only for the original reviewed archive-prepare
binding. The implementation calls archive.prepare unchanged, retaining its full
roster/repeat obligations, domain, input/counter checks and train/calibration/test
requirements. It additionally binds source/source-group identities to original
input-interface bytes and co-locates original join-group variants across splits.
The returned `observation_plan` is exactly the original archive.prepare output;
fresh original reports later enter through archive-attach, not a new receipt API.
A missing per-record scope still forbids attaching counts for that record.

`source_groups` uses exact source-interface byte identity; `original_join_groups`
retains the producer's separate grouping identity. Both constrain splitting.
For current V26 the159 rows contain145 original join groups but only100 distinct
source-interface byte groups. Compiler versions and repeated executions must not
be presented as additional independent source groups. Neither count alone proves
adequate held-out support. Current raw measurements are all from earlier compiler
snapshots, so no source qualification transfers to the final V26 package.

Before execution an original independently checked observer/runtime owner,
original per-record counter window, exact ELF compatibility and ordinary bounded
shared resources remain required. Before ranking, original counted outputs,
held-group validation and any enabled capacity category checks remain required.
`calibration_validation` already supplies source/shape-derived paired-schedule
requests for out-of-domain regimes; this planning path does not waive or run them.
No asynchronous/spill/attention/KV timing domain is enabled by making this plan.
# Joining independently retained public archives

`merlin_timing.archive_union.join((first_dataset, second_dataset))` retains all
original cases, records, splits and missing observations. Each component must
come from the original archive prepare/attach route. Exact model, hardware,
window, feature schema and public profile must match. Duplicate executions,
record IDs, source-group leakage and changed source families refuse. This pure
operation neither fits parameters nor qualifies timing.

For public diagnostic queues, the optional host selection is
`NativeArchiveScreenSelection(..., additional_archives=(AdditionalArchiveEvidence(
archive, archive_sha256, binding_path, observations_path),))`. The declaration is
data, not a saved owner. `prepare_public_native_exploration` reopens every original
archive and report, runs unchanged held gates over the complete union, rechecks
every raw training/held activity field, and retains all original source catalogs.
The live Count and native-semantics capabilities remain required. Its explicit
V4 record names the additional archive digests; default selection records remain
unchanged. The formal ordinary-screen installer refuses this new diagnostic-only
selection rather than implicitly extending its original admission.

A global held PASS does not qualify every component, shape or mechanism.
Retain per-archive/regime residuals and refusals separately. Current source
geometry, complete numerical execution, actual raw counts and original semantics
must still fit the selected domain. An unsupported current program remains
UNKNOWN; historical cycles never become fresh measured cycles or promotion.
