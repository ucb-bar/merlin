---
title: Tracking experiments — the dashboard and the watch view
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
related: [experiment_console, storage, reproducibility, artifact_evaluation, phase2_test_justification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/discovery.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/model.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/serve.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/diagnose.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/placement.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/timeline.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/machine.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/__init__.py
  - packages/merlin-experiments/console/internal/ui/model.go
  - merlin/contract/compiler_stages.yaml
  - packages/merlin-experiments/src/merlin_experiments/tracking/terminal.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/terminal_views.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/terminal_probe.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/terminal_metadata.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/records.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/html.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/text.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/__init__.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/views.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/live.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/tail.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/charts.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase0.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase0_html.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/activity.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/activity_html.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase1_detail.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase1_views.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/compiler.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase2_paired.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/phase2_views.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/resources.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/explorer.py
  - packages/merlin-experiments/src/merlin_experiments/cli.py
  - src/merlin/targetgen/target_index.py
  - merlin/contract/storage.yaml
---

# Tracking experiments: the dashboard and the watch view

Two read-only views of a run, built from the records the phase owners already write:

```bash
merlin-experiment dashboard <run_dir>            # one self-contained HTML page for a run
merlin-experiment dashboard --target <target>    # every run of a target, with champion lineage
merlin-experiment watch <run_dir>                # the same summary in the terminal, refreshed
merlin-experiment watch <run_dir> --once         # print once (CI, scripts)

merlin-experiment dashboard --phase0 <dir>       # Phase 0 corpus, requirements and coverage
merlin-experiment dashboard <run_dir> --monitor MONITOR.md --load load.tsv
merlin-experiment dashboard <experiment_root>    # a paired Phase 2 experiment (state/checkpoint.*.json)
merlin-experiment dashboard --target <t> --explorer   # every run, cross-phase lineage, linked run pages
merlin-experiment dashboard --compare RUN_A RUN_B     # two runs of one phase side by side
merlin-experiment dashboard <run_dir> --live --cpus 24-31 --port 8765   # refresh and serve on 127.0.0.1
```

Every form accepts `--operator-private` (see "Private material") and `--out`; all of them except
`--compare` default to the target's dashboard home.

The page is a single HTML file. Its CSS, its SVG charts and its short tooltip script are all inline,
so it opens from disk, through an ssh port-forward, or as a mailed copy, and it never fetches anything.
By default it is written to `out/artifacts/experiments/<target>/dashboard/<run>.html`. A target
view goes to `.../dashboard/target.html`. That home is the storage contract's `experiment-dashboards`
product root (`merlin/contract/storage.yaml`). `--out` writes elsewhere, and `--open` also opens the
page in a browser.

## Experiment command console

The interactive terminal console (`merlin-experiment console`) has its own user guide:
**[experiment_console.md](experiment_console.md)**. It covers setup, finding and browsing experiments,
every panel and key, common debugging tasks, the bottleneck findings and troubleshooting. Notes for
maintainers:

- **Two processes.** The data process (`python -m merlin_experiments.tracking.console.serve`) owns
  every record read. It streams `merlin.console.snapshot.v1` JSON lines (after the first snapshot,
  only the sections that changed) and answers `select`, `inspect` and `refresh` commands on stdin.
  The Bubble Tea v2 frontend (`packages/merlin-experiments/console`) only renders that stream. The
  section contract is the docstring of `tracking/console/sources/__init__.py`, and
  `console/internal/proto/proto.go` mirrors it. Change both together.
- **Sources** are read-only, bounded readers (`io.Reader`: stat-cached documents, a per-refresh
  byte budget, offset tails). A source returns the same section object when nothing changed, so the
  server skips re-encoding it. Fast tier: resources, machine, agents, tokens, jobs, events, timeline.
  Slow tier: capsules, progress, perf, golden (operator only), passes, stages.
- **State** is decided by `discovery.classify` and always carries its basis. An owner record alone
  never decides it.
- **Bottlenecks** are rules in `diagnose.py` over recorded numbers. The critical-path attribution
  comes from `sources/timeline.py`.
- **Footprint.** `placement.py` keeps the console off the CPUs that live experiments hold, derived
  from runner affinity and cgroup containment with no machine-specific configuration. `serve.py`
  throttles itself above 15% of one core.

## Unified animated terminal console (legacy single screen)

```bash
merlin-experiment watch --task /path/to/campaign-task --legacy
# Direct module entrypoint, with the research package installed:
python -m merlin_experiments.tracking.terminal --task /path/to/campaign-task
```

Install `packages/merlin-experiments` into the same environment as Merlin and
run on the Linux worker (for example through SSH). See the
[artifact setup guide](artifact_evaluation.md) for a fresh machine. No browser,
web service or additional terminal package is required. For a plain snapshot:

```sh
python -m merlin_experiments.tracking.terminal \
  --task /path/to/campaign-task --once --no-color --width 160 --height 55
```

`--task` names the managed campaign directory, not `out/runs/.../<run-id>`.
Discovery reads `.tmp/*/process/owner.json` and the exact hash-bound selection
referenced there. `--selection /absolute/path/selection.json` restricts the
matching owner; it does not manufacture a missing owner or bypass a hash check.
If these records are absent or moved independently, `selection_unavailable`
means the viewer cannot identify the campaign, not that zero capsules passed.
Use `merlin-experiment watch /path/to/run --once` or
`merlin-experiment dashboard /path/to/run` for retained run-directory evidence
without those live-owner records. Such static views cannot reconstruct past
per-core utilization. Do not fabricate live ownership for an archived run.

One screen contains the per-core timeline, current utilization and actions, agent
parent/child tree, weakest recorded floor-to-cycle ratios, recent optimization
attempts, capsule progress and current jobs. No view switching is required. Use a
terminal at least **150 columns × 34 rows** to show twenty cores with room for the
summary panels. Smaller terminals report omitted core rows; they cannot show an
unlimited number of rows at once.

Colors and letters identify activity: pink `A` agent, cyan `2` L2/Spike, green `3`
L3/RTL simulation, orange `B` build, purple `V` verification, yellow `C`
calibration, gray `?` unattributed, and `·` idle. Activity labels are observations,
not correctness verdicts. The display follows the task's latest native owner.
`Space` freezes the view while sampling continues; `q` exits and restores the
terminal. Animation refreshes independently of resource polling (about eight
frames per second; polling defaults to once per second). `NO_COLOR` or
`--no-color` disables colors. `--once` prints one frame. `--window 120` selects
the visible history in seconds; there is no invented history before opening.

This view samples `/proc` CPU counters and live thread positions in addition to
reading bounded record tails. CPU utilization is measured independently; thread
placement is sampled, **not an exact core-residency trace**. Exact capsule and
stage attribution requires the newer setup records' native thread identities.
Older PID-only records remain unattributed. Process observations never become
scientific performance evidence, and no evaluator or simulator is launched.

Cycle estimates retain their recorded model status and scope. The displayed
floor/cycles percentage is the structurally bound compute floor divided by the
estimated complete-call cycles; it is **not measured hardware utilization or an
attainable whole-SoC roofline**. Unknown metrics are not zero, provisional timing
is not qualified timing, and proposed optimizations are not demonstrated gains.
Agent lineage is shown from recorded parent IDs; a compiler branch is shown only
when recorded. Missing optimization receipts remain explicitly unavailable.

The unified console also shows numerical executions separately from returned
verdicts. These counts cover unique observed **snapshot/capsule pairs**, not the
current compiler's certified coverage. Its latest-check line identifies the
snapshot and recorded queue and completion wall times. Alerts flag zero observed
numerical executions, repeated structured failures before execution, request
timeouts, and progress records older than self-check responses. Reads are bounded;
absence in this inventory is not proof of campaign-wide absence. A repeated case
failure is diagnostic grouping, not evidence that unrun cases fail. Missing stage
timing remains unknown. These operator alerts never change grading or dispatch.

The class progress display distinguishes the 153 original correctness capsules,
20 calibration candidates and 77 scalability/performance cases when that roster
is selected. L2 and L3 remain separate obligations. A historical union, a current
snapshot count and the diagnostic snapshot/capsule-pair count are not interchangeable.
Performance rounds and roofline/model fields only appear when their underlying
records exist. Missing or stale values remain unavailable rather than zero.

`--operator-private` enables operator details; keep this display outside the
compiler author sandbox. Without it, unqualified private identities and records
are redacted. The console never modifies campaign records and needs no additional
terminal UI dependency.

## What it reads, and what it never does

The evidence views **never measure target performance, grade or build**. Every performance number is a field some owner wrote; the terminal console additionally samples host resource telemetry:

| Phase | Records |
|---|---|
| orchestration | `orchestration.json`, `resolved-plan.json` (state, attempts, plan binding) |
| phase 1 | `qa_history/verdict_*.json`, `plateau.json`, `oot_commits.jsonl`, `freeze.json`, `run_manifest.yaml`, `qa_loop_summary.yaml`, `environment.yaml` (target) |
| phase 2 run | `run.json`, `resumed_seed.json` (store roots, seed lineage), `whole_model_objective_config.json` (bar reference, plateau rule, orientation figures), `iterations.jsonl`, `stage/sessions.json`, `stage/rounds/round_*.round.json` |
| phase 2 store | `<job>/job.json`, `result.json` and its archived attempts (`attempts/<n>/`, legacy `*_attempt_<n>/`), `attribution.json`, `plateau.json`, `solo_streak.json`, `batch_runner.json`, `board_outage.json`, `board_outages.jsonl`, `batches/*/batch.json` |
| target | `out/artifacts/targets/<target>/INDEX.yaml`, the phase run roots, `merlin-experiment runs` |

A few figures are simple arithmetic over those fields, and each one is labelled where it appears:

- the best-so-far line, a running minimum over valid, attributable, first-replicate readings;
- ages and run windows;
- the total over the per-group rooflines a result records;
- the package-authored share, which comes from the owner's own `feedback.package_authored` over the stored routes.

A job's result is the one that stands across its attempts, read by the measured mode's own reader
(`attempts.effective_result`): a re-queued job whose retry was lost to the host still shows the
verdict an earlier attempt reached, and the row names that attempt (`from_attempt`).

If a record is missing, the page says **"not recorded"**. It never shows a zero or a guess in its place. The
"Records read" section at the bottom lists every file consulted. Each file is marked as read, absent,
unreadable, or of an unexpected schema.

## The views

**Phase 1:**

- a stat row: state, latest grade, number of grades, highest tier, plateau sentence;
- score progression: capsules passed and capsules graded, per grade over time;
- a capsules × grades heatmap: the highest tier passed for each passing capsule, and the status
  of each failing one;
- a capsule × tier matrix for the latest grade;
- first-failure planes;
- the failing capsules, with their recorded plane, category and detail;
- the freeze, the formal manifest, the QA-loop summary and the OOT commit log.

**Phase 2:**

- cycles per candidate over time:
  - valid readings, and wrong-output readings marked ✕;
  - the best-so-far line and the run's window;
  - reference lines: the bar (the reference measured on the same machine), a derived roofline when
    results record one, and the config's orientation figures (context only);
  - a reading more than 4× the median is drawn on the top edge, with its value on hover.
- the package-authored priced share, on the same time axis;
- outcomes by class:
  - measured;
  - `correctness`: `MEASURED_INVALID`, the functional gate, or a capsule screen that failed on the bytes;
  - `infra`: a fault the record names as infrastructure's;
  - `refused`;
  - `prohibited_instruction`: the whole-ELF instruction rule;
  - `declined`: the coverage gate;
  - pending and superseded.
- the latest failures, with their recorded reasons;
- the distance to the roofline for each form, when results carry `diagnostics.per_group`;
- the plateau rule and its session trace, the session and round records, and the circuit breaker;
- the board: the queue and the oldest waiting job, open and closed outages, the solo streak and recent batches;
- champion lineage: the seed's resume chain, the ledger's `best` moves, and any `INDEX.yaml` rows that cite the run.

**Target:** the champion lineage from `INDEX.yaml` (phase-0 seal → phase-1 frozen commit → phase-2
best → champion), the phase-2 bests, frozen compilers and releases, and one row per run under each
phase root with its state.

## Liveness: LIVE, STALLED, STOPPED

A run's state comes from its own records:

| State | What the records show |
|---|---|
| `STOPPED` | A phase-2 run whose `stage/sessions.json` records why it stopped (plateau, bar, budget, crash loop). |
| `FINISHED` | A phase-1 run with a `qa_loop_summary.yaml` or a `freeze.json`. |
| `ENDED` | The orchestration recorded a terminal state. |
| `RELAUNCHED` | A later phase-2 run's `run.json` says it was resumed from this one. |
| `STALLED` | None of the above, and the newest measured candidate (phase 2) or grade (phase 1) is older than `--stall-hours` (default 6). With no candidate measured yet, the age counts from the run's start. |
| `LIVE` | None of the above, and within the threshold. |

A process check (a batch runner's pid, an orchestration attempt still marked running) is shown as
the state *at generation time* and never decides the state on its own.

## INDEX.yaml stays current

`out/artifacts/targets/<target>/INDEX.yaml` is regenerated on three events:

- when a phase-1 run freezes (`phase1/feedback/freeze.py`);
- when a phase-2 run's `best` tag moves (`OotLedger.sync`);
- when a champion is exported (`champions.export_champion`).

All three go through `target_index.refresh_for_run`, which only acts on a run under the canonical
`out/runs/<target>/phase<N>/` root. It never raises: a failed refresh is printed and leaves the
index stale, and `merlin-experiment index <target> --check` reports that. Run
`merlin-experiment index <target>` by hand at any time.

## Records written by older runs

Runs written before the current phase-2 layout lack `run.json`. They also have a
`resumed_seed.json` of another schema, whose store is recorded under a different key. The dashboard
does not interpret those layouts. It reports the schema difference in the inventory, and
`--store <path>` names the measurement store when the run's own record does not.

## Phase 0: corpus and coverage (`--phase0`)

`--phase0 <dir>` takes a derivation root (`requirements.yaml`, `synthesis.yaml`, `derivation.json`), a
generation run (`capsules/MANIFEST.yaml`, `capsules/<category>/<name>/capsule.yaml`,
`coverage/{generation,phase1-capsule-coverage,phase2-capsule-coverage}.json`), a bare corpus, or a
directory holding several of these (each derivation gets its own section, and a table compares them).
The view shows:

- **Capsule inventory.** A family → op treemap, then counts by family, op, form, geometry stratum,
  epilogue, dtype, tier, category, label and performance family. The written corpus is used when
  present; otherwise the capsules each derivation planned. A stratum the capsule does not record is
  classified from its `M/K/N` by `merlin.capture.shape_taxonomy` and marked `*`.
- **Requirements against coverage.** The cells matrix (family × dtype/alignment) and one table per
  axis: shape geometry, conv windows, epilogue stages, groups, carried state and adjacency scope. Each
  required key gets a Phase 1 cohort column and a Phase 2 cohort column from the capsule-coverage
  outputs; a recorded gap is red, a covered key green, and a key with no coverage output says "not
  recorded". The typed required instances are drawn as application × signature and application ×
  semantic-family matrices, with the operations each cohort was measured to witness.
- **Cohorts.** MANIFEST `phase_corpora` Phase 1 / Phase 2 / diagnostic members side by side, by
  category; the performance families; each cohort's coverage status and blockers; and the performance
  form classes against `form_perf_coverage`, with the missing classes and unrepresented strata.
- **Distributions.** Required conv windows by observed regions, the windows and shapes of the
  capsules (M against N, log scales), and the strata.

## Phase 1: agent activity, timeline and compiler evolution

A Phase 1 run page adds these sections to the ones above:

- **Agent activity.** This section is read from `rounds/round_NN.codex_events.timestamped.jsonl`, the
  provider's arrival-stamped stream. It shows:
  - what the agent is working on now: open tool calls, its latest message, its reasoning summary and
    its plan;
  - a feed of messages, edits, self-checks, tests, builds and every error;
  - every command with its exit code, the files it edited, and per-round calls and tokens.
  Command labels (self-check, test, build, read, verdict wait) come from the words a command runs.
  They are a reading aid and never feed a grade.
- **Timeline.** One Gantt holds the agent's turns and tool calls by kind, the grades, the self-checks
  from `selfcheck_log.jsonl`, simulator jobs and self-check requests per engine, and the freeze or
  operator seal. Jobs come from the brokers' `.qa_channel/` (the live workspace, else the run's
  `agent_evidence_snapshot/`).
  - Running spans have a dashed outline, and a "now" line marks generation time.
  - A job's end is its marker file's write time.
  - Self-check offsets are anchored at `qa_loop_state.yaml` `cumulative.started_at`.
  - When the run is longer than three hours, a second chart shows the last two hours.
- **Compiler evolution.** For each graded snapshot in the run's `oot/` history (read with
  `git ls-tree`/`cat-file`, no checkout) and the final `submission/`, the page lists:
  - passes, dialects and ops, parsed structurally: TableGen `def`s, Python classes through `ast`, and
    C++ `getArgument()` names;
  - manifest entry points, commands, components and optimization surfaces;
  - LOC by file;
  - what changed since the previous checkpoint.
- **Pass rate by capsule family** over the grades. A capsule's family comes from its own
  `capsule.yaml`, found under the run's corpus roots (`--corpus` adds more).
- **Tokens and time**, from `cost_time_toolcalls.yaml`, `timing_detailed.json`, `qa_loop_state.yaml`
  and the stream's own turn usage.
- **In the terminal**, `watch` adds an `AGENT` line from the same stream: the last event, open tool
  calls and the latest message.
- **Monitor notes** (`--monitor PATH`). The file has a `## <utc>` heading per check, and the first
  `STATUS: OK | WATCH | STUCK` line of each check gives its status. The page shows the latest note,
  a status strip and every check.

## Phase 2: paired trials × members × cohorts

Point the dashboard at a checkpointed experiment root (`state/checkpoint.*.json`). The chain names
each trial's authoring stage and each measured cell. For a cell that is still running, the chain does
not name it yet; `--measurement-root` and `--stage-root` find it. The page shows:

- the checkpoint chain against the expected stages, and the speedup by family once sealed;
- the cells grid, one table per cohort (`tuning`, `held_out`, `held_out_form`), member × trial. Each
  cell shows:
  - the candidate/baseline cycle ratio (geometric mean over replicates; below 1 is faster) and the
    speedup it implies;
  - mean candidate and baseline cycles;
  - "single obs" for a member with one gSIM observation, and how many replicates were carried;
- roofline position: `candidate_over_roofline` per member from the latest tuning feedback, per trial;
- broker actions (`profile-tuning-member`, `profile-whole-model`, `analyze-command-buffers`, ...) on a
  timeline taken from the stage's event stream, and the broker receipts by action;
- holdout commit and reveal times;
- measurement slots, with executed commands and local memory per arm, and tokens per trial.

The records carry no wall clock. Every time on this page is a sealed file's modification time, and the
page says so. The measurement-slot chart is a reconstruction: each execution ends at its raw record's
write time, starts its recorded duration earlier, and is packed into the declared fan-out.

## Host resources (`--load PATH`)

A TSV with a header row, `utc load1 cpu_busy_pct gsim spike verilator codex`, one row a minute (the
format `record/sample_load.sh` writes). The page draws CPU busy %, load average and the simulator and
agent process counts over time. Malformed rows are counted and skipped.

## Run explorer and comparison

`--target T --explorer` indexes every Phase 0 release and run, every Phase 1 run (state, level,
driver/model, start/end, latest and formal grades, frozen commit, submission and corpus-seal digests)
and every Phase 2 run (trials, cells, best ratio, sealed) in sortable tables with one filter box. It
draws the lineage the records bind:

| Edge | Bound by |
|---|---|
| release → Phase 1 run | the run's `environment.yaml` `corpus_review.review_digest` equals the release's `seal.json` `review_digest` |
| Phase 1 run → paired Phase 2 | the campaign manifests' `functional_run_id`; their `functional_submission_sha256` must equal the run's `freeze.json` `submission_sha256` |
| Phase 1 run → whole-model Phase 2 | the index's `origin` commit equals the run's frozen commit |
| Phase 2 run → champion | the index's champion lineage |

An inconsistent binding (for example a Phase 2 experiment bound to a submission digest that differs
from the Phase 1 freeze) is drawn red and listed at the top of the page. A binding the records do not
make is listed as "not recorded" or "broken". Every node links to that run's page, which the explorer
writes beside itself.

`--compare A B` takes two runs of one phase:

- Phase 0: capsules added, removed and changed, and per-cohort coverage gaps closed and opened;
- Phase 1: grade progression against hours since each run's start, per-capsule tier changes, and
  tokens and time;
- paired Phase 2: per-member ratio deltas and roofline-position deltas.

## Private material

By default the page is public:

- hidden capsules, the held-out layer guard's digest, revealed held-out members and operator-only
  `reference_comparison.json` ratios appear as **counts** or as "present";
- held-out members are named by ordinal;
- a release's review note is omitted.

`--operator-private` names them. The page says which mode it is in. Do not hand an operator-private
page to an agent under test.

## Live views (`--live`)

`--live` rewrites the page every `--interval` seconds (default 60) and serves it on
`127.0.0.1:--port` (default 8765) with a stdlib HTTP server and a meta refresh, ready for a VS Code or
ssh port forward. The server answers GET for the page and the pages it links, by plain file name, and
nothing else. The live view adds nothing to the run:

- **Reads only.** It reads only files the run already writes. It never locks, renames, truncates or
  opens a run file for writing, and it refuses an output path inside any directory it reads.
- **Incremental.** Event streams are tailed from the byte offset of the previous poll, up to their
  last newline (a line still being written waits). A replaced or truncated stream is re-read. Feeds
  and call lists are capped.
- **Change-driven.** Each section is re-rendered only when the stat signature of its inputs changes.
  Clock-dependent sections also refresh every five minutes.
- **Low priority.** The process runs at nice 19 and idle I/O class (`ionice`, when present), pinned to
  `--cpus` when given.
- **Freshness.** The page states "data as of <newest record read>; page generated <now>; refreshed
  every N s", so a stalled run reads differently from a stalled viewer.

Measured on a synthetic Phase 1 run with a 200 MB event stream growing about 90 KB/s, refreshed every
10 s and pinned to one CPU: catching up on the whole file took 1.7 s of CPU, steady state averaged
about 0.3 % of one CPU, and RSS stayed at 61 MB.
