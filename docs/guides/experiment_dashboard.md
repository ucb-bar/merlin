---
title: Tracking experiments — the dashboard and the watch view
kind: guide
status: current
owner: experiments
last_verified: 2026-10-06
related: [storage, reproducibility, phase2_test_justification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/tracking/records.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/html.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/text.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/__init__.py
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
```

The page is a single HTML file. Its CSS, its SVG charts and its short tooltip script are all inline,
so it opens from disk, through an ssh port-forward, or as a mailed copy, and it never fetches anything.
By default it is written to `out/artifacts/experiments/<target>/dashboard/<run>.html`. A target
view goes to `.../dashboard/target.html`. That home is the storage contract's `experiment-dashboards`
product root (`merlin/contract/storage.yaml`). `--out` writes elsewhere, and `--open` also opens the
page in a browser.

## What it reads, and what it never does

The views **never measure, grade or build**. Every number on the page is a field some owner wrote:

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
