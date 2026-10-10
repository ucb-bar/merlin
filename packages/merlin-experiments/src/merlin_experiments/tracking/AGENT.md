# AGENT.md — packages/merlin-experiments/src/merlin_experiments/tracking

## Purpose

Experiment tracking views: a static HTML dashboard and a live terminal view, from records only.

## Modules

- `activity.py` — What the authoring agent did, from its own event stream: a feed, tool-call spans and token use.
- `activity_html.py` — The agent-activity section of a run page: what it is doing now, a feed, and per-round tokens/time.
- `charts.py` — Inline-SVG chart primitives the richer views share: a Gantt, a treemap, a scatter, a time series.
- `compiler.py` — What compiler the agent has built, checkpoint by checkpoint: passes, dialects, ops, entry points, LOC.
- `explorer.py` — Explore a target's runs: an index of every phase's runs, the lineage between them, and comparisons.
- `html.py` — Render a tracking summary as ONE self-contained HTML page: inline CSS, inline SVG, a few lines of JS.
- `live.py` — ``dashboard --live``: regenerate the page every N seconds and serve it on 127.0.0.1, stdlib only.
- `phase0.py` — Read a Phase 0 derivation or generation run into the corpus-and-coverage view's summary.
- `phase0_html.py` — Render the Phase 0 corpus-and-coverage summary (:mod:`.phase0`) as one self-contained page.
- `phase1_detail.py` — Phase 1 jobs, self-checks and grades on one timeline, with family pass rates and recorded cost and time.
- `phase1_views.py` — The richer Phase 1 sections of a run page: timeline, compiler evolution, family pass rates, cost/time.
- `phase2_paired.py` — Read a checkpointed paired Phase 2 experiment: trials x members x {tuning, held_out, held_out_form}.
- `phase2_views.py` — Render a checkpointed paired Phase 2 experiment (:mod:`.phase2_paired`) as page sections.
- `records.py` — Read an experiment's own records into one summary: what the dashboard and ``watch`` show.
- `resources.py` — Host load samples and a monitor's notes: two operator-side records a run view can sit beside.
- `tail.py` — Read growing records cheaply and without touching them: offset tailing and stat signatures.
- `text.py` — The terminal view of a tracking summary: plain text, ANSI colour optional, no curses.
- `views.py` — Assemble the dashboard pages from the readers, with per-section reuse for ``--live``.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- Read records, never produce evidence: no module here measures, grades, builds, or writes into a run
  directory or a measurement store. The only write is the dashboard page itself.
- A missing, unreadable or foreign-schema record is `None` plus an inventory row, rendered "not
  recorded" -- never a crash, a zero or a guessed default.
- Classify phase-2 outcomes from the fields their owners write (`isa_prohibited`, `coverage_regression`,
  `screen_failed`, `infra_*`, `timing_status`) using the owners' constants and `gates.is_infra_refusal`;
  never by matching text patterns of your own.
- The page stays self-contained: inline CSS/SVG/JS only, no network, no external file.
- Private material (hidden capsules, held-out members and digests, operator-only reference ratios,
  review notes) is a count or "present" unless the caller passes `operator_private`; every page that
  can show it names its mode.
- A live view adds nothing to a run: it opens run files read-only (never locks, renames, truncates or
  writes them), tails JSONL from the last byte offset up to the last newline, re-renders a section only
  when its inputs' stat signature changes, and refuses an output path inside any directory it reads.
- Times that are file modification times (sealed Phase 2 records, job marker files) are labelled as such;
  a reconstructed chart (measurement slots) says it is a reconstruction.
