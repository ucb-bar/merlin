# AGENT.md — merlin/python/merlin/agentreport

## Purpose
Read a directory of agentic experiment runs and say, per run, **what it is and what it did** — arm,
phase, model, capsules passing over time, tool spans, concurrency, tokens and cost — so a report can
be regenerated instead of hand-assembled.

This package answers only what it can prove. Every value it returns is paired with an
:class:`~merlin.agentreport.availability.Status` (`MEASURED` / `DERIVED` / `UNAVAILABLE(reason)`),
and a reader that cannot measure something **refuses with a reason** rather than returning a zero. A
zero plots as a finding; a refusal plots as a gap.

## Modules
- `availability.py` — the `Status` / `Availability` ledger every other module returns alongside its data.
- `index.py` — walks run roots and identifies each run. Roots, arm names and the bench→phase map are
  all **parameters**: they are declared by the launcher, which lives outside the library.
- `spans.py` — tool-call spans, transcript first then the driver's raw event stream, plus `concurrency()`.
- `passes.py` — capsules passing over time, from `selfcheck_log.jsonl` or (derived) from verdict mtimes.
- `tokens.py` — token buckets and the three-state cost classification.

## Facts about the data that are encoded here, not rediscovered
- **The arm is not the directory.** Arms 3 and 4 launch into the same run-dir subtree, and
  `run_manifest.yaml:arm` records that subtree. Resolve from the input bundle, else the run-id prefix.
  When the two disagree the bundle wins (an arm *is* its grant set) and the conflict is recorded.
- **Runs written before 2026-09-04 have `tool_use` blocks with no `id`**, so the transcript's
  tool_use→tool_result join is impossible and `agent_trace.timeline` returns a clock with no spans.
  `spans.py` falls back to the driver's `item.started`/`item.completed` stream, which kept its join key.
  This is the difference between 9 plottable runs and 141.
- **`arrived_at` marks when the harness READ a line.** 56–65% of raw-stream pairs come back under
  10 ms because start and finish were flushed together. Long durations are real; short ones are not.
  `concurrency()` therefore recomputes without the short spans and **refuses if the answer moves**.
- **`selfcheck_log.jsonl` `n_passed` is trustworthy** (40,335/40,335 rows agree with their own
  `failing` list). The hazard is the 873 rows with **no denominator** — a self-check that could not
  run — which chart as a collapse to zero. They are excluded and counted.
- **`wall_offset_s` resets each round** and must be rebased onto one clock.
- **Metered, notional and unpriced are three different quantities.** Unpriced is `None`, never `0.0`.

## Rules
- No target-name literals, no `import re` — this is gated library code.
- Nothing here may read the experiment tree; the caller passes in what it needs.
- Any new derived metric needs a **paired** test: one that proves it finds the thing, one that proves
  it refuses when the thing is absent. See `merlin/tests/infra/test_agent_report_core.py`, whose
  mutation coverage is checked by deliberately breaking each reader.
