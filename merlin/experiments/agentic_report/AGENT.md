# AGENT.md — merlin/experiments/agentic_report

## Purpose
Turn the agentic run directories into a regenerable report: an index, a facts file, a figure kit and
the written update. It reads BOTH benches (functional and performance), which is why it sits here
rather than inside either one.

## Pipeline
```
build_report.py rescue    # copy light telemetry out of the roots marked fragile
build_report.py index     # identify every run: target, bench, phase, arm, driver, model
build_report.py facts     # extract every number, and mark the best run per cell + every ladder
figures.py                # render the figure kit
write_report.py           # write the prose from the same facts file
```
Everything lands under `out/artifacts/agentic-report/`. The figures and the prose read
`run_facts.json` and nothing else, so a caption cannot drift from the run that produced it.

## What lives here vs in the library
`merlin/python/merlin/agentreport/` does the reading and may not know a target name, an absolute path
or an arm. Those are declared in **`roots.yaml`** and threaded in. Adding a root, a target or an arm
is an edit to that file.

## Things this tool refuses to do, and why
- **It will not compare fractions across different corpora.** The corpus grew from 11 to 96 capsules
  over the study, so a bar chart of "best run per arm" would show arms differing when what differs is
  the denominator. `fig01` draws only tag-matched ladders — same corpus, same model, same day.
- **It will not sum metered and notional cost.** A subscription seat is not billed per token.
- **It will not publish an overlap that depends on flush-suspect spans.** `arrived_at` marks when the
  harness read a line, so a start and a finish read together, or a pile of ends read at one instant,
  manufacture concurrency nothing ever executed. Both cases are checked; a run whose overlap does not
  survive is refused, and one where only a minority is suspect reports the surviving figure and says
  so.
- **It will not drop a run it cannot read.** Gaps are drawn as gaps and counted in the caption. An
  absent panel reads as "no data existed", which is a different claim from "we could not measure it".

## Known shape of the data
- Only ~half the runs kept a self-check log; the rest were graded continuously and their progress
  clock is the verdict files' mtime, which is marked `derived` on the figure.
- Most of the strongest scores are on a seat-billed model, so the best-score and best-known-cost runs
  are disjoint sets.
- Arms 2 and 3 are thinly populated on two of the three targets; only radiance has a genuine ladder
  spanning arms 1, 3 and 4 on one corpus.
