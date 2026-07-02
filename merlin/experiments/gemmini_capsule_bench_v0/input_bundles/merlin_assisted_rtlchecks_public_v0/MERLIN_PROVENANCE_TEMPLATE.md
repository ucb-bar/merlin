# Merlin provenance — `merlin_assisted` pilot run

> Fill this in after you converge (or stop) and save it as `submission/docs/merlin_provenance.md`.
> It is **not graded for correctness** — it documents *how* the Merlin authoring tools helped, so the
> A/B comparison can attribute any advantage. Be honest: "did not use" / "did not help" are valid and
> useful answers. Write from what actually happened, not from intent.

## 1. Merlin tools used

List each allowed Merlin tool you actually invoked or read, and what you used it for. (Leave blank /
"none" if you authored without it.)

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `targetgen/synthesize/` | | |
| `targetgen/generate/` (scaffold gen) | | |
| `xdsl_dialects/` (dialect patterns) | | |
| `targetgen/contract/interface_emit.py` | | |

## 2. Files generated with Merlin tooling

Which files in `submission/` were produced (in whole or part) by a Merlin generator/scaffold vs.
hand-authored? Note any post-generation edits you had to make.

| submission file | origin (generated / hand / mixed) | notes |
|---|---|---|

## 3. Failures encountered, and which Merlin tooling diagnosed

For each notable failure during iteration: the round, the capsule, the failure plane / trace
violations from `qa/verdict.json`, what you changed, and — specifically — whether a Merlin tool
(diagnostic/scaffold/plan) helped you find or fix it.

| round | capsule | failure plane / violations | fix | Merlin tool that helped (or "none") |
|---|---|---|---|---|

## 4. Files changed per iteration

A short per-round log of which `submission/` files changed (mirror / summarize `docs/iteration_notes.md`).

| round | files changed | result (better/worse/same; n_passed) |
|---|---|---|

## 5. Final-artifact integrity (self-attestation — the grader verifies independently)

- Does the final artifact import any Merlin runtime code (`import merlin` / `from merlin` /
  `merlin.runtime.reference` / `merlin.runtime.simulator` / `reference_outputs` / `pipeline.execute`)?
  **[ yes / no ]** — if yes, list where (this is an integrity failure; fix before freezing).
- Is the final artifact **self-contained** (graded only through its CLI entrypoints, no import of the
  Merlin package at runtime)? **[ yes / no ]**
- Any Merlin authoring artifacts that ended up in `submission/` by accident (scaffolL leftovers,
  generated `runtime/` adapters, scaffold leftovers, etc.)? **[ list / none ]**

## 6. One-line summary

Did Merlin tooling materially help this run, and how? (e.g. "scaffold saved the dialect boilerplate;
no help on the acc_scale rounding bug" / "did not use Merlin tools".)
