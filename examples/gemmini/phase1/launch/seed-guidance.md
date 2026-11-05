
## Starting from a seed compiler

This workspace continues a previous run. `submission/` holds its compiler, and you have everything that run had: start with `SEED_INDEX.md` at the workspace root, which lists what it left, one line each.

The seed is a starting point to evolve, not a design to preserve. Make big changes whenever you judge they pay off: restructure or rewrite whole family lowerings, replace the seed's routes with library routes wholesale, change layouts, tiling or scheduling strategy, delete code that no longer earns its place. You do not need to keep the seed's structure or its choices, and big changes are not riskier here than small ones: every grade shows exactly which cases a change affected. Unchanged executables keep their proven verdicts and model cycles, and changed ones are re-checked first. That record is your safety net, not a reason for caution.

What the previous run left, all outside `submission/`:

* `SEED_EVIDENCE/SEED_START_GRADE.md`, when present: the seed compiler graded unchanged on this run's grader before launch, which test cases pass (most carried) and their cycles. That is where you start; a case that passed there and fails after your edits is a regression your change introduced, and the grade feedback names the error.
* `SEED_VARIANTS/`: every variant compiler it built, whole, with `OUTCOMES.json` (what each changed and which records name it). Copy one into `variants/` to continue it, or promote it into `submission/` when it wins.
* `SEED_PERFORMANCE/baseline.json`: its cycles, roofline, utilization, headroom and fences per case. Grades and performance briefs show `seed_cycles` and `vs_seed` for each case, so you can see from the first grade where you are ahead of or behind the seed. `SEED_PERFORMANCE/perf_history.jsonl` is its full history.
* `OPTIMIZATION_RULES.json` (keep maintaining it), `SEED_EXPERIMENTS/` (its A/B and evidence records, including rejected experiments: repeat one only with a new reason), `SEED_TOOLS/` (its helper scripts), `PREVIOUS_SESSION.md` (where it stopped), `SEED_NOTES/` (its reports and notes; they may describe an older harness) and `SEED_EVIDENCE/` when present (the seed's baseline map and structure diff).

Correctness is already established for the seed's executables, so spend your time on cycles. Give helpers whole operation families to improve against the seed's numbers, the slowest large cases first, and integrate each win as soon as it keeps numerics. Check the cases a change affects (`--capsules` with their names, or `--panel FAMILY`) instead of sweeping everything. Performance is read from the Spike cycle-model estimates; gSIM confirmation is optional and nothing waits for it.
