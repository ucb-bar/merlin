# Comparison table (gemmini_capsule_bench_v0)

Apples-to-apples across arms: same task, same capsules, same hidden set, same grader.
`public` is the 4-capsule pilot (the agent's iterate-to-pass gate); **`full-suite` is all 25 capsules** (every test — see reports/full_suite_audit.md). Cycles are diagnostic-only. Process metrics come from the agent transcript (`available:false` recorded honestly when the CLI emits no usage).

| arm | run_id | model | wall(s) | tokens | cost$ | tool_calls | public | full-suite | hidden | tier | numeric | integrity | first-failure | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merlin_assisted | pilot_merlin_0001 | claude-opus-4-8 | 2783.269 | 6901059 | 16.6637 | 85 | 4/4 | — | 3/3 | L3 | True | clean | none | 1 |
| raw_baseline | dry_0002 | opus | 0.005 | 20950 | 0.1718 | 2 | 2/2 | — | 1/1 | L3 | True | clean | none | 1 |
| raw_baseline | rb_pilot_0001 | claude-opus-4-8 | 2645.199 | 11114773 | 24.3505 | 134 | 0/4 | — | 0/3 | L1 | True | blocked(rate_limit) | rate_limit (no fair attempt) | 1 |
| raw_baseline | rb_pilot_0002 | claude-opus-4-8 | 2475.014 | 8977359 | 19.1746 | 106 | 4/4 | 19/25 | 3/3 | L3 | True | clean | none | 1 |
| raw_baseline | rb_pilot_cpp_01 | claude-opus-4-8 | 5639.071 | 23679058 | 46.9808 | 225 | 4/4 | 18/25 | 3/3 | L3 | True | clean | none | 1 |
| raw_baseline | rb_pilot_rep_01 | claude-opus-4-8 | 2977.692 | 7900180 | 18.1395 | 87 | 4/4 | — | 3/3 | L3 | True | clean | none | 1 |
| raw_baseline | rb_pilot_rep_02 | claude-opus-4-8 | 334.287 | 1048771 | 3.1187 | 19 | 0/0 | — | 0/0 | None | None | blocked(rate_limit) | rate_limit (no fair attempt) | 1 |
| raw_baseline | rb_pilot_rep_03 | claude-opus-4-8 | 11.416 | 0 | 0.0 | 0 | 0/0 | — | 0/0 | None | None | blocked(rate_limit) | rate_limit (no fair attempt) | 1 |

_Notes: `dry_*` rows are dummy pipeline-validation runs (not agent results). `rb_pilot_0001` is the PRE-grader-fix diagnostic run (failed trace_check only because of two grader bugs since fixed — schema `$id` resolution + rocc_decode SSA-name regex); it is kept for the record, not a baseline result. `integrity=blocked(rate_limit)` rows (`rb_pilot_rep_02/03`) were rejected by the org five-hour session limit (zero real work) and are NOT baseline failures — they are excluded from the pass-rate in `reports/repeatability.md`. `rb_pilot_cpp_01` is the explicit-C++ OOT baseline; `rb_pilot_0002` is the agent's-choice (Python) baseline. Real measured agent runs are produced by `run_baseline_qa_loop.py`. Both arms must be graded by the same (patched) grader and the same task file — see COORDINATION.md. Cycles are diagnostic-only and never gate pass/fail._
