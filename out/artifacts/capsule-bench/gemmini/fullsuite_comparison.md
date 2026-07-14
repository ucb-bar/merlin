# Full-suite comparison (gemmini_capsule_bench_v0)

Per-arm pass over **all** capsules (dynamic n/n, not a hardcoded pilot count). Time split (cumulative across quota-resumes): `active` = doing work (agent+oracle, from the driver), `quota_wait` = slept waiting on the 5h limit; within active, `agent`/`sim` = agent subprocess vs oracle (spike+verilator) wall. Cert tier = highest REQUIRED tier reached (L3 = real cycle-accurate RTL). Cycles are diagnostic-only and never gate.

| arm | run_id | suite | public | hidden | pass | tier | integrity | rounds | tokens | cost$ | active(s) | quota_wait(s) | agent(s) | sim(s) | wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merlin_assisted | pilot_merlin_0001 | pilot | 4/4 | 3/3 | PASS | L3 | clean | 2 | 6901059 | 16.6637 | — | — | — | — | 2783.269 |
| raw_baseline | dry_0002 | pilot | 2/2 | 1/1 | PASS | L3 | clean | 1 | 20950 | 0.1718 | — | — | — | — | 0.005 |
| raw_baseline | rb_pilot_0001 | pilot | 0/4 | 0/3 | no | L1 | clean | 5 | 11114773 | 24.3505 | — | — | — | — | 2645.199 |
| raw_baseline | rb_pilot_0002 | pilot | 4/4 | 3/3 | PASS | L3 | clean | 3 | 8977359 | 19.1746 | — | — | — | — | 2475.014 |
| raw_baseline | rb_pilot_cpp_01 | pilot | 4/4 | 3/3 | PASS | L3 | clean | 4 | 23679058 | 46.9808 | — | — | — | — | 5639.071 |
| raw_baseline | rb_pilot_rep_01 | pilot | 4/4 | 3/3 | PASS | L3 | clean | 2 | 7900180 | 18.1395 | — | — | — | — | 2977.692 |
| raw_baseline | rb_pilot_rep_02 | pilot | 0/0 | 0/0 | no | None | FAIL[contract]: no manifest.yaml in package /path/to/merlin/experiments/gemmini_capsule_bench_v0/runs/raw_baseline/rb_pilot_rep_02/submission | 8 | 1048771 | 3.1187 | — | — | — | — | 334.287 |
| raw_baseline | rb_pilot_rep_03 | pilot | 0/0 | 0/0 | no | None | FAIL[contract]: no manifest.yaml in package /path/to/merlin/experiments/gemmini_capsule_bench_v0/runs/raw_baseline/rb_pilot_rep_03/submission | 8 | 0 | 0.0 | — | — | — | — | 11.416 |

_`active`+`quota_wait` = `wall` (cumulative across resume invocations). `agent`+`sim` split `active` (the rest of active is harness/finalize overhead). `—` = a run predating this instrumentation (e.g. pilot runs launched before run_fullsuite.py)._
