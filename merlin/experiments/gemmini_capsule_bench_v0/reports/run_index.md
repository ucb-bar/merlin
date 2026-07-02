# Run index (gemmini_capsule_bench_v0)

8 run(s).

| run_id | arm | model | integrity | public | hidden | oracle |
|---|---|---|---|---|---|---|
| pilot_merlin_0001 | merlin_assisted | claude-opus-4-8 | clean | 4/4 | 3/3 | spike+verilator(L0-L3) |
| dry_0002 | raw_baseline | opus | clean | 2/2 | 1/1 | spike+verilator(L0-L3) |
| rb_pilot_0001 | raw_baseline | claude-opus-4-8 | clean | 0/4 | 0/3 | spike+verilator(L0-L3) |
| rb_pilot_0002 | raw_baseline | claude-opus-4-8 | clean | 4/4 | 3/3 | spike+verilator(L0-L3) |
| rb_pilot_cpp_01 | raw_baseline | claude-opus-4-8 | clean | 4/4 | 3/3 | spike+verilator(L0-L3) |
| rb_pilot_rep_01 | raw_baseline | claude-opus-4-8 | clean | 4/4 | 3/3 | spike+verilator(L0-L3) |
| rb_pilot_rep_02 | raw_baseline | claude-opus-4-8 | FAIL[contract]: no manifest.yaml in package /scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0/runs/raw_baseline/rb_pilot_rep_02/submission | 0/0 | 0/0 | spike+verilator(L0-L3) |
| rb_pilot_rep_03 | raw_baseline | claude-opus-4-8 | FAIL[contract]: no manifest.yaml in package /scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0/runs/raw_baseline/rb_pilot_rep_03/submission | 0/0 | 0/0 | spike+verilator(L0-L3) |
