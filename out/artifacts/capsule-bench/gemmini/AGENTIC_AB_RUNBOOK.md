# Agentic A/B — runbook for adding data points

Goal: grow the pilot (baseline N=3 / merlin N=1) by launching matched A/B **pairs**. Each pair is two
independent `claude --print` sessions with everything identical except the one variable under test:

| Arm | Driver | Tools the agent has |
|-----|--------|---------------------|
| **A — raw_baseline** | `run_baseline_qa_loop.py --arm raw_baseline` | regular tools only |
| **B — merlin_rtlchecks** | `run_rtlchecks_qa_loop.py` | regular tools **+ our new RTL-derived compile/checks feedback** (advisory; FileCheck over emitted MLIR + decoded trace, bounds from CIRCT-extracted RTL facts) |

Both run the same redacted QA-gate loop (iterate-to-pass on the 4 pilot capsules, goldens masked),
the same model, the same sandbox, and the same token/cost/round accounting. The only thing arm B's agent
sees extra is the RTL-grounded feedback block in each round's verdict.

## Why two accounts
Concurrent opus sessions on ONE account rate-limit each other — that is exactly what blocked the pilot
(2/3 baseline + others lost to rate limits). Give each arm its own `CLAUDE_CONFIG_DIR` so they run truly
in parallel. Create two logged-in config dirs once:
```
CLAUDE_CONFIG_DIR=~/.claude-acctA claude   # log in account A, then exit
CLAUDE_CONFIG_DIR=~/.claude-acctB claude   # log in account B, then exit
```

## Launch
```
cd experiments/gemmini_capsule_bench_v0/scripts
PY=/path/to/merlin/.venv/bin/python

# 1) validate (no launch, no budget): prints exact commands + collision/account preflight
$PY launch_ab_batch.py --tag n02 --dry-run

# 2) launch one matched pair, each arm on its own account, sandboxed, backgrounded
$PY launch_ab_batch.py --tag n02 \
    --baseline-account ~/.claude-acctA --merlin-account ~/.claude-acctB

# more pairs at once (rb_n02_1/2/3 + rtlchecks_n02_1/2/3):
$PY launch_ab_batch.py --tag n02 --pairs 3 --baseline-account ... --merlin-account ...
```
Each session backgrounds itself (`start_new_session`), logs to `runs/<arm>/<run-id>.launch.log`, and the
batch writes `runs/ab_batch_<tag>.json` (run-ids + pids + logs). Run dirs are collision-checked — reusing
a `--tag` aborts before launching.

## Defaults & knobs
- `--model claude-opus-4-8`, `--max-rounds 6`, `--round-timeout 3600`, `--sandbox bwrap` (trusted results;
  `none` is dev-only). `--skip-hidden` to skip the hidden grading record. `--pairs N` for N pairs.

## After convergence (folds into the figures with no extra wiring)
```
$PY scripts/agg_agentic_results.py        # rebuilds reports/agentic_results.json (auto-marks valid/invalid + N)
cd ../gemmini_perf_bench/scripts && $PY gen_agentic_plots.py   # regenerates the 4 fig_agentic_*.png
```
`agg_agentic_results.py` reads every run under `runs/{raw_baseline,merlin_assisted}/*/`, so new runs are
picked up automatically; `valid` = converged + real (rate-limit-blocked/dummy runs are excluded) and the
caption N updates itself. Then `build_results_png.py` refreshes the poster.

## Note on what counts
The QA substrate each round (grading, redacted verdict) is UNCOUNTED; only the agent's autonomous
authoring (tokens/cost/tool-calls/wall, summed across rounds) is the measured effort. Launching real LLM
sessions spends budget — that is an operator decision, hence this wrapper never auto-launches.
