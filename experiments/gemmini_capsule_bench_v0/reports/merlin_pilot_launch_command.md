# merlin_assisted pilot — exact launch command

> **Do not launch from the harness-implementation session.** A valid run is launched by the script
> below as a fresh scripted Opus agent (implementation Claude ≠ experiment agent). This doc gives the
> exact, real, supported command — no invented flags.

## The command

```bash
cd /scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0/scripts
REPO=/scratch/agustin/projects/oscar-merlin
export PATH=$REPO/third_party/llvm-install/bin:/scratch2/agustin/chipyard-autocomp/install/bin:$PATH
export MERLIN_MLIR_INSTALL=$REPO/third_party/llvm-install
export MERLIN_CLANG=/scratch2/agustin/merlin/build/host-merlin-release/install/bin/clang-23

python3 run_baseline_qa_loop.py \
    --arm merlin_assisted \
    --run-id pilot_merlin_0001 \
    --model claude-opus-4-8 \
    --effort high \
    --max-rounds 8 \
    --round-timeout 2700 \
    --qa-timeout 1200 \
    --sandbox none

# The loop auto-runs the official freeze → public+hidden record via grade_agent_run.py.
# Then append the comparison row:
python3 gen_reports.py
```

Run it identically to the raw arm except `--arm merlin_assisted` and a distinct `--run-id` — same
launcher, same capsule subset, same QA redaction, same freeze→hidden, same sandbox mode.

## Why this launcher (not `run_agent_experiment.py`)

The task's intended shape named `run_agent_experiment.py` with `--capsules`/`--hidden-repair`/`--sandbox
bwrap`. Those do not match the codebase, and that launcher is a **single-shot** runner with **no
redacted QA loop**. The apples-to-apples iterate-to-pass loop that `raw_baseline` actually ran is
`run_baseline_qa_loop.py`. For a valid A/B the merlin arm must use the **same launcher + same sandbox
mode**. The loop was parameterized with a backward-compatible `--arm` (default `raw_baseline`, so the
in-flight raw invocation is unchanged) precisely so both arms share one driver.

## Flag mapping (intended → real)

| Intended (task wording) | Real | Note |
|---|---|---|
| `run_agent_experiment.py` | `run_baseline_qa_loop.py` | only the latter has the shared redacted QA loop |
| `--arm merlin_assisted` | `--arm merlin_assisted` | newly added, backward-compatible (default raw_baseline) |
| `--capsules A0,A2,A4,B0` | (implicit) `pilot_capsules/` subset | hardcoded `PILOT_SUBSET` = A0/A2/A4/B0 + hidden H0/H1/H2 |
| `--hidden-repair disabled` | (structural) | the loop repairs only against the **redacted public/dev** verdict; the hidden phase runs **post-freeze** in `grade_agent_run.py` with no repair |
| `--sandbox bwrap` | `--sandbox none` | **parity:** bwrap crashes the Bun-based `claude` binary here (SIGILL/FailedToOpenSocket); raw_baseline ran `none`. Isolation = golden-masked copied workspace + strengthened transcript audit + integrity scan |
| `--run-id pilot_merlin_0001` | `--run-id pilot_merlin_0001` | distinct id per run |
| `--model opus` | `--model claude-opus-4-8` | exact model id |

## What the run produces

`runs/merlin_assisted/pilot_merlin_0001/`: `environment.yaml` (arm/sandbox/isolation/golden-mask),
`rounds/round_NN.transcript.jsonl`, `qa_history/verdict_round_NN.json`, `qa_loop_summary.yaml` (per-round
`answer_access_clean` + audit hits), `submission/`, `cost_time_toolcalls.yaml`, `freeze.json`,
`grading_public/` + `grading_hidden/` (post-freeze), `run_manifest.yaml`, `final_report.md`. `gen_reports.py`
appends one row to `reports/comparison_table.md`.

## After the run (operator-side, NOT in the agent sandbox)

```bash
python3 merlin_similarity_audit.py --run-id pilot_merlin_0001
# -> reports/merlin_similarity_audit_pilot_merlin_0001.md  (copy-vs-prior-backends check)
```

## Notes / future work
- `--sandbox bwrap` for `claude` is unresolved in this environment; if fixed later, the same bundle
  deny-list masks the oracle helpers for real. Not a parity blocker today.
- Strict audit parity: the raw arm ran under the pre-hardening audit; the hardened audit is strictly
  additive (detection-only, never a grading gate). If exact audit parity is wanted, re-grade the raw arm
  under the current driver (cheap) — it does not change pass/fail.
