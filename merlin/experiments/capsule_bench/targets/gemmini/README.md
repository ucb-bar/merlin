# gemmini_capsule_bench_v0 — isolated, instrumented target-generation experiment

Turns `capsule_bench_v0` (the fidelity harness) into a scientific A/B experiment: a fresh agent
generates a Gemmini MLIR OOT backend in a sandbox, the submission is frozen + hashed, then graded
through the public/dev capsules (L0–L3: golden / ref==sim / spike / policy-selected elaborated RTL)
and, post-freeze,
the hidden capsules. Process metrics (wall / tokens / $ / tool-calls) come from the agent transcript.

> **The session that built this harness is NOT an experiment run.** A valid run is created only by
> `scripts/run_agent_experiment.py` (fresh workspace, one declared bundle, captured transcript +
> metadata, frozen artifact, separate grading phase, `run_manifest.yaml`).

## Arms (same task, same capsules, same hidden set, same grader)
- **raw_baseline** — un-tooled agent; bundle `input_bundles/raw_baseline_public_v0`.
- **merlin_assisted** — allowed Merlin authoring tools; bundle `input_bundles/merlin_assisted_public_v0`.
- **grader (private)** — hidden capsules + oracle adapters; used only in the post-agent grading phase.

## Run one experiment
```
# measured agent via the redacted-QA iterate-to-pass loop (sandbox=none; golden-masked copy):
python experiments/capsule_bench/targets/gemmini/scripts/run_baseline_qa_loop.py \
    --run-id rb_pilot_0003 --arm raw_baseline --model claude-opus-4-8 --effort high \
    --max-rounds 8 [--max-rate-limit-waits 3] [--resume]
#   PILOT_LANG=cpp forces a C++ OOT backend; default lets the agent choose (Python or C++).
#   --max-rate-limit-waits: on a five-hour-limit rejection, sleep until the window resets and retry
#     the same round (lets a run span windows unattended) — handles a quota hit while the process lives.
#   --resume: continue an interrupted run from its durable checkpoint (run_dir/qa_loop_state.yaml,
#     written after every round + every wait) — survives a process death (reboot/OOM/session-end) and
#     restores BOTH round progress AND accumulated active/rate-limit-wait time (does not restart at 0).
#     run_repeatability.py --resume skips already-complete runs and resumes mid-flight ones.

# repeatability sweep (n runs; rate-limit-aware, spans windows unattended):
python .../scripts/run_repeatability.py --n 3 --arm raw_baseline
python .../scripts/reclassify_repeatability.py --ids rb_pilot_rep_01,rb_pilot_rep_02,rb_pilot_rep_03

# full-suite audit — re-grade frozen submissions against ALL 25 capsules on the RTL oracle, in
# parallel, for cycles + per-class coverage + active-vs-waiting timing (no agent re-run, $0 API):
python .../scripts/full_suite_audit.py --backends rb_pilot_cpp_01,rb_pilot_0002 --workers 8 --tiers L2,L3

# aggregate all runs into the comparison table (joins in the full-suite N/25 column when present):
python experiments/capsule_bench/targets/gemmini/scripts/gen_reports.py

# reference-styled figures (no agent re-run; reads artifacts already on disk, works on live/partial runs):
python .../scripts/plots/make_plots.py \
    --runs raw_baseline/rb_full_01,merlin_assisted/merlin_full_01,raw_baseline/rb_pilot_cpp_01
#   -> reports/plots/{fig1_activity_trajectory, fig2_failure_planes, fig3_capsule_heatmap, fig4_ab_summary}.png
```

## Run-dir layout (immutable after grading)
```
runs/<arm>/<run_id>/
  TASK.md  input_bundle_manifest.yaml  environment.yaml
  transcript.jsonl  claude_stdout.log  claude_stderr.log  cost_time_toolcalls.yaml
  qa_loop_summary.yaml  qa_loop_state.yaml  (per-round + checkpoint; latter enables --resume)
  workspace/   submission/   iterations/iteration_XXX/
  grading_public/score_capsule.json   grading_hidden/score_capsule.json
  freeze.json   run_manifest.yaml   final_report.md
```

## Protocol
- **Public/dev**: agent may iterate until all required capsules pass (`iterations/`).
- **Freeze**: `freeze_run.py` hashes `submission/`; `freeze.json` pins the artifact + repo SHA.
- **Hidden**: graded only after freeze; hidden-repair is disabled by default (a separate labeled
  phase if ever enabled).
- **Isolation**: the workspace is assembled from the bundle's `allowed` only; `denied` paths
  (Merlin internals, reference/simulator, prior backends, hidden goldens, prior submissions) are
  asserted absent and bwrap-masked. The grader runs OUTSIDE the sandbox (it needs spike/verilator +
  hidden goldens).

## Cycles, tiers & metrics
- **Oracle tiers**: L0 golden, L1 ref==sim, trace-check, **L2 spike** (functional), **L3 elaborated
  RTL** (engine selected centrally and pinned by `MERLIN_REQUIRED_RTL_ENGINE`; this is where
  `cycles_diagnostic` comes from). A tier name is not a simulator name.
- **Cycles** are L3 `rdcycle`-bracketed counts from that exact required RTL engine; diagnostic-only,
  never gate pass/fail. `--workers N` runs the per-capsule oracle instances in parallel
  (deterministic — cycles are identical to a serial run).
- **Active-vs-waiting timing**: grader reports `timing_rollup` = `{suite_wall_s, sim_active_s,
  oracle_wait_s, parallel_speedup}` — `oracle_wait_s` is time blocked on a VCS/FireSim queue/FPGA
  slot (normally ≈0 for a local engine). The QA loop reports `active_wall_s` vs `rate_limit_wait_s`.
- **Reports**: `reports/comparison_table.md` (per-run `public` 4-pilot + **`full-suite` N/25**),
  `reports/full_suite_audit.md` (25-capsule matrix + per-class coverage + cycles + timing),
  `reports/repeatability.md` (rate-limit-aware valid pass-rate).
- **Figures** (`scripts/plots/`, → `reports/plots/`): `fig1_activity_trajectory` (smoothed agent-activity
  prevalence through the transcript, phase-shaded by round, with the bold capsules-passing line — the
  "metric through transcript" view), `fig2_failure_planes` (stacked failure-plane burn-down per round),
  `fig3_capsule_heatmap` (per-capsule pass/fail by round), `fig4_ab_summary` (coverage · cycles · effort
  · token economics), `fig5_scorecard` (cream-card run scorecard). `scripts/plots/_trajectory.py` extracts
  the tidy per-round/per-capsule/timeline data from on-disk artifacts (no agent re-run); per-round
  token/cost is recorded natively for runs graded after this change, older runs fall back to whole-run
  telemetry + the per-round capsule grades. `scripts/plots/_style.py` is the shared "ML-paper" theme
  (muted-pastel fills + darker matching edges, value labels, rounded callouts, solid 3D drop-shadows,
  cream rounded cards). fig1 background is selectable (`--bg rounds|reasoning|turnphase|stream`);
  `--compare-run` emits `fig1_compare_backgrounds.png` (one run under every background mode).

## Status
Harness built + validated end-to-end. Measured baselines pass the pilot at L3: `rb_pilot_0002`
(Python) and `rb_pilot_cpp_01` (C++ OOT). Full-suite audit + parallel oracle + rate-limit-aware
sweeps are wired. L4 VCS available; L5 FireSim honestly unavailable (replay hook not wired).
