# AGENT.md — merlin/experiments/gemmini_capsule_bench_v0

The **agentic A/B capsule-bench harness** for the Gemmini compiler: bwrap-isolated agent runs
(baseline / merlin-assisted / merlin+CIRCT arms), a QA loop, oracle grading, and reporting. It
*consumes* merlin (`scripts/` add `merlin/python` to `sys.path`, import `merlin.targetgen`).

- **Tracked source**: `scripts/` (harness: `run_agent_experiment`, `run_baseline_qa_loop`,
  grader, plots), `task/`, `input_bundles/` (per-arm manifests + `denied_files.txt` isolation
  lists), `contracts/`, and tracked `reports/` (METHODOLOGY.md, ARMS.md, `abc4_archive/` marked
  `DO_NOT_DELETE`).
- **Generated output**: runs route to `runs/gemmini/capsule-bench/` (constant `RUNS` in
  `scripts/_common.py`); per-experiment `runs/` + `_qa_ws/` are gitignored.
- **Isolation note**: the bwrap sandbox masks all of `/scratch`; answer-bearing reference
  backends are additionally listed in `denied_files.txt` / bundle manifests / the
  `transcript_tooling_audit` cheat regex under BOTH `generated_targets/` and `artifacts/targets/`.
