# AGENT.md — experiments/muon_perf_bench_v0

## Purpose

The Muon SIMT perf-bench harness — the Muon analog of `experiments/gemmini_perf_bench/`. Runs the FP32
GEMM corpus through the Muon backend via `merlin.targetgen.muon_capsule_runner` (tier ladder L0 golden /
L1 consistency / L2 cyclotron `--timing`) and reports GFLOP/s vs the Muon SIMT FP peak.

## What is tracked (harness source)

- `scripts/` — `run_muon_perf.py` (perf report), `run_muon_qa_loop.py` (agentic QA loop),
  `agent_selfcheck.py` (grader).
- `kernels/` — the capsule corpus (`capsule.yaml` + `capsule.interface.mlir`).
- `task/`, `input_bundles/` — the agent task spec + public RTL-checks bundle.

## What is generated (not tracked)

- `runs/` — routed to **`runs/muon/perf-bench/<run-id>/`** (three-root convention); the per-experiment
  `runs/` dir is gitignored. Reference package: `artifacts/targets/muon/reference_v0` (regenerable).

## Reproduce

```
.venv/bin/python experiments/muon_perf_bench_v0/scripts/run_muon_perf.py \
    --package artifacts/targets/muon/reference_v0 --run-id ref_v0
# → runs/muon/perf-bench/ref_v0/perf_results.json + perf_table.md
```
