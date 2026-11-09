# AGENT.md — experiments/studies/llm_kernel_vs_compiler_v0

Status: active — the `feat/kernel-vs-compiler` study tooling is folded into main; this directory is the study's home.

## Purpose

Study of where two ways of bringing workloads to a new accelerator (Radiance) cross over: repeatedly
LLM-generating a kernel per workload, versus spending LLM effort once to generate a compiler, freezing
it, and compiling unseen workloads with no further agentic adaptation. `TASKS.md` is the task register
with DONE/PARTIAL/OPEN state; `merlin experiment study board` reads it beside the run matrices and the
baseline record, so every number comes from disk rather than from the register.

## Layout

- `methods/{bedrock_kernel,codex_kernel,gemini_kernel}/` — per-driver kernel-generation method specs.
- `scripts/` — matrix runner (`run_matrix.py`), kernel-agent driver (`run_kernel_agent.py`), the
  AutoComp bridge (`autocomp_bridge.py`, `run_autocomp.py`), eligibility manifest builder, model
  inventory/check, provenance audit, and the `kvc_capture*.sh` capture wrappers.
- `voided_runs.yaml` — reviewed list of matrix runs voided by a harness defect: kept in the spend, out
  of every rate.
- `eligibility/`, `workloads/`, `shim/` — curated inputs (which workloads/models qualify, the workload
  corpus, the compatibility shim the frozen compiler is driven through).

## Provenance

Runs live under `out/runs/radiance/...` and results under `out/artifacts/`; nothing generated is
tracked here.
Consumes the library only (`merlin.targetgen`, `merlin.benchharness`); nothing in the library reads
this directory.
