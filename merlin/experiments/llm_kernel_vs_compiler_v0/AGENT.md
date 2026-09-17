# AGENT.md — merlin/experiments/llm_kernel_vs_compiler_v0

Status: active — developed on `feat/kernel-vs-compiler`; this copy tracks that branch.

## Purpose

Study of where two ways of bringing workloads to a new accelerator (Radiance) cross over: repeatedly
LLM-generating a kernel per workload, versus spending LLM effort once to generate a compiler, freezing
it, and compiling unseen workloads with no further agentic adaptation. `STATUS.md` records what is
verified right now; `TASKS.md` is the task register with DONE/PARTIAL/OPEN state.

## Layout

- `methods/{bedrock_kernel,codex_kernel,gemini_kernel}/` — per-driver kernel-generation method specs.
- `scripts/` — matrix runner (`run_matrix.py`), kernel-agent driver (`run_kernel_agent.py`), the
  AutoComp bridge (`autocomp_bridge.py`, `run_autocomp.py`), eligibility manifest builder, model
  inventory/check, provenance audit, and the `kvc_capture*.sh` capture wrappers.
- `eligibility/`, `workloads/`, `shim/` — curated inputs (which workloads/models qualify, the workload
  corpus, the compatibility shim the frozen compiler is driven through).

## Provenance

Runs live under `out/runs/radiance/...` and results under `out/artifacts/`; nothing generated is
tracked here. The study branch is `feat/kernel-vs-compiler` in its own worktree (`$KVC_WORKTREE`).
Consumes the library only (`merlin.targetgen`, `merlin.benchharness`); nothing in the library reads
this directory.
