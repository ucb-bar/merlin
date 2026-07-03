# AGENT.md — merlin/experiments/targetgen_evals

Self-contained, **import-isolated** LLM eval project: measures whether structured pipelines
(evidence grounding, schema planning, deterministic emission) beat unconstrained LLM editing at
generating the Gemmini MLIR target dialect. It has its own `pyproject.toml` and **imports zero
`merlin.*`** — it treats merlin as an *external subject* it evaluates (the isolation is the
research premise; keep it).

- Runs via `python -m harness.cli {init-run,validate,compare}`. Methods v0–v6 in `methods/`,
  frozen inputs in `datasets/gemmini/`, agent roles in `skills/`.
- Lives off the repo root but **inside** this checkout so its R4 check can `git diff -- merlin/`
  against the subject; `harness/git_root()` discovers the repo root (do not reintroduce
  `root.parent`). Its `runs/`/`reports/` are its own lifecycle (layout-linter exempt).
- History note: relocated from the old top-level `targetgen-evals/`. aet ships a parallel native
  `targetgen` suite; this is the concrete project instance.
