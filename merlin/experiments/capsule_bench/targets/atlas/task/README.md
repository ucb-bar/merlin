# atlas task/ — the graded task is GENERATED, not authored here

This directory intentionally contains no `TASK_*.md` prose. In the `full` experiment mode the capsule-bench
harness **generates** the per-arm task prompt from the target manifest at run time via `render_prompt`
(target-agnostic) — the generated prompt, not a static file, is the source of truth for what the agent is
graded on. The four arms differ only in their allowed/denied toolset (see `../input_bundles/`), which is
generated from `target_experiment.yaml` by `merlin.targetgen.generate_bundles`.

This retained README documents prompt ownership; generated-prompt workflows no longer require
an authored `task/` directory. Materialized input bundles remain required.
Do not fork gemmini's hand-authored task prose here — that would drift from the generated prompt.
