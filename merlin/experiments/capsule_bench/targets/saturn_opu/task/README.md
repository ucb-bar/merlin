# saturn_opu task/ — the graded task is GENERATED, not authored here

This directory intentionally contains no `TASK_*.md` prose. In the `full` experiment mode the capsule-bench
harness **generates** the per-arm task prompt from the target manifest at run time via `render_prompt`
(target-agnostic) — the generated prompt, not a static file, is the source of truth for what the agent is
graded on. The arms differ only in their allowed/denied toolset (see `../input_bundles/`), which is
generated from `target_experiment.yaml` by `merlin.targetgen.generate_bundles`.

This directory exists so `_common.require_scaffolding()` passes (it checks for `task/` + `input_bundles/`)
and so the `experiments/.../task/` grant every arm is given actually resolves — it did not, and an absent
grant was silently skipped rather than reported.
Do not fork gemmini's hand-authored task prose here — that would drift from the generated prompt.
