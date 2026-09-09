# Task prompts for `gemmini_universal`

This directory carries NO hand-authored `TASK.md`, deliberately.

For `--experiment full` (the arm-4 launch shape) the graded contract is **generated**:
`run_baseline_qa_loop._build_task` calls `merlin.targetgen.generate_prompt.render_prompt(te, manifest,
experiment, arm)`, so the prompt is derived from this target's own descriptor + capability manifest +
RTL fact bundle. A committed `TASK_full.md` is no longer the source of truth on any target, and copying
gemmini's would be actively wrong here: gemmini's names `--convert-iface-to-gemmini`, declares
`output_dtype: i32`, and says nothing about a device whose mvout cannot move an int32 accumulator row.

For `--experiment realistic` the harness prefers `task/TASK_realistic.md` when a target ships one and
falls back to the same generated prompt when it does not. This target ships none, so `realistic` also
renders from the descriptor.

The directory itself exists because every generated arm bundle grants
`experiments/capsule_bench/targets/gemmini_universal/task/` read-only; a grant naming a path that does
not exist is the shape that produced a bundle which *looked* like it handed an arm its material and
handed it nothing.
