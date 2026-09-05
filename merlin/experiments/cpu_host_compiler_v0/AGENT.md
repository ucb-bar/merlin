# AGENT.md — merlin/experiments/cpu_host_compiler_v0

## Purpose

Four-arm agentic experiment for producing a reusable CPU-host compiler with scalar and RVV paths.  The
same frozen generic corpus and grader are used for every arm; only the nested authoring information changes.

## The four arms

`arm1_raw_cpp` ⊂ `arm2_cpp_scaffold` ⊂ `arm3_generated_cpu_dialect` ⊂
`arm4_agentic_pass_authoring`.

Arm 4 may author passes, but a pass is promoted only through the same deterministic train/validation
acceptance gate as a knob, flag, or heuristic.  Agent judgment proposes code; it never decides whether
the proposal wins.

## Invariants

- The five paper networks and their shapes/results are absent from development and selection.
- A run is `NO_GO` until the portable capsules, grader, K1 probe, AET, Chia, and Codex authentication pass.
- Every paid run uses the Codex subscription driver, Chia scheduling, and the AET sink.
- Raw Codex JSONL, line-arrival timestamps, every reported token subset, active/wall time, tool calls,
  grader time, and billing mode are retained in `out/runs/k1_cpu/cpu-host-compiler/`.
- The registered schedule is the exact four-block Williams design in `analysis_plan_v1.yaml`. Provider
  sampling is unseeded; row `seed` values are paired-block identifiers in AET/run IDs, not Codex seeds.
  Cross-block carryover is excluded only after the retained K1 washout/requalification boundary passes.
- Generated output never lands here.
- Search screens two train capsules from each of all six generic families on trusted Spike. Each sweep's
  deterministic width-one top survivor then receives exactly six balanced parent/child K1 measurements
  per capsule on one controller-private post-freeze train and validation shape from each of all six generic
  families; heldout is never
  opened.
