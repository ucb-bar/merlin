---
title: Replaying compiler construction actions
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/construction_actions.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/construction_study.py
---

# Replay compiler construction with the ordinary experiment tools

Use `--offline-actions /absolute/public-actions.json --fresh-backend --max-wall-s 1800`
with the ordinary selected target, bundle, library, source seal, resources and
sandbox arguments. Remove `--preflight-only`/`--codex-canary`; this mode performs
ordinary fresh admission and then executes scripted author actions without a
provider. It does not resume a retained compiler. Its initial submission is empty.
The source release and library must be refreshed after harness/library edits;
never reuse a stale seal or modify an archived run to make this command pass.

The automatic author can execute the same action plan inside its own workspace:

```sh
python construction_actions.py public-actions.json --out study-001 --timeout 1800
```

Both paths execute the same client bytes, ordinary compiler/library commands, and
`agent_selfcheck.py`. The native offline path uses the existing sandbox composer,
answer masks, source grants, independent brokers and per-request snapshots. It
adds no private imports, companion backend grant, numerical shortcut or new gate.
Ordinary shell/compiler actions and client submissions use two concurrent slots
by default (`--command-workers 1..16` inside the author workspace). Completion
processing only polls ready children, so a long build does not block independent
construction or test admission. Submitted checks run concurrently in the existing
rolling broker. All commands inherit the same dynamic CPU reservation, with no
fixed per-role CPU partitions. Use explicit `after` dependencies for edits or
builds that share ownership of outputs or sources; independent actions must own disjoint
files. Builds must respect the selected shared pool rather than assume each
command independently owns all CPUs.
Automatic authors retain their separately recorded helper ceiling (up to 16) and
20 shared CPUs; this scripted driver does not pretend to spawn model helpers.

Each action names a semantic capability and compiler boundary. The supported
boundaries are `graph`, `numeric_layout`, `schedule_storage`, `isa_lowering`,
`host_runtime`, and `integration`. Optional `after` lists explicit earlier action
IDs. A failed prerequisite defers only those dependent actions. There is no
all-capsule readiness barrier. The full 153 correctness + 20 calibration + 77
scalability obligations remain unchanged; the 77 are the primary performance
optimization cohort.

A plan has `schema: "merlin.construction_actions.v1"` and an `actions` array:

```json
{
  "schema": "merlin.construction_actions.v1",
  "actions": [
    {"id": "inspect", "capability": "connected-regions", "boundary": "graph",
     "kind": "command", "argv": ["python", "capability_query.py", "--groups"]},
    {"id": "screen-residual", "capability": "residual", "boundary": "integration",
     "kind": "selfcheck", "argv": ["--capsules", "JC_resnet50_prepared_connected", "--sim", "spike"]},
    {"id": "screen-projections", "capability": "connected-projections", "boundary": "integration",
     "kind": "selfcheck", "argv": ["--capsules", "JC_tiny_llama_prepared_connected", "--sim", "spike"]},
    {"id": "screen-epilogue", "capability": "matmul-bias", "boundary": "integration",
     "kind": "selfcheck", "argv": ["--capsules", "PF00_fused_matmul_bias_m16k16n16", "--sim", "spike"]}
  ]
}
```

This is a check-selection example, not an implemented compiler. Insert public
construction actions before relevant checks: ordinary argv may invoke generic
Merlin passes/ISA generation, edit compiler sources, build them, and inspect their
outputs. A self-contained operator-authored construction script may be supplied
as `python -c SOURCE` in the explicit public plan, so no extra unsealed source
mount is needed. Such code is an operator input, not autonomous author output.
Never embed withheld schedules, reference kernels or private performance answers.
The three named capsules above are existing public smoke selections, not an
assertion that they equal arbitrary residual/shared-projection/MLP study graphs.

Use exact graph/numeric/layout sources and preserve externally visible outputs.
Neutral ISA generation must come from public declarations and generic Merlin
tooling, not an existing gemmini-mlir implementation. Any extra generated neutral
scaffold requires its own explicit selected source treatment; the compiler itself
still starts empty. The model agent chooses transforms and schedule heuristics.

Logs live in `construction_study/events.jsonl`, per-action stdout/stderr, and
`construction_study/result.json` in the workspace. The private run adds
`construction_study_result.json` and launcher logs. Events retain plan/action
identity, request ID and available graded source hash. Original self-check output
retains detailed numerical, L2, L3 and model feedback; a command return code or
completed action count is not a capsule count or performance result. Pending
checks are collected without resubmission. New evaluations remain attached to the
broker's actual source/runtime identities, including explicit newer-byte notices.
Replay means the same declared actions and validation machinery; it does not
claim identical timings or unchanged source state across separate runs.

See [the compiler contribution map](../../examples/compiler/compiler_change_map.md)
for implemented API boundaries and their remaining execution obligations.
Use the [construction study milestones](../design/compiler_coevolution_study.md#readiness-and-subsequent-comparisons)
to distinguish tool admission, first target execution, corpus completion and
complete-model validation. A successful scripted action is insufficient to
advance any numerical execution milestone.
