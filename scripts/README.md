# `scripts/` — dev-convenience drivers

This directory holds shell-script drivers that orchestrate multi-step
`./merlin` pipelines. They are **not** primitives — every step they run is
already available as a `./merlin` subcommand. They exist as scripts because
the end-to-end flow is multi-stage (compile → build → run → stage) and
keeping the orchestration in a single inspectable shell file is easier to
debug than chaining shells around `./merlin` calls.

## When to add a script here vs. extend `./merlin`

- If you find yourself running the same `./merlin` sequence twice, add a
  driver here.
- If the orchestration is a single `./merlin` call with arguments, **don't**
  add a script — pass the arguments directly.
- If you find yourself adding logic that isn't pure orchestration (parsing
  vmfbs, decoding logs, computing schedules), the workflow belongs as a flag
  on an existing `./merlin` subcommand, not as a script here.

## Contents

| File | Purpose | Composes |
|---|---|---|
| `dronet_spike_e2e.sh` | Phase F end-to-end: dronet → kernel-embedded vmfb for bare-metal Spike | `./merlin compile` + the `samples/SaturnOPU/simple_embedding_ukernel/` build pattern |
| `radiance_muon_smoke.sh` | Single-command Merlin × Radiance smoke test | `./merlin compile --target radiance_muon` + `./merlin chipyard run-radiance-muon` |
| `zephyr_e2e.sh` | Merlin × Zephyr × FireSim end-to-end driver | `./merlin compile` + `./merlin build --profile zephyr` + `west build` + `./merlin chipyard stage-zephyr-workload` |
| `verify_mcp_registration.sh` | Verify `.mcp.json` registration of `merlin-targetgen` is healthy from a fresh shell | Validates the MCP server starts and exposes its 7 tools |

## Anti-patterns to avoid here

- **Don't** add scripts that reimplement a `./merlin` subcommand (e.g., a
  script that calls `iree-compile` directly bypassing `./merlin compile`).
- **Don't** add per-model scripts (e.g., `compile_yolov8.sh`). Add the model
  to a YAML in `models/` or to `benchmarks/<target>/`.
- **Don't** add debug or one-shot scripts here. Those go in `tmp/` (active)
  or `tmp/archive/investigations/<name>/scripts/` (frozen).
