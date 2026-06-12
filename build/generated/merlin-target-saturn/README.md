# merlin-target-saturn

Generated Merlin target package for `saturn`.

This repository was scaffolded by `merlin.targetgen`. It is **human-reviewable**: the generator does not claim its synthesized artifacts are correct. Start from `contracts/` and `docs/evidence_report.md`.

## Layout

- `contracts/` — the five synchronized plans
- `xdsl/` — xDSL prototype dialect
- `include/`, `lib/`, `tools/` — MLIR/C++ dialect scaffold (placeholder)
- `runtime/` — Merlin runtime **adapter** (command encoding, simulator, metrics)
- `zephyr/` — Zephyr runtime-backend module
- `llvm/` — LLVM extension plan (out-of-tree first)
- `examples/`, `tests/`, `docs/`

## Core rule

Merlin owns the core dialects (`contract`/`schedule`/`interface`/`runtime`) and the runtime abstraction. This target only provides a dialect + a runtime adapter.
