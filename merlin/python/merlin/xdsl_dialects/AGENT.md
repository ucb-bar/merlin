# AGENT.md — merlin/python/merlin/xdsl_dialects

## Purpose

merlin's own **core dialects** in xDSL. This is the default, fast Python plane for iterating on the `contract`, `schedule`, `interface`, `runtime`, and `dse` dialects before committing to stable MLIR/C++. Dialect namespaces are bare (no `m`/`merlin.` prefix).

## What belongs here

- xDSL dialect definitions, parsers/printers, verifiers.
- `_common.py` (shared enums/guards), `contract.py`, `schedule.py`, `interface.py`, `runtime.py`, `dse.py`.
- `lowering/` — the staged contract → schedule → interface → target → runtime lowering and its cross-op analyses, ending in a command-buffer dict for `merlin.runtime` (the Python engine).

## What does not belong here

- Stable production dialects (those would graduate to a future MLIR/C++ plane — not built; see
  `docs/design/compiler_plane.md`).
- Adapters to external xDSL tooling (implement in-package; see `docs/design/integrations.md`).
- Target dialects (generated into target repos by TargetGen).

## Interfaces

Prototypes the five core dialects (contract/schedule/interface/runtime/dse) in Python. Consumes/produces `merlin/schemas/` artifacts. `lowering/emit_command_buffer.py` emits dicts conforming to `command_buffer.schema.yaml`, executed by the `merlin.runtime` Python engine.

## Invariants

- These are rapid prototypes — expect churn, but keep `module.verify()` green.
- Stable dialects may later graduate to a future MLIR/C++ plane (see `docs/design/compiler_plane.md`); keep names aligned.
- `dse` is a real (minimal) dialect by explicit decision — it mirrors the `interface_candidate`/`dse_result`/`exploitability_report` schemas and never participates in lowering. `kernel`/`search` stay schemas-first.
- Local single-op checks live in `verify_`; cross-op checks (use-after-evict, placement legality, command-buffer consistency) live in `lowering/analyses.py`.

## Testing expectations

Per-dialect build/verify/invalid/round-trip tests plus the end-to-end lowering test under `merlin/python/tests/` (`test_xdsl_*.py`, `test_xdsl_lowering_e2e.py`).

## Notes for future agents

xDSL is the default way we handle MLIR-style IR until a specific pipeline is chosen. Copy the proven xDSL 0.65 idioms (field-annotation attribute params, `EnumAttribute` + `SpacedOpaqueSyntaxAttribute`, `func.ReturnOp`). See `docs/xdsl.md` and `docs/dialects.md`.
