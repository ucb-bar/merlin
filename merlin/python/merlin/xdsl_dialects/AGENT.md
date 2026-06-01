# AGENT.md — merlin/python/merlin/xdsl_dialects

## Purpose

merlin's own **prototype dialects** in xDSL. This is the default, fast Python plane for iterating on `merlin.contract`, `merlin.schedule`, `merlin.interface`, and `merlin.runtime` before committing to stable MLIR/C++.

## What belongs here

- xDSL dialect definitions, parsers/printers, verifiers, lowering prototypes.
- `contract.py`, `schedule.py`, `interface.py`, `runtime.py`.

## What does not belong here

- Stable production dialects (those get promoted to `merlin/compiler/`).
- Adapters to external xDSL tooling (that is `merlin/integrations/xdsl/`).

## Interfaces

Prototypes the same four dialects scaffolded in `merlin/compiler/include/merlin/Dialect/`. Consumes/produces `merlin/schemas/` artifacts.

## Invariants

- These are rapid prototypes — expect churn.
- Stable dialects may be promoted to MLIR/C++ under `merlin/compiler/`; keep names aligned.
- Do not create `merlin.dse` or `merlin.kernel` dialects — those stay schemas-first.

## Testing expectations

Small xDSL round-trip / verifier tests under `merlin/python/tests/`.

## Notes for future agents

xDSL is the default way we handle MLIR-style IR until a specific pipeline is chosen. See `docs/xdsl.md` and `docs/dialects.md`.
