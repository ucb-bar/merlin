# AGENT.md — merlin/compiler

## Purpose

The eventual **stable MLIR/C++ compiler plane**: durable dialects, lowering passes, and target plugin integration. Currently scaffold only.

## What belongs here

- C++/TableGen dialect definitions, passes, conversions, and `merlin-opt`/`merlin-translate`.
- The four core dialects under `Dialect/{Contract,Schedule,Interface,Runtime}/`.

## What does not belong here

- Experimental Python analysis (that lives under `merlin/python/merlin/`).
- xDSL prototypes (those live under `merlin/python/merlin/xdsl_dialects/`).
- `merlin.dse` or `merlin.kernel` dialects.

## Interfaces

Stabilized counterpart to the xDSL prototypes. Builds via `CMakeLists.txt`. Lit/unit tests under `merlin/compiler/tests/`.

## Invariants

- This is the stable plane — do not put experimental Python analysis here.
- Promote a dialect here only after its xDSL prototype stabilizes.
- Do not require a full LLVM build until the project genuinely needs it.

## Testing expectations

lit tests in `tests/lit/`, unit tests in `tests/unit/` (once the build is wired).

## Notes for future agents

No real passes yet. Headers/dirs are placeholders with READMEs describing intent.
