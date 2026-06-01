# AGENT.md — merlin/python/merlin/targetgen/backends

## Purpose

Code-emission backends: xdsl (default), mlir_cpp, tablegen.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Real algorithm implementations (TODO stubs only at this stage).
- Generated artifacts (write those to `output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- No real algorithms yet — placeholder modules with explicit TODOs only.
