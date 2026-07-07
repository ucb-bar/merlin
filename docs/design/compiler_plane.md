---
title: Design note: the future MLIR/C++ compiler plane
kind: design
status: current
owner: core
last_verified: 2026-07-07
related: [architecture]
code_refs: [merlin/python/merlin/xdsl_dialects]
---

# Design note: the (future) stable MLIR/C++ compiler plane

**Status: not built. The active compiler plane is Python + xDSL** under
`merlin/python/merlin/xdsl_dialects/` (five dialects: contract, schedule, interface, runtime, dse)
and `merlin/python/merlin/{targetgen,llvmlower}`.

The original design reserved a `merlin/compiler/` tree for an *eventual* stabilized C++/TableGen
plane — durable dialect definitions, lowering passes, and `merlin-opt`/`merlin-translate` tools —
into which an xDSL prototype would be "promoted" once it stabilized. That tree was pure scaffold
(AGENT.md placeholders, zero `.td`/`.cpp`/CMake) and was removed to keep the repo stub-free.

**If/when a C++ plane is actually needed:** create `merlin/compiler/` then with real TableGen +
CMake, promoting a specific xDSL dialect that has stabilized — don't reintroduce an empty skeleton.
Keep experimental analysis in Python; the C++ plane is only for stabilized, performance-critical
dialects/passes.
