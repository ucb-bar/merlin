---
title: Core dialects
kind: reference
status: current
owner: ir
last_verified: 2026-07-07
related: [dialects, contracts, lowering_pipeline]
code_refs: [merlin/python/merlin/xdsl_dialects]
---

# Core dialects

Merlin owns five core dialects. Target dialects (gemmini, saturn, radiance, toynpu) are
generated/external and are **not** core.

| Human name               | MLIR namespace | Purpose |
| ------------------------ | -------------- | ------- |
| Merlin Contract Dialect  | `contract`     | Facts, obligations, capabilities, legality |
| Merlin Schedule Dialect  | `schedule`     | Compiler decisions and selected policies |
| Merlin Interface Dialect | `interface`    | Target-independent HW/SW interface abstractions |
| Merlin Runtime Dialect   | `runtime`      | Generic execution model (command buffers, devices, events, metrics) |
| Merlin DSE Dialect       | `dse`          | Interface candidates, variant runs, exploitability regimes (minimal, descriptive) |

Dialect namespaces are bare — no `m` prefix and no `merlin.` prefix. Ops read
`contract.assume`, `schedule.place`, `interface.resident_pack`, `runtime.submit`,
`dse.candidate`.

## Pipeline

```
linalg / tensor / scf / vector / affine
  -> contract    (what is true / what must be proven)
  -> schedule    (what compiler decision was chosen)
  -> interface   (what HW/SW abstraction is exposed)
  -> target dialect (how the target implements it)
  -> runtime     (how work is launched, synchronized, measured)
  -> runtime backend (simulator / host / baremetal / zephyr / firesim / external)
```

The boundary rule: `contract` says what is true, `schedule` says what was chosen,
`interface` says what abstraction is exposed, the target dialect says how it is
implemented, and `runtime` says how it is launched and measured. `dse` sits beside the
pipeline: it records candidates and measured variant results; it never lowers.

## Prototyping plane

Each core dialect is prototyped first in xDSL under
`merlin/python/merlin/xdsl_dialects/{contract,schedule,interface,runtime,dse}.py` and
promoted to a stable MLIR/C++ plane (**not yet built** — see `docs/design/compiler_plane.md`)
once the syntax is stable, has verifier tests, and has at least one lowering. See
`docs/xdsl.md`.

## What is intentionally *not* a dialect yet

`kernel` and `search` remain YAML/schema/tooling layers. They are promoted to IR only
after a schema representation, an xDSL prototype, a compiler use case, a lowering path,
verification tests, and an execution/simulation path exist. (`dse` crossed that bar in a
minimal form: it mirrors the `interface_candidate`/`dse_result`/`exploitability_report`
schemas as IR so candidates and measurements can live next to the pipeline.)
