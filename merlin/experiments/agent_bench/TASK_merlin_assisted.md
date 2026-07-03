# Task — generate an out-of-tree Gemmini target-backend package (Merlin-assisted)

Identical artifact, identical contract, identical hidden grading as the baseline
(`TASK_baseline.md`) — read it for the contract, entrypoints, hard rules, and success criteria.
The output is the **same** artifact type: `mlir_oot_target_backend`, `integrity_exempt: false`,
graded by the same `grade.sh` on the same public + hidden examples.

## What is additionally available to you

Beyond the public contract + toolchains, you MAY use the Merlin target-generation tooling to
*produce* your package (the package must still be self-contained and pass the integrity scan — the
helpers are scaffolding, not a runtime dependency you import into the submission):

- `merlin.targetgen` authoring helpers (registry, synthesize, generate, evidence/ingest).
- the xDSL prototype dialects + lowering scaffolding.
- target-spec scaffolding and the package generator.
- structured failure provenance / the AET recording layer.
- documented command-buffer semantics and the MLIR-faithful codegen patterns.

## Hard rules (unchanged)

- Emit BOTH a command buffer AND a lowered LLVM/RoCC kernel; correct artifact class.
- The **submitted package** must not `import merlin` / read the reference outputs — it is invoked
  only via its CLI entrypoints and is integrity-scanned. (Using Merlin tools to *author* it is
  fine; shipping a package that imports them is not, unless you legitimately vendor self-contained
  code.)
- You must NOT read the reference solution packages (`artifacts/targets/gemmini/
  {merlin_native_v0,hand_smoke_oot}`) or the hidden tests.

## Success criteria

Same as baseline: required g0/g1 public pass; hidden g0/g1/g2 variants; stretch g2.

The comparison measures whether this tooling helps: highest rung reached, hidden pass rate, wall
time, cost, tool calls, failure planes, artifact completeness — vs. the raw baseline.
