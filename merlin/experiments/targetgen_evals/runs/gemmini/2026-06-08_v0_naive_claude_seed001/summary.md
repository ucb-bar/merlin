# Run: 2026-06-08_v0_naive_claude_seed001 *(smoke test)*

| Field | Value |
|---|---|
| target | `gemmini` |
| method | `v0_naive_claude` |
| seed | 1 |
| budget | `cheap_smoke` |
| git | `617a7a24dbfdf9dff6cede4c4a4f6fa0a825c41e` |

## Architecture Rules

Passed: 8 / 10

- ✓ **R1** generated-repo-naming: generated/gemmini-mlir/ exists
- ✗ **R2** xdsl-before-tablegen: xdsl/ directory is absent or empty; xDSL artifacts must exist before TableGen/C++ promotion
- ✓ **R3** no-premature-tablegen: No TableGen/C++ files found (correct for xDSL-first workflow)
- ✗ **R4** merlin-core-immutable: Merlin core files modified since init: merlin/benchmarks/semantic_memory/capacity_stress_reuse.yaml, merlin/benchmarks/semantic_memory/matmul_bias_requant_relu.yaml, merlin/benchmarks/semantic_memory/no_reuse_matmul.yaml, merlin/benchmarks/semantic_memory/repeated_rhs_matmul.yaml, merlin/compiler/include/merlin/Dialect/Contract/AGENT.md
- ✓ **R5** op-evidence: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
- ✓ **R6** op-verifier-coverage: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
- ✓ **R7** op-lowering-exit: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
- ✓ **R8** no-scheduling-in-semantics: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
- ✓ **R9** no-runtime-in-types: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)
- ✓ **R10** unsupported-fails-early: dialect_plan.yaml not present; skipping op-level checks (OK for empty run)

## Overall: `fail`


- xdsl/ directory does not exist under generated/gemmini-mlir/; no xDSL artifacts produced yet
