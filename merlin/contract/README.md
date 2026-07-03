# `merlin/contract/` — the experiment ABI (v0.1)

This is a **repo-independent contract**: the fixed interface against which an *out-of-tree target
backend package* is built, invoked, certified, and scored. It exists so two agents — a raw
baseline (docs + this contract only) and a Merlin-assisted one (docs + contract + Merlin tooling)
— solve the **same** problem and are graded by the **same** runner. The contract surface is a
**subprocess + file boundary**: a package is a self-contained tool invoked only via the CLI
entrypoints below; it never imports the harness's internals (see `integrity_policy.md`).

## What a package must do

Consume an `*.interface.mlir` written in the frozen `merlin_iface` grammar (see
`interface_grammar.md`) and expose four CLI entrypoints (see `mlir_oot_backend_contract.yaml`):

```
parse                       interface.mlir            -> ok/diagnostics
lower_interface_to_target   interface.mlir            -> target MLIR (stdout)
emit_command_buffer         interface.mlir,{out.json} -> command_buffer.json
lower_target_to_llvm        interface.mlir            -> LLVM/RoCC MLIR (stdout)
```

The Merlin runner (`python -m merlin.targetgen.oot_runner`) then certifies:

```
interface.mlir --(package)--> command_buffer.json --> reference_outputs(cb) == simulate(cb)   [L0, always]
interface.mlir --(package)--> LLVM/RoCC --> RV64 ELF --> spike|verilator oracle               [L1/L2]
require: oracle_output == reference_outputs(cb) == simulate(cb)   (integer, bit-exact ==)
```

## Files

| File | Role |
|------|------|
| `VERSION` | contract version (`0.1`) |
| `interface_grammar.md` | the frozen `merlin_iface` input grammar |
| `interface_dialect_contract.yaml` | required interface ops/types/attrs |
| `target_dialect_contract.yaml` | required target dialect (namespace/ops/types) |
| `command_buffer_abi.yaml` | command-buffer opcode semantics |
| `mlir_oot_backend_contract.yaml` | artifact type, entrypoints, LLVM pin |
| `oracle_runner_contract.yaml` | oracle ladder + `OUT/METRIC/DONE` output format |
| `telemetry_schema.yaml` | required recorded-run metadata |
| `scoring.yaml` | the K0–K10 conformance ladder |
| `integrity_policy.md` | the no-harness-import / no-cheat rules |
| `examples/*.interface.mlir` | golden public inputs (g0/g1/g2) |
| `examples/expected_command_buffer_g0.json` | golden cb for g0 |
| `schemas/*.schema.json` | JSON Schemas (fail-closed validators) |

## Languages

Packages may be **C++** (a real out-of-tree `gemmini-opt` built against LLVM/MLIR 23 — see
`mlir_oot_backend_contract.yaml`) or **Python** (a self-contained CLI tool). The contract is
language-agnostic; only `manifest.language`, the optional `build` block, and the entrypoint
`argv` differ.
