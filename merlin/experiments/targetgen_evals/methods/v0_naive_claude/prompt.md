# v0_naive_claude — Method Prompt

## Role

You are generating the `gemmini-mlir` MLIR dialect target for the Gemmini systolic-array accelerator.
You are operating as the **naive baseline**: you have access to the frozen source snapshot and may write
any file you judge appropriate in the run's `generated/gemmini-mlir/` directory.

## Inputs (read-only)

- `datasets/gemmini/source_snapshot/` — frozen subset of the Gemmini hardware source
- `datasets/gemmini/selected_docs/` — selected documentation and ISA tables
- `datasets/gemmini/selected_kernels/` — representative baremetal kernel sources
- `datasets/gemmini/golden/expected_contract.yaml` — expected hardware concepts (for reference)
- `datasets/gemmini/golden/expected_dialect_features.yaml` — expected ops, types, attrs, exits

## Outputs (write only inside the run directory)

All output must be written under the run directory provided at launch time:

```
<run_dir>/
  generated/gemmini-mlir/   ← your primary output
  contracts/                ← optional: planning artifacts
    target_contract.yaml
    dialect_plan.yaml
    lowering_plan.yaml
    runtime_adapter_plan.yaml
```

You may write whatever files you judge useful inside `generated/gemmini-mlir/`.
The harness will validate the directory structure and contents.

## Hard constraints

- **Do not modify** `datasets/`, `methods/`, `configs/`, `harness/`, `reports/`.
- **Do not modify** any file outside `<run_dir>/`.
- **Do not clone** external repositories.
- **Do not modify** Merlin (`../merlin/`) or the real Gemmini source (`../gemmini/`).
- All generated code must live under `<run_dir>/generated/gemmini-mlir/`.

## What this baseline measures

This method measures whether unconstrained LLM-driven repo editing is sufficient to produce
a valid, architecture-compliant Gemmini dialect — without any structured planning, schema
constraints, or evidence grounding. It is the comparison ceiling.

A method that scores equally with this baseline provides no additional value.
A method that scores lower but at less cost may still be useful.
A method that scores higher demonstrates that structured generation adds value.

## Suggested approach (non-binding)

1. Read `source_snapshot/` to understand Gemmini's hardware interface.
2. Read `expected_contract.yaml` and `expected_dialect_features.yaml`.
3. Write `contracts/dialect_plan.yaml` as a planning artifact.
4. Write `generated/gemmini-mlir/xdsl/` with xDSL Python dialect definitions.
5. Write `generated/gemmini-mlir/contracts/` with any additional planning.
6. Do not write TableGen (`.td`) or C++ unless you set `promotion_flag: true` in the manifest.
