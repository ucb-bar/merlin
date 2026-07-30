# v0_naive_claude — Method Prompt

## Role

You are generating the `{target}-mlir` MLIR dialect target for the {target} systolic-array accelerator.
You are operating as the **naive baseline**: you have access to the frozen source snapshot and may write
any file you judge appropriate in the run's `generated/{target}-mlir/` directory.

## Inputs (read-only)

- `datasets/{target}/source_snapshot/` — frozen subset of the {target} hardware source
- `datasets/{target}/selected_docs/` — selected documentation and ISA tables
- `datasets/{target}/selected_kernels/` — representative baremetal kernel sources
- `datasets/{target}/golden/expected_contract.yaml` — expected hardware concepts (for reference)
- `datasets/{target}/golden/expected_dialect_features.yaml` — expected ops, types, attrs, exits

## Outputs (write only inside the run directory)

All output must be written under the run directory provided at launch time:

```
<run_dir>/
  generated/{target}-mlir/   ← your primary output
  contracts/                ← optional: planning artifacts
    target_contract.yaml
    dialect_plan.yaml
    lowering_plan.yaml
    runtime_adapter_plan.yaml
```

You may write whatever files you judge useful inside `generated/{target}-mlir/`.
The harness will validate the directory structure and contents.

## Hard constraints

- **Do not modify** `datasets/`, `methods/`, `configs/`, `harness/`, `reports/`.
- **Do not modify** any file outside `<run_dir>/`.
- **Do not clone** external repositories.
- **Do not modify** Merlin (`../merlin/`) or the real {target} source (`../{target}/`).
- All generated code must live under `<run_dir>/generated/{target}-mlir/`.

## What this baseline measures

This method measures whether unconstrained LLM-driven repo editing is sufficient to produce
a valid, architecture-compliant {target} dialect — without any structured planning, schema
constraints, or evidence grounding. It is the comparison ceiling.

A method that scores equally with this baseline provides no additional value.
A method that scores lower but at less cost may still be useful.
A method that scores higher demonstrates that structured generation adds value.

## Suggested approach (non-binding)

1. Read `source_snapshot/` to understand {target}'s hardware interface.
2. Read `expected_contract.yaml` and `expected_dialect_features.yaml`.
3. Write `contracts/dialect_plan.yaml` as a planning artifact.
4. Write `generated/{target}-mlir/xdsl/` with xDSL Python dialect definitions.
5. Write `generated/{target}-mlir/contracts/` with any additional planning.
6. Do not write TableGen (`.td`) or C++ unless you set `promotion_flag: true` in the manifest.
