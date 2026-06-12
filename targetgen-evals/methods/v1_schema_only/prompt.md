# v1_schema_only — Method Prompt

## Role

You are generating planning artifacts only. You may write YAML/JSON contract files.
You may NOT write any generated xDSL, TableGen, or C++ code.

## Inputs (read-only)

Same as v0. Additionally, read the JSON Schema definitions under `harness/schemas/`.

## Outputs

- `contracts/target_contract.yaml`
- `contracts/dialect_plan.yaml`
- `contracts/lowering_plan.yaml`
- `contracts/runtime_adapter_plan.yaml`

No code output. This method measures whether structured schema planning alone is better than nothing.
