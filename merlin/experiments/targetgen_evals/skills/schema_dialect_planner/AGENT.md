# Role: schema_dialect_planner

## Purpose
Produce `dialect_plan.yaml`, `target_contract.yaml`, `lowering_plan.yaml`, and
`runtime_adapter_plan.yaml` conforming to the harness JSON schemas.
Every op claim must be evidence-grounded.

## Allowed inputs (read-only)
- `<run_dir>/contracts/evidence_graph.jsonl` (if produced by evidence_extractor)
- `<run_dir>/contracts/kernel_records.jsonl` (if produced by kernel mining)
- `datasets/{target}/golden/expected_dialect_features.yaml` (reference only)
- `harness/schemas/*.schema.json`

## Allowed outputs (write)
- `<run_dir>/contracts/target_contract.yaml`
- `<run_dir>/contracts/dialect_plan.yaml`
- `<run_dir>/contracts/lowering_plan.yaml`
- `<run_dir>/contracts/runtime_adapter_plan.yaml`

## Forbidden modifications
- Any code file (.py, .td, .cpp, .mlir)
- `datasets/`, `methods/`, `configs/`, `harness/`

## Validation command
```bash
python -m harness.cli validate <run_dir>  # checks R5–R7
```

## Success criteria
- All schemas validate against `harness/schemas/dialect_plan.schema.json`
- Every op has `evidence`, `verifier`, and `lowering_exits`
- No op has `scheduling_policy` set
