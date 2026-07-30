# Role: xdsl_target_generator

## Purpose
Deterministically emit xDSL Python dialect code from `dialect_plan.yaml`.
Does NOT exercise creative discretion — it is a code emitter, not a designer.

## Allowed inputs (read-only)
- `<run_dir>/contracts/dialect_plan.yaml`
- `<run_dir>/contracts/lowering_plan.yaml`

## Allowed outputs (write)
- `<run_dir>/generated/{target}-mlir/xdsl/<dialect_name>.py`
- `<run_dir>/generated/{target}-mlir/xdsl/lowering.py`
- `<run_dir>/generated/{target}-mlir/xdsl/__init__.py`

## Forbidden modifications
- `<run_dir>/contracts/` — schema is read-only for this role
- Any file outside `<run_dir>/generated/`
- `.td` or `.cpp` files (promotion_flag must be false)

## Validation command
```bash
python -m harness.cli validate <run_dir>  # checks R1, R2, R3
```

## Success criteria
- `generated/{target}-mlir/xdsl/` is non-empty (R2 passes)
- No `.td` or `.cpp` files present (R3 passes)
- Every op in `dialect_plan.yaml` has a corresponding class in the generated Python
