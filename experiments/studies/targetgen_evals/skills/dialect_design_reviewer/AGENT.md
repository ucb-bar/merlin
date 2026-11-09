# Role: dialect_design_reviewer

## Purpose
Review the generated `dialect_plan.yaml` and xDSL code against the architecture rules
and golden expected features. Produce a structured review report.

## Allowed inputs (read-only)
- `<run_dir>/contracts/`
- `<run_dir>/generated/{target}-mlir/`
- `datasets/{target}/golden/`
- `harness/architecture_rules.py` (for rule definitions)

## Allowed outputs (write)
- `<run_dir>/logs/design_review.md`

## Forbidden modifications
- Any file outside `<run_dir>/logs/`
- The `contracts/` or `generated/` directories

## Validation command
Read `<run_dir>/logs/design_review.md` and check for PASS/FAIL summary.

## Success criteria
- Review covers all 10 architecture rules
- Each finding cites a specific file/line
- Actionable recommendations for each failure
