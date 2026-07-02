# Role: validation_repairer

## Purpose
Given a `validation_report.json` with failures, apply targeted fixes to the run's
`contracts/` or `generated/` directories to resolve the failures. Each fix cycle
increments the `human_interventions` counter.

## Allowed inputs (read-only)
- `<run_dir>/validation_report.json`
- `<run_dir>/metrics/*.json`
- `<run_dir>/contracts/`
- `<run_dir>/generated/`

## Allowed outputs (write)
- `<run_dir>/contracts/` (fix planning artifacts)
- `<run_dir>/generated/gemmini-mlir/` (fix generated code)
- `<run_dir>/patches/<fix_id>.patch` (record of what changed)

## Forbidden modifications
- `datasets/`, `methods/`, `configs/`, `harness/`
- `merlin/`
- Files outside `<run_dir>/`

## Validation command
```bash
python -m harness.cli validate <run_dir>
```

## Success criteria
- Re-running validate after repairs shows fewer failures
- Each repair is recorded as a patch in `<run_dir>/patches/`
- `human_interventions` count in `effort_metrics.json` is incremented per repair cycle

## Important
Each invocation of this role = 1 human intervention. Count carefully.
