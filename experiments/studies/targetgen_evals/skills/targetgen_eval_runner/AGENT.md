# Role: targetgen_eval_runner

## Purpose
Orchestrate an end-to-end evaluation run for a single method and target.
Calls `init-run`, invokes the method's agent (in a sandboxed sub-session), then calls `validate`.

## Allowed inputs (read-only)
- `targetgen-evals/methods/<method>/` — method definition, prompt, allowed tools
- `targetgen-evals/configs/` — target and budget configs
- `targetgen-evals/datasets/<target>/` — source snapshot and golden files

## Allowed outputs (write)
- `targetgen-evals/runs/<target>/<run_id>/` — all run outputs

## Forbidden modifications
- `targetgen-evals/datasets/` — source dataset is read-only
- `targetgen-evals/methods/` — method definitions are read-only
- `targetgen-evals/configs/` — configs are read-only during a run
- `targetgen-evals/harness/` — harness code is read-only
- `merlin/` — Merlin core must not be modified (R4)
- Anything outside `targetgen-evals/`

## Validation command
```bash
python -m harness.cli validate runs/<target>/<run_id>
```

## Success criteria
- `validation_report.json` written with `status: validated`
- `metrics/summary_metrics.json` written
- No crash on empty or partial generated repo
