# Observability — targetgen-evals

## Tracking policy

| Concern | Canonical tool | Notes |
|---|---|---|
| Reproducibility record | `run_manifest.yaml` + `runs/<target>/<run_id>/` | Immutable after `init-run` |
| Paper tables | `reports/<target>/metrics.csv` + MLflow | `compare` writes CSV; MLflow gives an interactive UI |
| Debug traces | OpenTelemetry spans | Available in `full` / `debug` mode only |
| Warnings / backend failures | `logs/tracking_warnings.jsonl` | Always written, never crashes |

## Tracking modes

| Mode | Backends active | When to use |
|---|---|---|
| `local` | LocalBackend only | CI, quick iteration, no server needed |
| `mlflow` | Local + MLflow | Paper baselines, comparing methods |
| `full` | Local + MLflow + OTel | Deep debug with distributed tracing |
| `debug` | Local + OTel (no MLflow) | Trace-only, no MLflow server |

```bash
# local — default, no external deps
python -m harness.cli init-run --target gemmini --method v0_naive_claude --seed 1

# mlflow — requires MLflow server (see docker-compose.mlflow.yml)
python -m harness.cli init-run --target gemmini --method v0_naive_claude --seed 1 \
  --tracking mlflow --mlflow-tracking-uri http://localhost:5000

# full — MLflow + OTel (SigNoz or any OTLP backend)
python -m harness.cli validate <run_path> \
  --tracking full \
  --mlflow-tracking-uri http://localhost:5000 \
  --otel-endpoint http://localhost:4318/v1/traces
```

## What gets written

Every run writes to `runs/<target>/<run_id>/logs/`:

| File | Contents |
|---|---|
| `events.jsonl` | Named events (init, validation started/completed, etc.) |
| `params.json` | Key-value run parameters |
| `metrics.jsonl` | Numeric metrics with timestamps |
| `artifacts.jsonl` | Artifact file registry |
| `tracking_warnings.jsonl` | Backend failures (never crashes the harness) |

## Graceful degradation

If MLflow or OTel is unavailable (server down, package not installed), the harness:
1. Prints a `[tracking warning]` to stderr
2. Appends the warning to `logs/tracking_warnings.jsonl`
3. Continues normally — local files are always written

This means **`--tracking mlflow` never fails your experiment** even if the MLflow server is unreachable.
