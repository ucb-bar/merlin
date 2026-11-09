# MLflow Quickstart

## Start the server

```bash
cd targetgen-evals/observability
docker compose -f docker-compose.mlflow.yml up -d
```

MLflow UI will be available at http://localhost:5000.
Runs are stored in a local SQLite DB + artifact volume — data persists across container restarts.

## Run an experiment with MLflow tracking

```bash
cd targetgen-evals
python -m harness.cli init-run \
  --target gemmini --method v0_naive_claude --seed 1 \
  --no-smoke \
  --tracking mlflow --mlflow-tracking-uri http://localhost:5000

python -m harness.cli validate <printed_run_path> \
  --tracking mlflow --mlflow-tracking-uri http://localhost:5000

python -m harness.cli compare --target gemmini
```

## Reading the experiment table

- Open http://localhost:5000
- Select experiment `targetgen-evals-gemmini`
- The **Metrics** columns map to `summary_metrics.json` fields
- **Tags**: `target`, `method`, `seed`, `tracking_mode`
- **Artifacts**: `run_manifest.yaml`, `validation_report.json`, `metrics/`

## Stop the server

```bash
docker compose -f observability/docker-compose.mlflow.yml down
```

Data survives because it's in the `mlflow-data` Docker volume.
To wipe: `docker compose -f observability/docker-compose.mlflow.yml down -v`
