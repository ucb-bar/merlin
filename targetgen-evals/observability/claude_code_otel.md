# Claude Code + OpenTelemetry

Claude Code can export OTLP traces to any compatible backend when the right environment
variables are set. These traces capture prompt/completion spans and can be correlated with
targetgen-evals run spans.

## Environment variables

```bash
# Enable OTLP export from Claude Code
export CLAUDE_CODE_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=claude-code-targetgen

# Optional: correlate with your run
export OTEL_RESOURCE_ATTRIBUTES="run_id=<run_id>,target=gemmini,method=v0_naive_claude"
```

## Running with a local OTel Collector

Start the collector (uses `otel-collector.yaml` in this directory):

```bash
docker run --rm -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/observability/otel-collector.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector-contrib:latest \
  --config /etc/otel-collector-config.yaml
```

Then run your harness:

```bash
python -m harness.cli validate <run_path> \
  --tracking full \
  --otel-endpoint http://localhost:4318/v1/traces
```

## Span hierarchy

```
validate_run
  └─ validate.architecture_rules
  └─ validate.schema
  └─ validate.xdsl
  └─ validate.passes
  └─ validate.evidence
  └─ validate.design
  └─ validate.runtime_mock
  └─ validate.merlin_integration
```

## Correlating with MLflow

The harness writes `otel_trace_id` into `run_manifest.yaml` (observability.opentelemetry.trace_id)
and into `validation_report.json` (tracking.otel_trace_id). Use this to jump from an MLflow
run to the corresponding OTel trace in your viewer.
