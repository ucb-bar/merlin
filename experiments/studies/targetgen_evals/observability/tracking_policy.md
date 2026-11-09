# Tracking Policy

## What is canonical for what

| Question | Where to look | Tool |
|---|---|---|
| "Did this run produce valid xDSL?" | `runs/<target>/<run_id>/metrics/summary_metrics.json` | filesystem |
| "Which method performed best across seeds?" | `reports/<target>/ablation_table.md` | `compare` command |
| "Paper number for method v3, seed 1–5" | `reports/<target>/metrics.csv` | CSV / MLflow |
| "What happened during run X, step by step?" | `logs/events.jsonl` | local JSONL |
| "What are the per-span latencies?" | OTel backend (SigNoz / Logfire) | `--tracking full` |
| "Did any tracking backend fail?" | `logs/tracking_warnings.jsonl` | local JSONL |

## Decision tree for choosing tracking mode

```
Is this a smoke test / CI check?
  └─ YES → --tracking local  (no server needed)

Is this a real paper baseline?
  └─ YES → --tracking mlflow (requires docker-compose.mlflow.yml up)
  └─ Do you also want trace-level debug?
       └─ YES → --tracking full (MLflow + OTel)

Do you want OTel traces without MLflow?
  └─ YES → --tracking debug
```

## Adding a new metric

1. Compute the value in the relevant validator (`validate_*.py`)
2. Return it in the validator's result dict
3. Pull it into `build_summary()` in `collect_metrics.py`
4. Add it to `_ALL_COLUMNS` (and `_STRING_COLUMNS` if non-numeric)
5. Add it to `metrics.schema.json`

The metric will appear automatically in `metrics.csv` and the ablation table.

## Costs and tokens

`cost_usd` / `tokens_input` / `tokens_output` are `null` until an LLM integration
populates them. The intended flow:

1. LLM caller records cost + tokens in `run_dir / "logs" / "llm_usage.jsonl"`
2. A validator (`validate_evidence` or a dedicated `validate_cost.py`) reads this file
3. Returns `{"observed_cost_usd": ..., "tokens_input": ..., "tokens_output": ..., "token_source": "llm_usage.jsonl"}`
4. `build_summary()` picks these up automatically

Until that integration exists, these columns are `NA` — that is correct and expected.
