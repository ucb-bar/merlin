---
title: Running capsule-bench experiments on AWS Bedrock
kind: guide
status: current
owner: capsule-bench
last_verified: 2026-09-04
related: [adding_a_target, getting_started]
code_refs: [merlin/experiments/capsule_bench/harness, merlin/python/merlin/targetgen/aet_bridge.py]
---

# Running capsule-bench experiments on AWS Bedrock

The agentic arms drive their model through Claude Code (`claude --print`). The **provider toggle**
(experiments-only) routes that CLI to AWS Bedrock instead of the interactive subscription, and every run's
token/cost telemetry is mirrored into the shared **aet** store so spend is tracked across all experiments
against a budget ceiling. This is opt-in and never changes default (subscription) behavior.

## Authentication

Auth is a **bearer token** (`AWS_BEARER_TOKEN_BEDROCK`, the newer Bedrock API-key style — not an
access/secret pair) kept in the gitignored repo `.env`. The `.env` loader is read-only, so the driver
surfaces the token into `os.environ` for `--provider bedrock` (a pre-set env var or `--aws-profile`
still wins); `bwrap` inherits `os.environ`, so the sandboxed `claude` sees it. Nothing to export by hand.

Sanity check (no full run):

```bash
.venv/bin/python merlin/experiments/capsule_bench/harness/smoke_agent_check.py \
  --provider bedrock --model us.anthropic.claude-haiku-4-5-20251001-v1:0 --aws-region us-east-1
# -> "bedrock verdict: reachable=True"
```

## Model ids

Use a Bedrock **inference-profile id** for `--model`. Get the current list from the API
(`GET https://bedrock.<region>.amazonaws.com/inference-profiles`). Note the 4.x Claude profiles have **no
`-v1:0` suffix**:

| Model | Profile id |
|-------|-----------|
| Opus 5 / 4.8 | `us.anthropic.claude-opus-5`, `us.anthropic.claude-opus-4-8` |
| Sonnet 5 / 4.6 | `us.anthropic.claude-sonnet-5`, `us.anthropic.claude-sonnet-4-6` |
| Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

Non-Claude models (Nemotron, Kimi, GLM-5, Nova, Qwen, DeepSeek, …) are on the same key but **cannot**
drive `claude --print` — reach them through Chia's `BedrockLLM` (Converse agent loop) instead.

## Running an arm

Debug the pipeline cheaply first (haiku, one round, oracle off), then run the real model:

```bash
MERLIN_TARGET_EXPERIMENT=merlin/experiments/capsule_bench/targets/<target>/target_experiment.yaml \
MERLIN_AET_SINK=1 \
.venv/bin/python merlin/experiments/capsule_bench/harness/run_rtlchecks_qa_loop.py \
  --run-id <id> --provider bedrock --aws-region us-east-1 \
  --model us.anthropic.claude-sonnet-4-6 \
  --max-rounds 1 --round-timeout 600 --sandbox bwrap
```

`run_rtlchecks_qa_loop.py` is arm-4 (CIRCT/RTL checks); it reuses the baseline QA loop. `--no-oracle`
skips the sim/verilator grade (faster/cheaper for pipeline debugging). Use a capable model (Sonnet/Opus)
for real runs — a weak model (haiku) tends to author a schema-invalid submission manifest and the grader
correctly rejects it.

## Telemetry and budget

With `MERLIN_AET_SINK=1`, `aet_bridge` records each finished run into `<run_dir>/logs/` in the aet
format (per-turn tokens split input / cache-read / cache-create / output, **per-model** cost + tokens,
`num_turns`, tool calls). This is additive — the legacy `experiment_tokens` cost yaml + trajectory plots
still write.

The opt-in is **sticky per run**, not per process: once `<run_dir>/logs/metrics.jsonl` exists, a later
`--resume` keeps recording even when it is launched from a shell that never exported `MERLIN_AET_SINK`.
Before that, a resumed run went quiet and its telemetry ended wherever the first session did.

`num_turns` is the CLI's own figure where a CLI reports one, and otherwise the assistant turns actually
present in the transcript (`aet.agent.assistant_turns` always records the latter). Drivers whose
terminating `result` event carries no turn count — codex is one — used to land as `num_turns: 0`.

Track cumulative spend across **all** experiments and enforce the ceiling:

```bash
.venv/bin/aet spend out/runs --budget-usd 300     # NOT `python -m aet` (no __main__)
```

It prints total + per-model spend and **headroom**, exits non-zero if over budget, and reports runs whose
cost is unknown as **`unpriced (cost unavailable — not $0)`** rather than silently counting them as zero.
Only Claude runs get a dollar cost from Claude Code's `total_cost_usd`; for other models supply prices via
`AET_PRICE_TABLE` (a JSON/YAML model→rate map) — aet ships accurate Claude + Nova rates and leaves unknown
models cost-unavailable. Dollar figures are estimates; reconcile against AWS Cost Explorer / CloudWatch
Bedrock token metrics for billing-accurate accounting. Token counts are exact.

Cost signal: a single trivial Sonnet call is ≈ $0.12 (dominated by the ~32k-token Claude Code system
prompt written to cache on first call; cache-**reads** are cheap after). A one-round haiku arm-4 debug run
was ≈ $0.76.

## Caveats

- The `.venv` is shared across concurrent sessions; a `uv run`/`uv sync` elsewhere can transiently empty it
  or drop the editable aet install. The sink is guarded (no-ops without aet, never crashes a run). Reinstall
  with `.venv/bin/pip install -e ../aet` (aet lives at `../aet`; merlin's `pyproject.toml` has a
  `telemetry` extra + `[tool.uv.sources]` path/editable entry).
- Chia has its own guarded sink (`CHIA_AET_SINK=1`) feeding the same aet store from its own runs.
