# targetgen-evals

Research evaluation harness for MLIR target-dialect generation.

## Research goal

We are evaluating whether LLM-assisted target-dialect generation is useful *beyond prompt
engineering*. The question is not "can Claude write a Gemmini dialect?" but "does a structured
generation pipeline with evidence grounding, schema planning, and deterministic code emission
produce better dialects than unconstrained LLM editing?"

To answer this we need controlled, falsifiable comparisons with fixed datasets, isolated runs,
validation gates, and comparable metrics. This harness provides that scaffold.

## Why this is not prompt engineering

Prompt engineering optimises the *input to the LLM*. This harness evaluates *system-level
architectures* for dialect generation:

| Method | What it evaluates |
|---|---|
| v0_naive_claude | Ceiling of unconstrained LLM editing |
| v1_schema_only | Value of structured planning artifacts |
| v2_schema_generator | Value of LLM-as-planner + deterministic generator |
| v3_evidence_graph | Value of evidence-first grounding |
| v4_rtl_tools | Value of automated RTL analysis |
| v5_kernel_miner | Value of kernel-derived abstractions |
| v6_full | Full pipeline upper bound |

Each method is a different *system design*, not a different prompt for the same system.

## What gets compared

Every method run produces a `metrics/summary_metrics.json` with these dimensions:

| Metric | Measures |
|---|---|
| schema_valid | Is the run manifest well-formed? |
| xdsl_files | How much xDSL code was generated? |
| xdsl_op_estimate | How many ops were defined? |
| pass_tests_pass | How many positive tests pass? |
| evidence_coverage | What fraction of claims have evidence? |
| unsupported_claim_rate | What fraction of claims are unsupported? |
| arch_rules_passed | How many architecture rules pass? |
| arch_rules_failed | How many architecture rules fail? |
| human_interventions | How many repair cycles were needed? |
| cost_usd | How much did the LLM method cost? |
| time_to_first_pass_s | How long until the first passing test? |
| heldout_shape_success | Do held-out shapes generalise? |
| merlin_core_files_modified | Did the method contaminate Merlin core? |

## Run isolation model

Every run is fully isolated:

```
runs/gemmini/<YYYY-MM-DD>_<method>_seed<NNN>/
  run_manifest.yaml       ← immutable after init
  generated/gemmini-mlir/ ← all method output
  contracts/              ← planning artifacts
  metrics/                ← computed by harness validate
  logs/
  patches/
  validation_report.json
  summary.md
```

Runs cannot see each other. Methods cannot modify datasets, configs, harness, or Merlin.

## Layout

```
configs/        — target and budget configs (read-only during runs)
datasets/       — frozen source snapshot + golden files + tests (read-only during runs)
methods/        — method definitions and prompts (read-only during runs)
harness/        — validation and comparison code
skills/         — AGENT.md role constraints for sub-tasks
runs/           — isolated run directories (write target during runs)
reports/        — aggregated reports (written by compare command)
```

## Quick start

```bash
cd targetgen-evals

# Install harness dependencies
pip install -e ".[dev]"  # or: uv pip install -e ".[dev]"

# Create a smoke-test run (harness bootstrap only — not a real baseline)
python -m harness.cli init-run --target gemmini --method v0_naive_claude --seed 1

# Validate the empty run (will report failures — that is expected)
python -m harness.cli validate runs/gemmini/2026-06-08_v0_naive_claude_seed001

# Aggregate all runs into a report
python -m harness.cli compare --target gemmini
```

## When to run the real v0 baseline

The `init-run` smoke test only verifies the harness mechanics. It is marked
`is_smoke_test: true` and excluded from the ablation table.

Before running the real v0 baseline:
1. Populate and freeze `datasets/gemmini/source_snapshot/` (see `gemmini_source_curator` AGENT.md)
2. Run `init-run --no-smoke --budget standard_eval --method v0_naive_claude --seed 1`
3. Invoke the v0 method agent with `methods/v0_naive_claude/prompt.md`
4. Validate and inspect the report

## How to interpret metrics

- A method with `arch_rules_failed: 0` respects the two-plane architecture.
- A method with `evidence_coverage > 0.8` is well-grounded.
- A method with `pass_tests_pass / pass_tests_total = 1.0` produces a correct dialect.
- A method with `cost_usd` much lower than v0 at equal quality is more efficient.
- A method with `human_interventions: 0` is fully autonomous.
- `merlin_core_files_modified: 0` is required for all methods (R4).

The ablation table in `reports/gemmini/ablation_table.md` shows all methods side-by-side.
