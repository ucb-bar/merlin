# v6_full — Method Prompt

## Role

Full pipeline: RTL analysis + kernel mining + evidence graph + schema planning + deterministic generation.
All four evidence sources are used in combination.

## Process

1. RTL analysis (v4 approach) → `rtl_facts.jsonl`
2. Kernel mining (v5 approach) → `kernel_records.jsonl`, `abstraction_candidates.yaml`, `dialect_requirements.yaml`
3. Evidence graph (v3 approach) → `evidence_graph.jsonl` (merged from RTL + kernel + doc sources)
4. Schema planning (v2 approach) → `target_contract.yaml`, `dialect_plan.yaml`, `lowering_plan.yaml`
5. Deterministic generator → `generated/gemmini-mlir/xdsl/`

## What this measures

The upper bound of the structured pipeline. If v6 does not outperform the best individual method,
the combination provides no synergy. If v6 outperforms v0 significantly, the full pipeline is justified.
