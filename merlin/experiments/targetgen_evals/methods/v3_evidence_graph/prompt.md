# v3_evidence_graph — Method Prompt

## Role

You are building an **evidence graph** from the source snapshot before writing any schemas.

## Process

1. Read every file in `datasets/gemmini/selected_docs/`, `selected_rtl/`, `selected_kernels/`, `selected_traces/`.
2. For each hardware concept you find, emit an evidence record:
   ```json
   {"concept": "scratchpad", "source": "selected_rtl/ExecuteController.v:42", "quote": "..."}
   ```
3. Emit all records to `<run_dir>/contracts/evidence_graph.jsonl` (one JSON object per line).
4. From the evidence graph, derive `target_contract.yaml` with every claim citing an evidence record.
5. From the contract, derive `dialect_plan.yaml`.

## Constraint

Every op in `dialect_plan.yaml` must trace back to at least one line in `evidence_graph.jsonl`.
Ops without evidence citations are architecture violations (R5).

## What this measures

Whether evidence-first planning reduces unsupported claims and improves coverage
compared to schema-only planning (v2).
