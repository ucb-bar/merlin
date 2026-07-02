# v5_kernel_miner — Method Prompt

## Role

You are extracting **abstraction candidates** from kernel sources before designing the dialect.
The kernel corpus drives the dialect design bottom-up, rather than top-down from documentation.

## Why this matters

A dialect derived from kernels is grounded in real usage patterns. If the top 5 kernels all
require a "resident_rhs" pattern, the dialect must represent it efficiently. If no kernel
uses a feature, the dialect should not include it.

## Inputs (read-only)

- `datasets/gemmini/selected_kernels/` — baremetal Gemmini kernel sources
- `datasets/gemmini/selected_traces/` — Spike commit-log traces (optional)
- `datasets/gemmini/golden/expected_kernel_patterns.yaml` — ground-truth patterns (do not read until after extraction)

## Process

### Step 1: Extract kernel records

For each kernel in `selected_kernels/`, produce a record:
```jsonl
{"kernel": "matmul_os", "ops": ["config_ex", "mvin", "preload", "compute_preloaded", "mvout"], "reuse_pattern": "none", "tile_shape": [16, 16, 16]}
```

Write all records to `<run_dir>/contracts/kernel_records.jsonl`.

### Step 2: Mine abstraction candidates

Analyse `kernel_records.jsonl` to identify recurring patterns:
- Which op sequences appear in ≥ 2 kernels?
- Which tile shapes appear in ≥ 2 kernels?
- Which reuse patterns (resident_rhs, accumulate_k) appear in ≥ 2 kernels?

Write candidates to `<run_dir>/contracts/abstraction_candidates.yaml`.

### Step 3: Derive dialect requirements

For each abstraction candidate, write a dialect requirement:
```yaml
- candidate: resident_rhs
  evidence_kernels: [repeated_rhs_matmul, batched_conv]
  required_op: gemmini.pack_resident
  required_attr: is_resident: bool
```

Write to `<run_dir>/contracts/dialect_requirements.yaml`.

### Step 4: Build contracts

From `dialect_requirements.yaml`, derive:
- `contracts/target_contract.yaml`
- `contracts/dialect_plan.yaml`

Every op in `dialect_plan.yaml` must cite at least one kernel from `kernel_records.jsonl` as evidence.

### Step 5: (Handed off to generator)

A deterministic generator reads `dialect_plan.yaml` and emits `generated/gemmini-mlir/xdsl/`.
You do not write xDSL code.

## What this measures

Whether kernel-derived abstractions improve:
- Dialect relevance (ops match actual usage patterns)
- Held-out generalisation (do the patterns extend to unseen shapes?)
- Evidence coverage (every op traces to a real kernel)

Comparison to v3 (evidence-graph) isolates the value of kernel-mining over doc-reading.
Comparison to v2 (schema-only) isolates the value of bottom-up vs. top-down design.
