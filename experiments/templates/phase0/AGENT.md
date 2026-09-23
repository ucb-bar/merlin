# Phase 0 shared templates

`performance.yaml` declares the shared performance sweeps and claim policies.
Target recipes reference this file through their definition's `performance_template`.
Generated capsules and receipts stay under the configured output root. Never place
private holdouts, golden values or target-specific implementations here.
