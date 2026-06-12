# Role: gemmini_source_curator

## Purpose
Select and freeze a representative subset of the Gemmini hardware source into
`datasets/gemmini/source_snapshot/`, `selected_docs/`, `selected_rtl/`,
`selected_kernels/`, and `selected_traces/`.

This role is run ONCE before any official method run. The snapshot must be immutable
after curation.

## Allowed inputs (read-only)
- External Gemmini repository (chipyard/generators/gemmini/)
- Spike commit logs from `merlin/experiments/kernel_policy/stageF/`

## Allowed outputs (write)
- `datasets/gemmini/source_snapshot/` — frozen snapshot
- `datasets/gemmini/selected_docs/` — README excerpts, ISA table
- `datasets/gemmini/selected_rtl/` — ExecuteController, LoadController, StoreController
- `datasets/gemmini/selected_kernels/` — matmul, conv, im2col baremetal sources
- `datasets/gemmini/selected_traces/` — commit-log trace samples

## Forbidden modifications
- `datasets/gemmini/golden/` — golden files are set by the evaluation designer, not the curator
- Anything in `methods/`, `configs/`, `harness/`, `runs/`, `reports/`
- `merlin/`

## Validation command
Check that `datasets/gemmini/dataset_manifest.yaml` is updated with `frozen: true`.

## Success criteria
- `source_snapshot/` is non-empty and git-committed
- `dataset_manifest.yaml` updated with `frozen: true` and a `frozen_at` timestamp
- No golden files modified
