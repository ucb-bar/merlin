# AGENT.md — merlin/contract

## Purpose
The **experiment ABI**: the repo-independent, versioned contract an out-of-tree target-backend
package is built against (see `README.md`). Hand-authored, **frozen source-of-record** DATA — the
gemmini OOT-backend reference instance. The *accessor code* lives in
`merlin/python/merlin/targetgen/contract/` and reads this via `contract_dir()` / `$MERLIN_CONTRACT_DIR`.

## What lives here
- Contract specs: `*_contract.yaml`, `command_buffer_abi.yaml`, `scoring.yaml`, `integrity_policy.md`,
  `interface_grammar.md`, `VERSION`.
- `schemas/` — fail-closed JSON-Schema validators (distinct from the loose YAML data-model in
  `merlin/schemas/`; only `command_buffer` overlaps, kept in sync by a test).
- `examples/` — frozen golden inputs (g0–g2).
- `capsules/` — the gemmini reference capsule corpus (graded benchmark INPUTS). See `capsules/MANIFEST.yaml`
  for which capsules are regenerable (`generate_corpus.py`) vs hand-authored (A1, B3/B4, hidden H*).
  Kept committed regardless — agents are scored against exact goldens.

## What does NOT belong here
- Accessor/tool code (→ `merlin/python/merlin/targetgen/contract/`). Generated results (→ `artifacts/`).

## Used by
`merlin.targetgen` (capsule_runner, capsule_grade, contract.*, rtl_check_runner) + the
`gemmini_capsule_bench_v0` / `gemmini_perf_bench` experiments + `tests/gemmini/`.

## Invariants
Frozen data only; the gemmini specifics are a reference instance — keep new general machinery out.
