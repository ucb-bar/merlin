# AGENT.md — merlin/contract

## Purpose
The **experiment ABI**: the repo-independent, versioned contract an out-of-tree target-backend
package is built against (see `README.md`). Hand-authored, **frozen source-of-record** DATA — the
gemmini OOT-backend reference instance. The *accessor code* lives in
`merlin/python/merlin/targetgen/contract/` and reads this via `contract_dir()` / `$MERLIN_CONTRACT_DIR`.

## What lives here
- Contract specs: `*_contract.yaml`, `command_buffer_abi.yaml`, `scoring.yaml`, `integrity_policy.md`,
  `interface_grammar.md`, `VERSION`.
- `merlin_iface.irdl.mlir` — the SHARED `merlin_iface` contract-dialect spec (the IRDL a C++ OOT tool
  registers dynamically; mirrors `interface_grammar.md`). Regenerate with
  `python -m merlin.targetgen.rtl.gen_iface_irdl`. It is target-agnostic — lives here, not per-target.
  Two things to know before relying on it. (1) A dynamically registered dialect has NO custom parser,
  so it reads only the GENERIC op form; re-spell a capsule with
  `merlin.targetgen.contract.interface_emit.to_generic_form` first. The pretty form stays the contract
  surface. (2) The file carries ONLY constraints the IRDL interpreter evaluates — `irdl.c_pred` is
  silently dropped by mlir-opt and so can never fail. What it therefore does NOT check is named in its
  generated header, and stays the grader's job.
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
