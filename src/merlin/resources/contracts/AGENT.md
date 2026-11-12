# src/merlin/resources/contracts

## Purpose
The **experiment ABI**: the repo-independent, versioned contract an out-of-tree target-backend
package is built against (see `README.md`). The *accessor code* lives in
`merlin/python/merlin/development/targetgen/contract/` and reads this via `contract_dir()` / `$MERLIN_CONTRACT_DIR`.

## What lives here
- Contract specs: `*_contract.yaml`, `command_buffer_abi.yaml`, `scoring.yaml`, `integrity_policy.md`,
  `interface_grammar.md`, `VERSION`.
- `merlin_iface.irdl.mlir` — the SHARED `merlin_iface` contract-dialect spec (the IRDL a C++ OOT tool
  registers dynamically; mirrors `interface_grammar.md`). Regenerate with
  `python -m merlin.development.targetgen.rtl.gen_iface_irdl`. It is target-agnostic — lives here, not per-target.
  Two things to know before relying on it. (1) A dynamically registered dialect has NO custom parser,
  so it reads only the GENERIC op form; re-spell a capsule with
  `merlin.development.targetgen.contract.interface_emit.to_generic_form` first. The pretty form stays the contract
  surface. (2) The file carries ONLY constraints the IRDL interpreter evaluates — `irdl.c_pred` is
  silently dropped by mlir-opt and so can never fail. What it therefore does NOT check is named in its
  generated header, and stays the grader's job.
- `schemas/` — fail-closed JSON-Schema validators (distinct from the loose YAML data-model in
  `src/merlin/resources/schemas/`; only `command_buffer` overlaps, kept in sync by a test).
- `examples/` — compatibility links for Gemmini's g0–g2 reference inputs and
  expected command buffer; the source inputs belong to `examples/compiler-generation/gemmini/test-generation/reference/`
  and the expected output is a Gemmini test fixture. These are not core release data.
- `capsules/` — retained historical public corpus snapshots across multiple targets. See
  `capsules/MANIFEST.yaml` for regenerable and hand-authored members; hidden capsules and
  goldens are untracked answer surfaces. New test generation capsules are generated run artifacts,
  selected through a reviewed release rather than authored here.

## What does NOT belong here
- Accessor/tool code (→ `src/merlin/development/targetgen/contract/`). New generated results (→ `out/`).

## Used by
`merlin.development.targetgen` (capsule_runner, capsule_grade, contract.*, rtl_check_runner) + the
`gemmini_capsule_bench_v0` / `gemmini_perf_bench` experiments + `tests/gemmini/`.

## Invariants
The shared contract stays target-agnostic. Do not add new target-specific source inputs here.
Retain historical corpus paths until their consumers and privacy masks are migrated together.
