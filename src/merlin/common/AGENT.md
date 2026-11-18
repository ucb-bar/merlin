# src/merlin/common

## Purpose

Shared infrastructure every layer may import: repository and `out/` paths, run and product
layout, hardware provenance, content identity, agent-access identities, record readers,
target-neutral readers of target data, adapters to upstream tools, and the two subpackages
`process/` (owned child processes) and `storage/` (generated-output storage).
Compiler IR helpers (xDSL IO, MLIR queries, compile traces, lowering audits) live in
`merlin.compiler.ir`, not here: they depend on the compiler frontend.

## What belongs here

- `paths.py`, `artifacts.py` — the roots and the run/product layout CLAUDE.md names.
- `provenance.py` — pins, observation, `verify`/`require`/`record`/`source_digest`, citations.
- `provenance_claims.py` — the registry's byte claims: built artifacts, lost bytes, and the
  registry location/error (`pins_path`, `PinsError`) the pin verifier builds on.
- `access.py` — agent-access identities and host-path scrubbing of agent-visible text.
- `digest.py` — SHA-256 helpers, canonical JSON IO, tree hashes, the stat-keyed digest memo and
  standard-library Python source membership.
- `record_formats.py` — strict JSON, deterministic YAML, schema validation, lexical tokens and the
  regex-call scan the no-regex gate uses.
- `frozen_imports.py` — the stdlib-only frozen import resolver (pinned by path; stays standalone).
- `target_data.py` — the numeric format registry, RTL-facts block views and deterministic stimulus.
- `external_tools.py` — scoped access to model2MLIR, ModeLIR and SpecIR, and the cross-toolchain
  prefix.

## What does not belong here

- Workstream-specific logic (this is shared infrastructure only).
- Anything that imports a compiler, target or experiment package at module level.
- Generated artifacts (write those under `out/`).

## Invariants

- Every subdirectory must also contain an AGENT.md.
- The common access registry contains only target-neutral identities; the experiments
  sandbox extends it with mandatory packaged historical target denials before launch.
- `frozen_imports` provides stdlib-only, process-local source import isolation for
  trusted bootstraps. Callers own snapshot seals, invocation records and grading.
  Frozen imports never fall through to a live owner or unchecked bytecode. This
  provenance boundary is not a Python sandbox and does not propagate to subprocesses
  without an explicit bootstrap. Keep experiment-specific launch policy out of core.
  Selected interpreter bootstraps may use `frozen_imports.activate` with
  `expose_roots_to_path=False`: protected namespaces and resources use their pinned
  roots without adding an outer interpreter's dependency directory to `sys.path`.
  Split protected packages still merge only their declared ordinary source roots.
  The caller owns existing search paths and the selected interpreter's dependencies;
  this option grants no dependency, sandbox or compilation authority.
- `digest.python_members` inventories ordinary Python source files for existing run
  receipts; it owns no seal, target policy or optional-distribution dependency.
- `record_formats.strict_json` rejects ambiguous keys, non-finite numbers and oversized
  authority records before consumers interpret them; it does not confer provenance or
  admission. `digest.strict_json_equal` compares closed JSON records without Python
  bool/int, float/int, or tuple/list aliases; callers still own source and receipt identity.
- `provenance_claims` records explicitly declared unrecoverable artifact identities.
  Pin and artifact loaders reject redeclaring those identities as live; build products
  use ordinary checkout-relative paths and are hashed independently of git status.
- `external_tools` owns upstream discovery/import/process adapters, not upstream compiler
  semantics. Keep PyTorch/model2MLIR capture separate from compiler Python bindings and
  ModeLIR execution. Import availability checks must not mutate process-wide paths
  permanently. The model2MLIR adapter owns capture checkout aliases and the shared
  framework-capture interpreter; keep per-workload capture.toml selection and compiler-only
  Python policy separate. Optional quant dialect imports retain class identity; conflicting
  preloaded checkouts are unavailable, never purged or reloaded to simulate multi-checkout
  isolation. The ModeLIR adapter owns scoped imports and the single artifact cwd lock.
  Retained oracle imports, new-module eviction for discovery, and serialized artifact
  execution are distinct policies. Preserve configured-root resolution inside the artifact
  lock; never move numerical/oracle semantics here. This is not multi-checkout or
  noncooperating-thread isolation. `specir_importable` scopes checkout paths while retaining
  normal module identities. Discovery remains caller-owned: capture has its sibling fallback,
  test generation may use an installed package. Explicit roots reject conflicting cached
  owners, never reload them. Test-generation startup receipts and golden-cache identity
  include this adapter's source bytes; that does not establish a commitment to every
  upstream SpecIR source dependency.
