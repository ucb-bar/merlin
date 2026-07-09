# AGENT.md — merlin/schemas

## Purpose
The **cross-workstream coordination data-model** — every artifact exchanged between the workstreams
(TargetGen, kernel-mining, design-pressure/DSE, rvvgen) is defined here as a `*.schema.yaml`. The
workstreams coordinate through these schemas, not prose. Hand-authored (cannot be generated).

## What lives here
- `*.schema.yaml`, each with title, purpose, required top-level fields, and an example.
- Loaded via `merlin.common.schemas`. Presence is pinned by `check_structure.py` `REQUIRED_SCHEMAS`.
- Some are target-family-flavored (`rvv_package_manifest`, `rvv_result`) — that's the reusable schema
  for that family, not a leak.

## Distinct from `merlin/contract/schemas/`
This dir is the **loose cross-workstream YAML data-model**. `merlin/contract/schemas/*.schema.json`
is a **different family**: strict, fail-closed JSON-Schema validators for the experiment ABI. Only
`command_buffer` exists in both (kept in sync; guarded by `tests/infra/test_schema_consistency.py`).

## What does NOT belong here
- Instances/data (→ `artifacts/`, `merlin/benchmarks/`, or target dirs). Tool/analysis code.

## Invariants
Any cross-workstream artifact MUST have a schema here before code depends on it; no undocumented
JSON blobs. Add new schemas to `REQUIRED_SCHEMAS`. Each schema parses with `yaml.safe_load` and
carries an example.

Two legitimate kinds live here, and **every schema must be one of them** — a schema that is neither is
dead and gets retired (as `runtime_abi` was, to `docs/reference/runtime.md`, for having zero code
reference):
- **Validators** — instance-carrying artifacts that code builds + runs through
  `merlin.common.schemas.validate_or_raise` at the producer (most schemas: the plan family,
  `command_buffer`, `workload_region`, `evidence_report`, `dse_axis_triage`, …).
- **Vocabulary / reference specs** — the canonical structured contract that *code mirrors* rather than
  instance-validates (`metrics` ↔ `KNOWN_METRICS`/`COMMON_METRIC_NAMES` + generated C views; `trace`;
  `search_space` ↔ the DSE search/strategy modules). These are referenced by name in code, so they are
  the source of record, not prose.

The invariant (lintable, see `check_structure.py` "schema usage"): **every schema is referenced by
name in `merlin/python/`** — validated or mirrored. Zero references ⇒ dead ⇒ remove.

Mental model for the three homes: **`contract/schemas/`** = frozen fail-closed JSON ABI validators
(targetgen-only, the OOT boundary); **`schemas/`** (here) = internal YAML data-model (validators +
vocabulary specs); **`prompts/`** = versioned NL templates for LLM loops. Target-family-specific
schemas carry a family prefix (`rvv_`).
