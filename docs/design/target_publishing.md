---
title: Target publishing — the "target becomes its own repo" bridge (WS-E)
kind: design
status: current
owner: core
last_verified: 2026-07-15
related: [repo_structure, integrations]
code_refs: [merlin/python/merlin/targetgen/publish.py, merlin/targets/publish.yaml, merlin/contract/schemas/manifest.schema.json, merlin/python/merlin/targetgen/oot_runner.py]
---

# Target publishing — the layered "target becomes its own repo" bridge

Merlin tracks **every** codegen experiment internally under
`out/artifacts/targets/<target>/<package_id>/`. WS-E adds the bridge that exports the **certified
champion** package for a target as its own standalone, cloneable, buildable git repository —
`merlin.targetgen.publish` (console script `merlin-target-publish`).

## The layered model (locked decision)

The publish is **layered**, not a manifest dump:

- **The buildable out-of-tree (OOT) tree IS the repo content.** A consumer does `git clone` +
  `cmake -S . -B build && cmake --build build` and gets `build/bin/<target>-opt`. No Merlin checkout
  is required to build the target.
- **The package manifest + provenance ride along** as a release/provenance record under `.merlin/`.

Internally, all experiments stay tracked; publicly, **only the certified champion per target** is
exported. The single-champion invariant is enforced by `promote()`.

## Canonical repo layout

The same skeleton is emitted for every target; `family` decides which directories are populated.

```
<target>-mlir/
  README.md                 # generated: what / how-to-build / provenance summary
  CMakeLists.txt            # buildable OOT tree at ROOT
  include/<Dialect>/        #   rvv: placeholder + .gitkeep; gemmini: hoisted from mlir_oot/
  lib/
  tools/<target>-opt/       #   rvv: thin driver; gemmini: hoisted
  test/
  manifest.yaml             # rewritten contract manifest (repo root == {package})
  payload/                  # family assets:
                            #   rvv     -> schedule.mlir + knobs.yaml + baseline_runs/
                            #   gemmini -> dialect.py, lowering.yaml, contracts/, inputs/
  .merlin/                  # committed METADATA layer:
    manifest.yaml           #   identical to the root manifest (provenance copy)
    provenance.yaml         #   lineage / source package / run refs / merlin sha
    certification.yaml      #   recorded certification (oot_runner.certify / rvv spike gate)
    CHAMPION                #   one line: <package_id> <sha7> <cert_run_id>
```

### Families

- **rvv** (`family: vector_schedule`, no dialect): the payload is `schedule.mlir` + `knobs.yaml` +
  `baseline_runs/`. `tools/rvv-opt/` is a thin contract-shaped driver; `include/`/`lib/` carry a
  placeholder + `.gitkeep`. The publish bridge **synthesizes** a contract manifest around the
  generated buildable skeleton (the source rvv package uses the `rvv_package_manifest` schema, which
  is not contract-shaped).
- **gemmini** (`family: tensor_resident` / `mlir_oot_target_backend`): the existing `mlir_oot/` tree
  is **hoisted** to the repo root; `dialect.py`/`lowering.yaml`/`contracts/`/`inputs/` move under
  `payload/`. The contract manifest's build paths are rewritten so **repo root == `{package}`**:
  `{package}/mlir_oot` → `{package}` and `mlir_oot/build/bin/...` → `build/bin/...`. The rewrite is a
  structured token/string replacement (no regex) so the embedded `.merlin/manifest.yaml` stays
  self-consistent and `oot_runner.build_package` works verbatim on a fresh clone.

`manifest.yaml` carries `layout_version` + `family`. It lives at the repo **root** (so
`oot_runner.load_package(<clone>)` resolves `{package}` = repo root) and is mirrored byte-for-byte
into `.merlin/manifest.yaml` as the committed provenance copy.

## Certification gate

`publish()` refuses to export an uncertified champion unless `--no-gate` is passed (which emits a
loud warning):

- **gemmini / mlir_oot**: an `oot_runner.certify` pass, recorded as `status: rtl_certified` or
  `publication.certification == "pass"`.
- **rvv**: the recorded `spike_verified` (or better) gate.

## Git mechanics & idempotency

A real publish clones the resolved remote, replaces its working tree with the assembled repo,
makes **one commit per promotion** whose message embeds the provenance (champion package_id,
internal run id, Merlin git sha, certification summary), and adds an annotated
`v<version>-<package_id>` tag. **History is the provenance trail; HEAD is the current champion.**

Idempotency uses a fingerprint `sha256(package_id + merlin_sha + cert_run_id)` carried in the commit
trailer (`Merlin-Publish-Fingerprint:`) and the tag: if the remote HEAD trailer matches or the tag
already exists, the publish is a no-op. The `gh release` step is skipped for `file://` bare remotes.

## Remote resolution (never hardcoded)

`resolve_remote()` precedence, highest first: `--remote` override, then
`MERLIN_PUBLISH_REMOTE_<TARGET>` (via `merlin.common.paths.env`, honoring `.env`), then
`merlin/targets/publish.yaml`. The two ucb-bar remotes (`rvv-mlir`, `gemmini-mlir`) are wired as
config only; the bridge is verified against a **local bare remote** (`file://…`) and is never pushed
to GitHub by the harness.

## Provenance schema

`manifest.schema.json` gains an additive `publication` block (`champion`, `certification`,
`certified_by_run`, `promoted_at`, `promoted_by`, `fingerprint`); the schema keeps
`additionalProperties: true`, so `targetgen.contract.schemas.validate_manifest` is unaffected.

## What is reused (not re-implemented)

- `merlin.common.artifacts` — `utc_stamp` / `git_sha7` / `new_product` (each publish event is
  recorded as a versioned product under `out/artifacts/publish/<target>/`).
- `merlin.targetgen.oot_runner` — `load_package` / `build_package` (the fresh-clone build verify).
- `merlin.rvvgen.registry.load_rvv_package` — reads the rvv payload.
- `merlin.common.paths` — `artifacts_dir` / `build_dir` / `repo_root` / `env` / `targets_dir`.
  Staging + verification dirs live under `out/build/publish/`.
