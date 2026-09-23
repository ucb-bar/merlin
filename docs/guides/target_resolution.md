---
title: Selecting a target definition package
kind: guide
status: current
owner: targetgen
last_verified: 2026-09-20
related: [adding_a_target, generated_target_repos, targetgen, target_publishing]
code_refs: [src/merlin/targetgen/target_registry.py, src/merlin/targetgen/providers.py, src/merlin/targetgen/capability_manifests.py]
---

# Selecting a target definition package

A **target definition** in Merlin is a self-contained *out-of-tree package* — a directory that ships
its own capability contract and dialect plan. The package describes the hardware, and the same
layout is the interchange format you publish, pin, version, and clone. Retained in-tree reference
packages are migration debt, not the intended home for new target-specific implementations.

```
<target>-mlir/                        # any name / location — the package is identified by its contract
└── contracts/
    ├── target_contract.yaml          # the capability manifest (endpoint_kind, mesh, dtypes, encoding…)
    └── dialect_plan.yaml             # the ops/types/lowerings derived from the compute units
```

This is what `merlin.targetgen.capability_manifests.write_oot_target(name, dir)` emits. A published
compiler candidate or host schedule does **not** necessarily contain these support resources.
The target's **name** comes from the contract's `name:`
field — not the directory name — so a package can live anywhere and be versioned however you like.

## Provider roles

New support packages declare `provider.yaml` at their own root, separate from a compiler candidate's
`manifest.yaml`. For example a compiler repository can own `merlin-support/provider.yaml`:

```yaml
schema: merlin.provider.v1
id: my-device-support
target: my-device
role: support
contract: contracts/target_contract.yaml
```

The contract path is relative to this provider root and must resolve inside it, including symlink
resolution. Its `name` must match `target`. The registry accepts existing contract-only packages for
compatibility. `read_provider(root)` also describes `candidate_compiler` and `host_schedule` roles,
but neither participates in target-definition discovery. Roles are claims about purpose, not trust
grants, certification, or proof that a compiler executes. Existing ABI/capability/grading checks remain
unchanged. A legacy schedule with a compiler-looking wrapper is still reported as a host schedule.

## How Merlin picks *which* package to use

`merlin.targetgen.target_registry.resolve(name)` walks an **ordered search path** and takes the **first**
package whose contract `name` matches. Precedence, highest first:

| # | Source | `kind` | Use it for |
| - | ------ | ------ | ---------- |
| 1 | **`MERLIN_TARGET_PATH`** entries | `external` | **Explicit selection** — a specific versioned/named package, or a repo you cloned yourself. Always wins. |
| 2 | Reference metadata: legacy `merlin/targets/` roots, then checkout `examples/*/target/` | `reference` | Inspect authored inputs; Gemmini's contract now lives in its example. Legacy roots win duplicate names during migration. |
| 3 | `out/build/generated/<name>/` | `external` | The **freshly generated** package — dropped here by onboarding / `write_oot_target`, so a just-generated target resolves with **zero env**. |
| 4 | `out/artifacts/targets/<name>/` | `generated` | Legacy generated location (fallback). |

`MERLIN_TARGET_PATH` is an `os.pathsep`-separated list, read left-to-right; each entry is either a
package root (has `contracts/target_contract.yaml`) or a **directory of** such roots (its immediate
children are scanned). So one entry can point at a single pinned package or at a whole shelf of them.

Reference discovery requires an actual `contracts/target_contract.yaml`; empty legacy
directories do not shadow generated packages. Example metadata is discovered only in a
physical checkout, not from an installed package's `MERLIN_REPO_ROOT` override. An explicit
`MERLIN_TARGETS_DIR` selects the reference shelf instead of adding checkout examples.
Reference discovery is shared by resolution, listing and capability-residual discovery.
It never loads the example's Python files or grants executable support authority.

Two different providers for one target within the same shelf raise `TargetCollisionError`; no
alphabetical winner is chosen. Across explicitly ordered entries, the first wins and
`target_registry.discover(entries).shadows` records the alternatives. Repeated symlinks to the same
physical provider do not create a collision.

Selection is re-evaluated by this registry, but imported runtime plugins retain their process-local
module/registration state. Select a provider before loading its backend and use a fresh process when
switching versions of the same target; this change does not introduce safe plugin hot-reloading.

Resolution is read-only: it never fetches, generates contracts, or imports provider code. The old
`MERLIN_TARGET_AUTOFETCH` variable no longer triggers writes during lookup. Explicitly call
`merlin-target-fetch` for network retrieval or `target_registry.materialize` for derivation first.

Inspection is not execution permission. Backend, dialect and simulator-oracle plugins load
only from support providers explicitly selected on `MERLIN_TARGET_PATH`, including when
the provider was just generated. In-tree reference metadata and the generated home do
not autoload executable plugins. Candidate-compiler and host-schedule packages cannot
substitute for support providers. Removing a loaded provider's explicit selection requires
a fresh process, even if the same directory remains discoverable as reference metadata.

The host reads the selected support contract to construct the declared public inputs
(such as ISA, mesh and dtypes). The support package itself remains private: it may
contain oracle implementations and answers. A directory's location outside the
champion tree does not make it safe to grant to a candidate.

## The three common cases

**1 — Use the one you explicitly generated.** Materialization can place a package in
`out/build/generated/<name>/`, where discovery finds it without an environment override:

```bash
python examples/atlas/target/setup.py --materialize-target-package
python -c "from merlin.targetgen.target_registry import resolve; print(resolve('atlas').contract_path)"
# -> out/build/generated/atlas/contracts/target_contract.yaml   (kind=external)
# Select the support package explicitly before loading any executable plugins:
export MERLIN_TARGET_PATH=$PWD/out/build/generated/atlas
```

**2 — Pin a specific version / name.** Materialize (or publish) the package under any name/location and
select it explicitly — this overrides the generated default:

```bash
python -c "from merlin.targetgen.capability_manifests import write_oot_target; \
           write_oot_target('atlas', 'out/build/generated/atlas-v0.3-abc1234')"
export MERLIN_TARGET_PATH=$PWD/out/build/generated/atlas-v0.3-abc1234
```

**3 — Bring your own support package.** Clone a target repository containing a support provider and
point at that provider root (not a candidate-only or schedule-only repository root):

```bash
export MERLIN_TARGET_PATH=/path/to/target-repo/merlin-support
```

Because selection is a search path, you can also stack them —
`MERLIN_TARGET_PATH=~/my-atlas:$PWD/vendored-targets` — and the leftmost match wins.

## Materializing a package

```bash
# Generic, any target that has a capability manifest builder or a descriptor + CIRCT facts:
python -c "from pathlib import Path; from merlin.targetgen.target_registry import materialize; materialize('<name>', destination=Path('<new-dir>'))"

# Atlas, as part of setup (default dir = out/build/generated/atlas; --target-package-dir to choose):
python examples/atlas/target/setup.py --target-package-dir out/build/generated/atlas
```

The contract inside the package is **derived** — endpoint kind, mesh, and encoding come from the CIRCT
facts (see `capability_manifests.derive_manifest`); only what the RTL cannot yet ground is a
provenance-tagged residual. Regenerating from the same facts is deterministic.
`materialize` requires a new destination and propagates derivation failures; it does not silently
overwrite an existing provider or substitute fabricated facts.

## Related

- [Adding a target](adding_a_target.md) — the end-to-end onboarding flow.
- [Generated target repositories](../reference/generated_target_repos.md) — the full package layout.
- [Target publishing](target_publishing.md) — publishing a champion package to its `<target>-mlir` repo.
