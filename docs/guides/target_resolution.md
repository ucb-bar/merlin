---
title: Selecting a target definition package
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-27
related: [adding_a_target, generated_target_repos, targetgen, target_publishing]
code_refs: [merlin/python/merlin/targetgen/target_registry.py, merlin/python/merlin/targetgen/capability_manifests.py]
---

# Selecting a target definition package

A **target definition** in Merlin is a self-contained *out-of-tree package* — a directory that ships
its own capability contract and dialect plan. Nothing target-specific is committed into the Merlin
core to describe a piece of hardware; the package *is* the description, and the same layout is the
interchange format you publish, pin, version, and clone.

```
<target>-mlir/                        # any name / location — the package is identified by its contract
└── contracts/
    ├── target_contract.yaml          # the capability manifest (endpoint_kind, mesh, dtypes, encoding…)
    └── dialect_plan.yaml             # the ops/types/lowerings derived from the compute units
```

This is exactly what `merlin.targetgen.capability_manifests.write_oot_target(name, dir)` emits and what
a published `<target>-mlir` repository contains. The target's **name** comes from the contract's `name:`
field — not the directory name — so a package can live anywhere and be versioned however you like.

## How Merlin picks *which* package to use

`merlin.targetgen.target_registry.resolve(name)` walks an **ordered search path** and takes the **first**
package whose contract `name` matches. Precedence, highest first:

| # | Source | `kind` | Use it for |
| - | ------ | ------ | ---------- |
| 1 | **`MERLIN_TARGET_PATH`** entries | `external` | **Explicit selection** — a specific versioned/named package, or a repo you cloned yourself. Always wins. |
| 2 | in-tree `merlin/targets/<name>/` | `reference` | The curated package shipped inside Merlin (e.g. `gemmini`). |
| 3 | `out/build/generated/<name>/` | `external` | The **freshly generated** package — dropped here by onboarding / `write_oot_target`, so a just-generated target resolves with **zero env**. |
| 4 | `out/artifacts/targets/<name>/` | `generated` | Legacy generated location (fallback). |

`MERLIN_TARGET_PATH` is an `os.pathsep`-separated list, read left-to-right; each entry is either a
package root (has `contracts/target_contract.yaml`) or a **directory of** such roots (its immediate
children are scanned). So one entry can point at a single pinned package or at a whole shelf of them.

> The target **definition** is a *public input* (the compiler is told the ISA/mesh/dtypes), so it lives
> **outside** the champion/answer-surface tree `out/artifacts/targets/<name>/` that the eval launcher
> locks. Keeping the definition on the search path — never inside the answer surface — is what lets the
> launcher read the contract to build each arm's prompt.

## The three common cases

**1 — Use the one you just generated (default, zero config).** Onboarding and
`setup_atlas.py` materialize the package into `out/build/generated/<name>/`, so it is picked up
automatically:

```bash
python build_tools/scripts/setup_atlas.py            # writes out/build/generated/atlas/
python -c "from merlin.targetgen.target_registry import resolve; print(resolve('atlas').contract_path)"
# -> out/build/generated/atlas/contracts/target_contract.yaml   (kind=external)
```

**2 — Pin a specific version / name.** Materialize (or publish) the package under any name/location and
select it explicitly — this overrides the generated default:

```bash
python -c "from merlin.targetgen.capability_manifests import write_oot_target; \
           write_oot_target('atlas', 'out/build/generated/atlas-v0.3-abc1234')"
export MERLIN_TARGET_PATH=$PWD/out/build/generated/atlas-v0.3-abc1234
```

**3 — Bring your own clone.** A user who clones our published, target-specific optimizations repo
(same package format) alongside Merlin just points at it — nothing else changes:

```bash
git clone git@github.com:ucb-bar/atlas-mlir.git ~/atlas-mlir
export MERLIN_TARGET_PATH=~/atlas-mlir            # or a dir holding several such repos
```

Because selection is a search path, you can also stack them —
`MERLIN_TARGET_PATH=~/my-atlas:$PWD/vendored-targets` — and the leftmost match wins.

## Materializing a package

```bash
# Generic, any target that has a capability manifest builder or a descriptor + CIRCT facts:
python -c "from merlin.targetgen.capability_manifests import write_oot_target; write_oot_target('<name>', '<dir>')"

# Atlas, as part of setup (default dir = out/build/generated/atlas; --target-package-dir to choose):
python build_tools/scripts/setup_atlas.py --target-package-dir out/build/generated/atlas
```

The contract inside the package is **derived** — endpoint kind, mesh, and encoding come from the CIRCT
facts (see `capability_manifests.derive_manifest`); only what the RTL cannot yet ground is a
provenance-tagged residual. Regenerating from the same facts is deterministic.

## Related

- [Adding a target](adding_a_target.md) — the end-to-end onboarding flow.
- [Generated target repositories](../reference/generated_target_repos.md) — the full package layout.
- [Target publishing](target_publishing.md) — publishing a champion package to its `<target>-mlir` repo.
