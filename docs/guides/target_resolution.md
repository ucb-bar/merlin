---
title: Selecting a target definition package
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-10
related: [adding_a_target, generated_target_repos, targetgen, target_publishing, neutral_runtime_tooling]
code_refs: [src/merlin/targetgen/target_registry.py, src/merlin/targetgen/providers.py, src/merlin/targetgen/capability_manifests.py, src/merlin/runtime/backends/base.py, build_tools/upstreams/target_support.json]
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

## Provider roles and ownership

Support packages declare `provider.yaml` at their own root, separate from a compiler candidate's
`manifest.yaml`. An independently selected repository can use the declaration at
`merlin-support/provider.yaml`. Gemmini has no in-repo executable support default:
its handwritten provider and copied headers have been removed from this repository.
Fresh compiler experiments use reviewed data with shared execution tooling, or an
independently reviewed execution provider that contains no compiler solution.
The handwritten compiler's support cannot substitute for independent tooling.

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

## Data-only execution tooling

`runner.backend: chipyard_rocc` selects shared Merlin tooling from the selected
contract and existing RTL facts. This route requires the logical ABI v2 and
explicit toolchain and engine pins; it refuses executable `plugin` references.
It selects no compiler, device kernel, packing routine or schedule. See
[neutral runtime tooling](../reference/neutral_runtime_tooling.md).

`MERLIN_TARGET_CONTRACT` can select a reviewed contract independently of provider
discovery. Observed sealed contract/facts snapshots take precedence in their
existing scopes. Missing or malformed data refuses; it cannot trigger a legacy
provider import. A selected contract alone does not qualify the native worker.

## How Merlin picks *which* package to use

`merlin.targetgen.target_registry.resolve(name)` walks an **ordered search path** and takes the **first**
package whose contract `name` matches. Precedence, highest first:

| # | Source | `kind` | Use it for |
| - | ------ | ------ | ---------- |
| 1 | **`MERLIN_TARGET_PATH`** entries | `external` | **Explicit selection** — a specific versioned/named package, or a repo you cloned yourself. Always wins. **Unset**, the entries are the checkout's in-repo support providers (`examples/*/support`, see below). |
| 2 | Reference metadata: checkout `examples/*/target/`, then `merlin/targets/` compatibility roots | `reference` | Inspect authored inputs; explicit `MERLIN_TARGETS_DIR` replaces this search. |
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
only from support providers selected on `MERLIN_TARGET_PATH` (or, with it unset, the in-repo
`examples/*/support` providers), including when the provider was just generated. In-tree reference
metadata and the generated home do not autoload executable plugins. Candidate-compiler and host-schedule packages cannot
substitute for support providers. Removing a loaded provider's explicit selection requires
a fresh process, even if the same directory remains discoverable as reference metadata.

The host reads the selected support contract to construct the declared public inputs
(such as ISA, mesh and dtypes). The support package itself remains private: it may
contain oracle implementations and answers. A directory's location outside the
champion tree does not make it safe to grant to a candidate.

## In-repo support: the default when `MERLIN_TARGET_PATH` is unset

Merlin's trusted, agent-private support provider for a target is tracked beside its example at
`examples/<example>/support/`. `SOURCE.yaml` states one of two ownership forms:

- `merlin.canonical_example_support.v1` binds the **current tracked support tree** and file count.
  Its external `origin` records history, not a required checkout or a byte-identity
  claim about the current tree. A change needs a reviewed new tree identity.
- `merlin.vendored_support.v1` is a historical companion snapshot. It binds the companion commit
  and source tree; only listed `normalized` files may differ, with each original blob ID recorded.
  A change to this form is a new re-vendoring from a recorded companion commit.

`merlin/tests/infra/test_example_support.py` recomputes each tree from tracked members and checks
its record against [`target_support.json`](../../build_tools/upstreams/target_support.json). Neither
form is a compiler candidate or an authorization to show the support tree to an agent.
Fresh compiler experiments additionally exclude handwritten support from the
repository and require independent hardware, minimal software and runtime issuance.

| `MERLIN_TARGET_PATH` | Support selected for target `T` |
| -------------------- | ------------------------------- |
| unset | the in-repo provider whose `provider.yaml` declares `target: T`, if any |
| `""` (set, empty) | none: executable support refuses, reference metadata still resolves |
| any other value | exactly the listed entries; the in-repo default is not consulted |

The default is keyed by each provider's **declared** target, never by its example directory name,
and is computed by `target_registry.in_repo_support()` (`default_support_root(target)` for one
target). Two examples declaring the same target raise `TargetCollisionError`; an invalid
`provider.yaml` raises instead of dropping out of the selection. An installed distribution has no
checkout and therefore no default. A caller that hands the selection to a child or prepends an entry
should use `target_registry.effective_target_path()`, which spells the default out, so an unset
variable never turns into an explicit selection that silently drops it.

With the variable unset every in-repo provider is selected at once, so the first registry query
loads each one's declared plugins. Plugin ownership is still process-immutable: changing the
selection afterwards in the same process (for example a test that sets `MERLIN_TARGET_PATH` to one
fixture) is refused for every loaded target. Select explicitly before the first query, or run in a
fresh process, when only one provider should load. For that reason the test suites
(`merlin/tests`, `packages/merlin-experiments/tests`) start with `MERLIN_TARGET_PATH=""` unless you
export a value. Support-dependent tests require an explicitly selected external
provider. From-scratch experiments separately qualify its independent derivation;
selecting a provider is not experimental admission.

The in-repo support trees remain experimenter-side. Every `examples/*/support` directory is an answer
surface whatever is selected: agent sandboxes, bundle snapshots and clean rooms withhold it (only an
explicit grant of its `contracts/` sub-tree can reach inside, and no bundle declares one), and
publication refuses a support provider as a candidate. Canonical and historical snapshot updates
follow their different `SOURCE.yaml` identity rules above.

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

**3 — Bring your own support package.** To test a support revision other than the in-repo default, clone
a target repository containing a support provider and point at that provider root (not a
candidate-only or schedule-only repository root). An explicit value replaces the in-repo default for
every target, so list each provider the process needs:

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
