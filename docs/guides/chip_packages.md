---
title: Chip packages layered on a base target
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-04
related: [target_resolution, adding_a_target, zephyr, model2mlir]
code_refs: [src/merlin/targetgen/providers.py, src/merlin/targetgen/provider_compose.py, src/merlin/targetgen/oot_fetch.py, src/merlin/runtime/boards.py, src/merlin/runtime/backends/matrix_image.py, build_tools/scripts/vendor_support.py]
---

# Chip packages layered on a base target

A chip is usually "an ISA we already support, plus something": Kodiak is RVV plus an outer-product
unit, Iris is RVV plus a VME matrix unit. A **chip package** says only the "plus". It is a support
provider that `extends` a base provider and inherits everything else:

```
rvv-mlir        base: RVV contract (vector unit, dtypes, requant)       examples/rvv/support
 ├── kodiak-mlir   + BME outer-product unit, + Kodiak boards             examples/kodiak/support
 └── iris-mlir     + Iris config and boards (VME unit next)              examples/iris/support
```

Each package is its own repository; Merlin vendors a copy of each at `examples/<chip>/support` with
an exact `SOURCE.yaml`, so a checkout works with no fetch.

## What a chip package contains

| file | |
|---|---|
| `provider.yaml` | `schema: merlin.provider.v2`, `extends: {target, repo, commit}`: the base and the exact revision this layer was composed against |
| `contracts/overlay.yaml` | the delta to the base contract (the only contract file you edit) |
| `contracts/target_contract.yaml` | the effective contract, **generated** by `merlin-target-compose` |
| `contracts/composed.lock.yaml` | base revision, base contract digest, overlay digest, effective digest |
| `boards.yaml` | the chip's boards, in the [board schema](zephyr.md) |
| plugins, e.g. `matrix_lowering.py` | executable support the overlay's `plugin:` block names |
| `tests/`, `provenance.json` | the package's own checks |

## Overlay merge rules

`merlin-target-compose <root>` computes base ⊕ overlay:

* mappings merge recursively; the overlay wins on conflicts;
* lists of mappings that all carry `name` (e.g. `compute_units`) merge item by item;
* lists of scalars (e.g. `features`) are a union;
* `remove: [dotted.path, ...]` drops base facts that are not true of the chip, and
  `replace: [dotted.path, ...]` takes the overlay's value wholesale;
* the base's `plugin` block is **never inherited**: a plugin reference must resolve inside the package
  that declares it, so a layer declares its own;
* `name` is always the layer's target.

The base is found without fetching: on `MERLIN_TARGET_PATH`, in `out/build/generated/`, or vendored
under `examples/*/support`. If `extends.commit` is set, the base's recorded revision must equal it (a
vendored base's revision is its `SOURCE.yaml` commit; a clone's is its `HEAD`). `--check` recomputes
without writing and exits 1 on drift, so a moved base or an edited overlay cannot reach a build
unnoticed.

```sh
merlin-target-compose examples/kodiak/support --check
merlin-target-fetch kodiak          # clones kodiak-mlir, then rvv-mlir at the commit it pins
```

## Boards travel with the chip

`MERLIN_BOARD_CATALOG` takes a `:`-separated list. An entry is a catalog file or a support-provider
root, meaning its `boards.yaml`:

```sh
export MERLIN_BOARD_CATALOG=examples/board-catalog.yaml:examples/kodiak/support
```

A board declared by two selected catalogs is refused, not resolved by order.

## Adding a chip

1. Create `<chip>-mlir` with `provider.yaml` (`extends: rvv`, pinned commit), `contracts/overlay.yaml`
   (`rtl_source`, extra `compute_units`, `matrix_contract`, `plugin`, measurement substrates), and
   `boards.yaml` with facts read from the chip's own config, never guessed.
2. `merlin-target-compose .` and commit the generated contract and lock.
3. Pin every hardware revision the package reads in `merlin/contract/hardware_pins.yaml`.
4. Vendor it: `python build_tools/scripts/vendor_support.py --repo ../<chip>-mlir --example <chip> --url <repo url>`.
   This copies the provider root at one commit byte-for-byte and records the tree in `SOURCE.yaml` and
   `build_tools/upstreams/target_support.json`; `merlin/tests/infra/test_example_support.py` recomputes it.

When the base moves, re-run `merlin-target-compose` in each layer, bump `extends.commit`, and
re-vendor. The `--check` in each layer's tests fails until that is done.

## A matrix unit in a chip package

A chip package exposes a matrix unit to the whole-model route through `plugin.matrix_lowering`
(`geometry`, `selector`, `rewrite_prepared_file`, `load_signatures`, `build_object`). Select it on a
build with `--matrix-unit/--matrix-config/--matrix-support-target`, on both `merlin-compile` and
`build_tools/scripts/make_delivery.py`. Both build through
`merlin.runtime.backends.matrix_image.build_matrix_pair`:

* the shipped image, whose selected int8 contractions call the unit's kernel;
* a stand-in twin, with the same lowering (checked by digest) and a scalar tile body, which is what a
  functional simulator runs and grades;
* `routing_report.json`, listing what was routed, what was not and why (dtype, a transposed operand,
  an accumulator that is not zero, smaller than one tile), and the share of contraction MACs on the unit.

Only contractions whose operands are exactly A(M,K)·B(K,N) are routed. A transposed operand stays on
the vector unit and is reported.
