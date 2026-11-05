---
title: Chip packages layered on a base target
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-05
related: [target_resolution, adding_a_target, zephyr, model2mlir]
code_refs: [examples/lib/chip_setup.py, src/merlin/targetgen/providers.py, src/merlin/targetgen/provider_compose.py, src/merlin/targetgen/oot_fetch.py, src/merlin/runtime/boards.py, src/merlin/runtime/backends/matrix_image.py, src/merlin/targetgen/rtl/matrix_isa.py, src/merlin/kernels/matrix_shim.py, src/merlin/compile_cli.py, build_tools/scripts/vendor_support.py]
---

# Chip packages layered on a base target

A chip is usually "an ISA we already support, plus something": Kodiak is RVV plus an outer-product
unit, Iris is RVV plus a VME matrix unit. A **chip package** says only the "plus". It is a support
provider that `extends` a base provider and inherits everything else:

```
rvv-mlir        base: RVV contract (vector unit, dtypes, requant)       examples/rvv/support
 ├── kodiak-mlir   + BME outer-product unit, + Kodiak boards             examples/kodiak/support
 └── iris-mlir     + VME/Xsfmm matrix unit, + Iris boards                examples/iris/support
```

Each package is its own repository; Merlin vendors a copy of each at `examples/<chip>/support` with
an exact `SOURCE.yaml`, so a checkout works with no fetch.

## Setting up a machine

Each chip example sets a machine up with one command, `examples/<chip>/run.sh setup`
(`examples/lib/chip_setup.py`, plus what that chip adds). It needs `uv` and `git`, is safe to re-run
(every step is skipped when its result is already there and still right), writes only under `out/build/`
and the repo's `.env`, and never modifies a checkout it did not create. It ends with a table of every
prerequisite and, for anything still missing, the exact commands; it exits non-zero until nothing is
left. Paths can be passed as flags or exported; whatever is found and right is recorded in `.env`
(the process environment still wins), and `run.sh` loads `.env`, so no later shell needs exports.

**Done by setup** (cheap; downloads and builds of minutes):

| what | where | size, time |
|---|---|---|
| Merlin's venv (`uv sync`) | `.venv/` | ~150 MB, under a minute |
| model2MLIR, cloned at its pin (`model2mlir_capture`) unless `--m2m`/`MERLIN_M2M_DIR` names one | `out/build/ext/model2MLIR/` | ~50 MB |
| capture venv: Python 3.12, PyTorch nightly (CPU), `torch-mlir==20260531.828` (the wheel matching the LLVM pin), torchao, timm; model2MLIR on its path by a `.pth`, so a shared or read-only checkout works | `out/build/venv/model2mlir/` | ~1.3 GB, 2-5 min |
| Zephyr west workspace: `zephyr-chipyard-sw` (`dev`), its `zephyr_ws/zephyr` submodule, `west init --mf west-riscv.yml`, `west update` (skipped when `ZEPHYR_BASE` is set) | `out/build/ext/zephyr-chipyard-sw/` | ~2 GB, 5-10 min |
| a chip's spike model of its unit, when its package declares one (Kodiak's OPU) | `out/build/spike-ext/<chip>/` | seconds |
| a simulator built from a verified pinned source (Iris's riscv-isa-sim with Zvt) | `out/build/<pin>/` | ~0.3 GB, ~2 min on 32 cores |

`--no-fetch` prints the clone and west commands instead of running them.

**Found and verified, never installed** (large, or the location is yours to choose):

| what | flag / variable | how to get it | size, time |
|---|---|---|---|
| LLVM/MLIR 23 (`clang-23`, `mlir-translate`) at llvm-project `a47bddcc` | `--llvm` / `MERLIN_MLIR_INSTALL` or `MERLIN_CLANG` | [`llvm_toolchain.md`](llvm_toolchain.md), two-target recipe | ~13 GB, tens of minutes on many cores, hours on a laptop |
| chipyard with riscv-tools (spike, `riscv64-unknown-elf-gcc`, cmake, ninja) | `--chipyard` / `MERLIN_CHIPYARD` | `git clone https://github.com/ucb-bar/chipyard.git && ./build-setup.sh riscv-tools` | ~15 GB, about an hour |
| Zephyr SDK 0.17.0 (riscv64 toolchain) | `--zephyr-sdk` / `ZEPHYR_SDK_INSTALL_DIR` | printed by setup (minimal SDK + `setup.sh -t riscv64-zephyr-elf`) | ~1 GB, minutes |
| the chip's pinned hardware checkouts | per pin (`root_env` in `merlin/contract/hardware_pins.yaml`) | printed by setup, derived from the pin | see the chip's README |

A **pinned checkout** a chip reads its facts from (`--checkout-pin <pin>` in its `run.sh`) is looked for
at the pin's `root_env`, else at `out/build/ext/<pin>/`, and verified by content with
`merlin.common.provenance.verify`: the commit, the branch when the pin names one, the required paths, and
every pin nested in it (`nested_in`: the gitlink the superproject records and the submodule's HEAD).
Verified, it is recorded in `.env`. On another revision, setup prints the drift and the commands that
would move it (it never moves it). Absent, it prints the clone commands, generated from the pin
(`repo_canonical`, `branch`, `commit`, the nested pins' paths) and aimed at `out/build/ext/<pin>/`, where
the next run finds it. A simulator is built (`--build-spike <pin>=<VAR>`) only from a verified source,
out of tree into `out/build/<pin>/`, with a `merlin-build.json` recording the commit, the source digest
and the binary's sha256; it is rebuilt only when one of those changes.

A pin inside a prerequisite you configured (`--verify-pin`, e.g. Kodiak's `saturn_opu_int8` inside
`$MERLIN_CHIPYARD`) is verified and reported as a warning when it drifts: the flow still runs, and every
build records the revision it read.

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

### What Merlin gives a matrix unit, and what the package brings

| Merlin (generic) | the chip package |
|---|---|
| `targetgen.rtl.matrix_isa`: encodings as MATCH/MASK pairs from a decode table (`props` keys) and `BitPat` declarations, cross-checked against an `.insn r` header and optionally a simulator's `MATCH_*`/`MASK_*` table; geometry from the vector unit whose params enable the unit | `contracts/matrix_units.yaml`: which files and declaration names to read (addresses, never values) |
| `kernels.matrix_shim`: the memref-descriptor bridge, the K-major pack of A, the dense/aligned/fits checks with a counted scalar fallback, batched entries | the kernel: one C function `(c, at, b, bias, m, n, k)` with K-major operands, and its scalar stand-in |
| `kernels.decode.opu`: the audit, matching a word on exactly the bits its decoder looks at | the roles and pairs naming which derived instruction is which |
| `matrix_image.build_matrix_pair`, `routing_report.json`, the twin gate | `device_simulator(unit=, config=)` (optional): a simulator that executes the unit |

An instruction is identified by the bits its decoder reads, not by `(opcode, funct3, funct6)`: a unit
can keep several instructions in one funct6 slot and tell them apart by `rs2` (Iris's `sf.vtzero.t`,
`sf.vtmv.v.t` and `sf.vtdiscard` do), and the host core declares its configuration instructions as whole
32-bit patterns. Bits the RTL decoder ignores but an `.insn r` must spell (a `vm` bit) come from the
author's header and are recorded as such.

### Harts without the unit

A board can say which harts carry the matrix unit with `matrix_hart_ids` (default: every hart a vector
image may use). A routed image splits its tile loop over every worker, so `build_matrix_pair` refuses a
matrix image (and its twin) whose workers would include another hart. Iris declares `[0, 1]`: its third hart has
a vector unit (Rocket's DMA unit) and no matrix unit.

### Running the shipped image

When the package's plugin declares `device_simulator`, `merlin-compile --matrix-unit ... --run spike` also
runs the SHIPPED image on that simulator with the unit's ISA extensions and gates it
(`matrix.shipped_on_simulator` in the result). Declared but absent, the leg is `not_run` with the reason;
it never falls back to a simulator without the unit. Iris declares upstream riscv-isa-sim with Zvt (pin
`spike_zvt`, binary `$MERLIN_SPIKE_ZVT`); Kodiak declares none, so its shipped image is gated only by the
audit and the twin.
