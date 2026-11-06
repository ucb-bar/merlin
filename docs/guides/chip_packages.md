---
title: Chip packages layered on a base target
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-05
related: [target_resolution, adding_a_target, zephyr, model2mlir]
code_refs: [examples/lib/chip_setup.py, examples/lib/preflight.py, src/merlin/common/pin_drift.py, src/merlin/targetgen/unit_pins.py, src/merlin/targetgen/support_pins.py, src/merlin/targetgen/providers.py, src/merlin/targetgen/provider_compose.py, src/merlin/targetgen/oot_fetch.py, build_tools/upstreams/target_support.json, src/merlin/runtime/boards.py, src/merlin/runtime/backends/matrix_image.py, src/merlin/targetgen/rtl/matrix_isa.py, src/merlin/kernels/matrix_shim.py, src/merlin/compile_cli.py, src/merlin/compile/matrix_leg.py, src/merlin/targetgen/spike_extension.py, build_tools/scripts/vendor_support.py]
---

# Chip packages layered on a base target

A chip is usually "an ISA we already support, plus something": Kodiak is RVV plus an outer-product
unit, Iris is RVV plus a VME matrix unit. A **chip package** says only the "plus". It is a support
provider that `extends` a base provider and inherits everything else:

```
rvv-mlir        base: RVV contract (vector unit, dtypes, requant)       pinned; tracked copy at examples/rvv/support
 ├── kodiak-mlir   + BME outer-product unit, + Kodiak boards             pinned; fetched by examples/kodiak/run.sh setup
 └── iris-mlir     + VME/Xsfmm matrix unit, + Iris boards                pinned; fetched by examples/iris/run.sh setup
```

Each package is its own repository (`ucb-bar/<name>-mlir`). Merlin does not copy the chip packages: it
**pins** each one and fetches it at exactly that commit. Kodiak and Iris are separate tapeouts and are
never built together, but they share ONE rvv-mlir: everything chip-specific lives in the chip repository's
overlay (`replace:`/`remove:` included), never in a fork of the base.

## Pins: one record, fetched by commit

`build_tools/upstreams/target_support.json` (`pinned_support`, schema
`merlin.upstream_support_migration.v3`) records, for each chip package, `{target, repo, branch, commit,
tree}`, and ONE base pin (`rvv`). `merlin.targetgen.support_pins` reads it; no code names a target.

* **Fetch.** `merlin-target-fetch <target>`, or the chip example's `run.sh setup`, checks a package out
  at exactly its pinned commit into `out/build/generated/pinned/<target>@<sha12>/`: `git init`, `fetch
  --depth 1 origin <sha>` (a full fetch when a server refuses a commit want; the pinned commits are on
  `handwritten-implementation`, not the default branch), a detached checkout, then the commit and its tree
  are verified against the pin. Re-running it on a verified checkout does nothing and needs no network; a
  dirty or divergent checkout is refused, never reset. A bump is a new directory, so the old revision is
  never moved under a running build. `--url file:///...` fetches the same commit from a local clone (it
  is still verified against the pin); `--root` prints the selected root.
* **Single base.** A package's `provider.yaml` `extends.commit` must equal Merlin's base pin, or fetch,
  selection (`target_registry.resolve`, whichever way the package was selected) and
  `merlin-target-compose` refuse it: `kodiak-mlir composed against rvv@X; Merlin's base is rvv@Y:
  recompose kodiak-mlir against Y and bump its pin`.
* **Submodule base.** A chip repository may carry its base as a submodule at `base/rvv-mlir`, at the
  commit its `extends.commit` names (the chip repo's own test keeps the two equal). The fetch brings it
  in (`git submodule update --init --depth 1`) and refuses a gitlink that is not Merlin's base pin, so a
  build has one base, never two. A standalone clone of a chip repository needs `git clone
  --recurse-submodules`; inside Merlin the base is the same pinned rvv either way.
* **The base itself.** The rvv base keeps its tracked copy at `examples/rvv/support` (its `SOURCE.yaml`
  commit must equal the base pin; a test holds them together), because flows outside the chip examples
  read the `rvv` support contract with `MERLIN_TARGET_PATH` unset (the measurement authority of the RVV
  mining beam, for one). `merlin-target-fetch rvv` fetches the same commit, and the tracked tree equals
  the pinned tree.
* **Selection.** `MERLIN_TARGET_PATH` still wins. Unset, the default is every in-repo
  `examples/*/support` plus each pinned package fetched at its pin, keyed by its `provider.yaml` target. A
  pinned package that is not fetched is an error naming `examples/<chip>/run.sh setup`, never a silent
  fallback to another copy. A stale revision on the shelf is never selected.
* **Agent-private.** Fetched checkouts are support bytes like vendored ones: agent sandboxes, bundle
  snapshots and clean rooms withhold everything on the pinned shelf (`target_registry.fetched_support_dirs`,
  masked by `merlin.targetgen.sandbox.answer_surfaces`).

### Editing a package

Commit in `<chip>-mlir`, publish the commit, and bump its one sha in Merlin:

```sh
python build_tools/scripts/vendor_support.py pin kodiak <sha>   # fetches it, records commit + tree
examples/kodiak/run.sh setup                                    # fetches the new pin
```

`pin` refuses a commit whose `extends.commit` is not the base pin. Moving the base is the one change that
touches every package: re-vendor `examples/rvv/support` (`vendor_support.py vendor`, which moves the base
pin with it), recompose each chip package against the new base, and bump each pin.

## Setting up a machine

Each chip example sets a machine up with one command, `examples/<chip>/run.sh setup`
(`examples/lib/chip_setup.py`, plus what that chip adds), and the first thing it does is fetch the chip's
support package at its pin (`--support <chip>`). It needs `uv` and `git`, is safe to re-run
(every step is skipped when its result is already there and still right), writes only under `out/build/`
and the repo's `.env`, and never modifies a checkout it did not create. It ends with a table of every
prerequisite and, for anything still missing, the exact commands; it exits non-zero until nothing is
left. Paths can be passed as flags or exported; whatever is found and right is recorded in `.env`
(the process environment still wins), and `run.sh` loads `.env`, so no later shell needs exports.
Because the environment wins, a stale export (a `ZEPHYR_BASE` left from an old workspace) would replace
what setup just verified in every later command; setup ends with a warning naming each such variable,
and `preflight` lists every setting with its effective value and where it came from (`.env`, or a shell
export, flagged `!` when it overrides what `.env` records). `unset` them. `run.sh` lists the keys it
loaded from `.env` in `MERLIN_DOTENV_KEYS`, which is how the two tell a recorded value from an export.
A `/path/to/...` placeholder (a `.env` copied from `.env.example`) is treated as unset.

**Done by setup** (cheap; downloads and builds of minutes):

| what | where | size, time |
|---|---|---|
| Merlin's venv (`uv sync`) | `.venv/` | ~150 MB, under a minute |
| the chip's support package at its pinned commit, with its base (`--support`) | `out/build/generated/pinned/<chip>@<sha12>/` | a few MB, seconds |
| model2MLIR at its pin (`model2mlir_capture`): cloned unless `--m2m`/`MERLIN_M2M_DIR` names one; setup's own clone, when clean, is moved onto the pin when the pin moves | `out/build/ext/model2MLIR/` | ~50 MB |
| capture venv: Python 3.12; `torch==2.14.1`, `torchvision==0.29.1`, `torchao==0.18.0` from PyTorch's CPU index (it keeps every release; the nightly index keeps about two months); `torch-mlir==20260531.828` (the wheel matching the LLVM pin); model2MLIR's libraries (xdsl, structlog, ...) at the versions its `uv.lock` records, the rest at setup's own pins (`CAPTURE_UNLOCKED_PINS`); model2MLIR on its path by a `.pth`, so a shared or read-only checkout works. Setup checks the torch / torch-mlir pair by importing an exported module, and writes what it installed to `merlin-capture-env.json` | `out/build/venv/model2mlir/` | ~1.3 GB, 2-5 min |
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

A pin inside a prerequisite you configured (`--verify-pin`, and every pin the fetched package's matrix
units are read from, e.g. Kodiak's `saturn_opu_int8` inside `$MERLIN_CHIPYARD`) is verified by what a
build reads from it.

### Which drift blocks

Each pin's drift is sorted by whether it changes a result or only its citation
(`merlin.common.pin_drift.materiality`):

| pin | drift that only warns (results right, recorded) | drift that blocks |
|---|---|---|
| a matrix unit's sources (`saturn_opu_int8`, `saturn_vme`, `rocket_chip_iris`; from the package's matrix contract, `merlin.targetgen.unit_pins`) | another commit or branch whose every file the encoding and geometry derivation reads (the unit's `sources`, `config_scala`, `mixin_scala`, the pin's `requires_paths`) is byte-identical to the pin's, or to a local edit the pin declares | any such file different or unreadable, a required file missing: every build that routes to the unit refuses it (`build_matrix_pair`, a chip example's `op`) |
| `model2mlir_capture` | none | any: another commit or a changed read file. Its read set is a sample of what a capture depends on, so captures refuse it (`require_capture_pin`); a bundle records the revision that captured it, and `op` recaptures one made off the pin |
| a pinned checkout setup finds (`--checkout-pin`: `chipyard_iris`, `spike_zvt`) | none | any: setup reports `drift` and builds no simulator from it |

A blocked stage prints the commands that put the checkout on its pin (from the registry:
`repo_observed` or `repo_canonical`, `branch`, `commit`). To build from it anyway, name the pin in
`MERLIN_ALLOW_OFF_PIN` (comma-separated, shell or `.env`): the stage proceeds and what it builds records
the drift. `preflight` lists each of these pins as `ok`, `warn` or `stop`.

## What a chip package contains

| file | |
|---|---|
| `provider.yaml` | `schema: merlin.provider.v2`, `extends: {target, repo, commit}`: the base and the exact revision this layer was composed against (Merlin's base pin) |
| `base/rvv-mlir` | optional submodule: the base at `extends.commit`, so a standalone clone composes and tests on its own |
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

The base is found without fetching: the layer's own checked-out submodule base first, then
`MERLIN_TARGET_PATH` (unset: the in-repo and fetched pinned providers), `out/build/generated/`, or
vendored under `examples/*/support`. If `extends.commit` is set, the base's recorded revision must equal it
(a vendored base's revision is its `SOURCE.yaml` commit; a clone's or submodule's is its `HEAD`), and it
must be Merlin's base pin. `--check` recomputes without writing and exits 1 on drift, so a moved base or
an edited overlay cannot reach a build unnoticed.

```sh
merlin-target-fetch kodiak          # kodiak-mlir at its pin (+ base submodule), and the rvv base pin
merlin-target-compose "$(merlin-target-fetch kodiak --root)" --check
```

## Boards travel with the chip

`MERLIN_BOARD_CATALOG` takes a `:`-separated list. An entry is a catalog file or a support-provider
root, meaning its `boards.yaml`:

```sh
export MERLIN_BOARD_CATALOG=examples/board-catalog.yaml:$(merlin-target-fetch kodiak --root)
```

The chip examples' `run.sh` do this for you (`ex_select_support`), and stop with the setup command when
the package is not fetched.

A board declared by two selected catalogs is refused, not resolved by order.

## Adding a chip

1. Create `<chip>-mlir` with `provider.yaml` (`extends: rvv`, `commit:` Merlin's base pin),
   `contracts/overlay.yaml` (`rtl_source`, extra `compute_units`, `matrix_contract`, `plugin`, measurement
   substrates), and `boards.yaml` with facts read from the chip's own config, never guessed. Optionally add
   the base as a submodule at `base/rvv-mlir`, at the same commit.
2. `merlin-target-compose .` and commit the generated contract and lock.
3. Pin every hardware revision the package reads in `merlin/contract/hardware_pins.yaml`.
4. Publish the commit and pin it: add `{target, repo, branch, commit, tree, extends, example}` under
   `pinned_support.packages` in `build_tools/upstreams/target_support.json` (later moves:
   `vendor_support.py pin`), and give the example a `run.sh` whose `setup` passes `--support <chip>`.
   `merlin/tests/infra/test_support_pins.py` checks the record; a host that has fetched the package also
   runs the package's own suite against this checkout.

When the base moves, re-run `merlin-target-compose` in each layer, bump `extends.commit`, and bump each
pin. Until then, every layer over the old base is refused with the command above.

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

A package can declare a simulator that executes its unit, in one of two ways: its plugin's
`device_simulator(unit=, config=)` names a simulator binary and the ISA extensions that name the unit,
or its contract's `runner.spike_extension` names a spike `--extlib` model of the unit (extension name,
the variable holding the `.so`, and its digest or the inputs it must be built from; see
`merlin.targetgen.spike_extension`). Iris declares the first (upstream riscv-isa-sim with Zvt, pin
`spike_zvt`, binary `$MERLIN_SPIKE_ZVT`); Kodiak declares the second (its OPU model, built by
`run.sh setup` and recorded as `$MERLIN_KODIAK_OPU_SPIKE_EXTLIB`).

Either way, every matrix build that runs its twin on spike also runs the SHIPPED image on the declared
simulator and gates it: `merlin-compile --matrix-unit ... --run spike`, a chip example's `op`, and
`make_delivery.py --matrix-harts` (a chip example's `build --opu`/`package --opu`). The shipped image is
graded against the same references and must match the twin bit for bit (`matrix.shipped_on_simulator`
in the result and in the package manifest). A mismatch, or a shipped image that prints no output, fails
the image. Declared but unusable (the binary or `.so` is absent, or its digest differs), the leg is
`not_run` with the reason; it never falls back to a simulator without the unit. With no declaration it
is `not_declared`, and the image is gated by the audit and the twin alone.
