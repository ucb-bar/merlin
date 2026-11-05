# rvv-mlir

The **base RISC-V Vector (RVV) layer** for [Merlin](https://github.com/ucb-bar/merlin).

Everything in this repository is plain RVV. It contains no matrix-unit or OPU code and no
chip-specific facts. Chip packages build on it and add only what their chip has:

```
rvv-mlir      base: the RVV target contract (vector unit, dtypes, requant), RVV schedules
 ├── kodiak-mlir   Kodiak tapeout: Saturn + outer-product unit (OPU, BME encodings), its boards
 └── iris-mlir     Iris: 2× Shuttle+Saturn(VME) + Rocket(DMA vector), its boards (VME unit: next)
```

A chip package's `provider.yaml` says `extends: {target: rvv, commit: <sha of this repo>}`, and its
`contracts/overlay.yaml` holds only the delta. `merlin-target-compose` turns base + overlay into the
effective contract and records the exact base revision, so a chip package inherits *all* of RVV and
notices when this base moves.

## What is on which branch

| branch | what it is |
|---|---|
| `main` (this one) | the RVV **support provider** Merlin selects for target `rvv`: `provider.yaml`, `contracts/target_contract.yaml`, `tests/` |
| `stable/impr_tuned_wholemodel_vf_int8` | int8 W8A8 whole-model schedule (`vwmacc`), K1-verified; the RVV schedule the Kodiak builds use |
| `stable/sol_k1_int8` | int8 speed-of-light schedule, K1-verified, with its analysis |
| `baseline-int8_w8a8` | frozen, hand-written, unoptimized int8 control (the before/after reference) |
| `stable/rvv_tuned_v1_d1_vfmacc_outerproduct` | fp32 schedule, spike-verified |

> **Note on `rvv_tuned_v1_d1_vfmacc_outerproduct`.** "outerproduct" is the name of MLIR's
> `vector.contract` lowering strategy (`lower_contraction lowering_strategy = "outerproduct"`). It is
> **not** the Saturn outer-product unit. That package emits ordinary RVV (`vfmul.vv` + `vfadd.vv`). The
> OPU kernels are in `kodiak-mlir`.

Each schedule branch is a standalone package: `payload/schedule.mlir` (transform-dialect tiling and
vectorization of the contractions), `payload/knobs.yaml` (cflags, dtype strategy, tile sizes, LMUL policy,
expected instructions) and `.merlin/` provenance. Merlin applies one with
`merlin.mining.apply.apply_rvv_package(...)`; only the schedule and cflags change.

## Using it

```sh
git clone https://github.com/ucb-bar/merlin.git && cd merlin
uv sync --extra dev
# The Merlin checkout vendors this provider at examples/rvv/support and selects it by default.
# To use a clone of this repository instead:
export MERLIN_TARGET_PATH=/path/to/rvv-mlir
merlin-compile --workload <model> --dtype int8 --target rvv --board <board> --run spike --verify
```

For a specific chip, fetch its package. `merlin-target-fetch` follows `extends` and brings this base too:

```sh
merlin-target-fetch kodiak      # clones kodiak-mlir, then rvv-mlir at the commit kodiak pins
```

## Adding a chip on top of RVV

1. Create `<chip>-mlir` with a `provider.yaml`:
   ```yaml
   schema: merlin.provider.v2
   id: <chip>-support
   target: <chip>
   role: support
   extends:
     target: rvv
     repo: https://github.com/ucb-bar/rvv-mlir.git
     commit: <full sha of the rvv-mlir main you build on>
   ```
2. Put only the delta in `contracts/overlay.yaml`: the RTL config (`rtl_source`), extra compute units
   (a matrix unit), `matrix_contract`, and your own `plugin` entries. The base plugin block is never
   inherited, so executable support always lives in the package that declares it. Use `remove:` and
   `replace:` (lists of dotted paths) for anything the base says that is not true of your chip.
3. Put your boards in `boards.yaml` (Merlin's board schema; select with
   `MERLIN_BOARD_CATALOG=<chip>-mlir`).
4. Run `merlin-target-compose <chip>-mlir`. Commit the generated `contracts/target_contract.yaml`
   and `contracts/composed.lock.yaml`. CI runs `merlin-target-compose --check`.

Hardware revisions (RTL commits) are pinned in Merlin's single registry,
`merlin/contract/hardware_pins.yaml`, and a chip package refers to them by pin id.

## Checks

```sh
python -m pytest tests        # with Merlin installed
```

`provenance.json` records the content hash of the contract. Nothing here claims hardware execution or
candidate certification.
