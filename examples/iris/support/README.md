# iris-mlir

Merlin support for **Iris** (chipyard `iris-cy`, `IrisConfig`). This repository is a layer on top of
[`rvv-mlir`](https://github.com/ucb-bar/rvv-mlir). It inherits all of RVV and adds only Iris's own
configuration.

```
rvv-mlir   (RVV contract, schedules)
 ├── kodiak-mlir   (+ BME OPU, + Kodiak boards)
 └── iris-mlir     (+ Iris config and boards; VME unit next)    <- this repo
```

| Iris (from `IrisConfigs.scala`) | |
|---|---|
| harts 0–1 | Shuttle + Saturn, VLEN 256 / DLEN 128, `opuMxParams` (VME/Xsfmm matrix unit incl. MX/FP8), 128 KiB TCM each |
| hart 2 | Rocket + Saturn DMA vector unit, VLEN 256 / DLEN 256 |
| memory | 512 KiB L2, 128 KiB SBUS scratchpad @ `0x580000000`, serial-TL DRAM @ `0x80000000` |

## Status: skeleton

* **Works today:** RVV compilation for all three harts (`iris_rtl` board, `boards.yaml`), through the
  inherited rvv contract and schedules.
* **Next:** the VME matrix unit. Its instructions (`sf.mm.s.s`, `sf.vtzero.t`, `sf.vtmv.v.t`,
  `vsettnt/vsettm/vsettk`) are *not* Kodiak's BME OPU, so this layer will get its own `units:` block
  (encodings derived from saturn `vme`, cross-checked against `benchmarks/common/vme.h`), its own
  `matrix_lowering.py`, and a kernel modeled on `benchmarks/vme-fused-lm2-gemm-transpose`
  (2×2 super-tile, `sf.vtzero.t`, k4 loop, fused bias on readout, K-major A so no transpose is needed).
* **Not described:** Iris silicon or a bitstream. Boards are added when their facts are known.

## Use

```sh
cd merlin
export MERLIN_TARGET_PATH=/path/to/iris-mlir      # Merlin also vendors it at examples/iris/support
export MERLIN_BOARD_CATALOG=/path/to/iris-mlir
merlin-compile --workload <model> --dtype int8 --target rvv --board iris_rtl --harts 2 --run spike --verify
```

`merlin-target-compose --check .` (run by `tests/`) fails when the rvv base this layer pins has moved.
