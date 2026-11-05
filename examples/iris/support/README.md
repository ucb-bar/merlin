# iris-mlir

Merlin support for **Iris** (chipyard `iris-cy`, `IrisConfig`). This repository is a layer on top of
[`rvv-mlir`](https://github.com/ucb-bar/rvv-mlir). It inherits all of RVV and adds Iris's configuration,
its boards, and its **VME/Xsfmm matrix unit**.

```
rvv-mlir   (RVV contract, schedules)
 ├── kodiak-mlir   (+ BME OPU, + Kodiak boards)
 └── iris-mlir     (+ VME matrix unit, + Iris boards)    <- this repo
```

| Iris (from `IrisConfigs.scala`) | |
|---|---|
| harts 0–1 | Shuttle + Saturn, VLEN 256 / DLEN 128, `opuMxParams` (VME/Xsfmm matrix unit incl. MX/FP8), 128 KiB TCM each |
| hart 2 | Rocket + Saturn DMA vector unit, VLEN 256 / DLEN 256, **no matrix unit** |
| memory | 512 KiB L2, 128 KiB SBUS scratchpad @ `0x580000000`, serial-TL DRAM @ `0x80000000` |

## The VME route

int8 contractions `A(M,K)·B(K,N)` with a zero-initialised int32 accumulator, at least one tile in M and N,
are rewritten by Merlin into calls to `merlin_vme_gemm_i8_<n>`. This package supplies what those resolve to:

| file | |
|---|---|
| `contracts/matrix_units.yaml` | where the unit's facts are read from: saturn's decode table (`opuInsns`, `Instructions.scala`), the host core's `BitPat`s and `kmax` table (rocket-chip `Matrix.scala`), the config, the cross-check header (`vme.h`), and the simulator. Declaration names only, plus the register-operand layouts with their sources |
| `vme_kernel.py` | the GEMM microkernel, modeled on Miles Rusch's `vme_gemm.h` / `vme-fused-lm2-gemm-transpose`: a 2×2 super-tile of four TE×TE int32 tiles (`sf.vtzero.t`), `kmax` reduction rows per `sf.mm.s.s` (`altfmt` = B signed, `rd[0]` = A signed), readout with `sf.vtmv.v.t`, optional fused bias. Edge tiles run TE×TE with `sf.msettm`/`sf.msettn` clamped; a short last reduction step sets `sf.msettk`. No shape is refused |
| `vme_shim.py` | reads the contract, derives encodings (`merlin.targetgen.rtl.matrix_isa`), geometry (TE = VLMAX at e8 of the vector unit whose params set `useOpu`; alignment = DLEN/8), wraps the kernel in Merlin's generic shim (`merlin.kernels.matrix_shim`), compiles it, records provenance; declares the simulator |
| `matrix_lowering.py` | the plugin Merlin's `MatrixRouting` loads (`--matrix-support-target iris --matrix-unit saturn_vme --matrix-config IrisConfig`) |

Every instruction word is derived from the RTL and cross-checked against `vme.h`; with the simulator's
source present also against its `riscv/encoding.h` (all ten instructions agree on every bit both state).
The kernel reads TE and `kmax` from the hardware at run time and takes a scalar path if they are not what
its register layout was generated for.

Facts derived from the pinned sources (Merlin's `merlin/contract/hardware_pins.yaml`: `chipyard_iris`,
`saturn_vme`, `rocket_chip_iris`, `spike_zvt`): TE = 32, operand alignment 16 B, matrix configurations
(e8, w4) and (e32, w1), `kmax(8, 4) = 4`.

### What runs where

* **Shipped image**: the kernel's `.insn` words. Audited by Merlin: the unit's instructions present.
* **Stand-in twin** (`-DMERLIN_MATRIX_SCALAR_TILE`): same lowering, same tiling loop, scalar tile body,
  none of the unit's instructions; gated on stock spike.
* **Shipped image on spike with Zvt**: upstream riscv-isa-sim implements the Zvt draft the RTL follows.
  Build it (≈2 min on 32 cores) and point Merlin at it:

  ```sh
  git clone https://github.com/riscv-software-src/riscv-isa-sim && cd riscv-isa-sim
  git checkout f90d5af53f50b0feec7a7cbcb4411e1e0410c465 && git submodule update --init
  mkdir build && cd build && ../configure --prefix=$PWD/../install && make -j32 install
  export MERLIN_SPIKE_ZVT=$PWD/../install/bin/spike MERLIN_SPIKE_ZVT_SRC=$PWD/..
  ```

  It runs as `--isa=rv64gcv_zfh_zvfh_zvl256b_zvt32t_zvt8e_zvti8i32mm`. The kernel sets `mstatus.MS` first,
  as saturn's `crt.S` does; Iris's RTL has no such field and ignores the write. A spike verdict is a
  functional one under the Zvt draft's semantics, not an RTL one, and its counts are not Iris cycles.
* Matrix images use **harts 0–1 only** (`boards.yaml` `matrix_hart_ids`); a 3-hart matrix image is refused.

## Use

```sh
cd merlin
export MERLIN_CHIPYARD_IRIS=/path/to/chipyard-iris   # iris-cy, with generators/saturn and generators/rocket-chip
export MERLIN_BOARD_CATALOG=examples/board-catalog.yaml:/path/to/iris-mlir
merlin-compile --workload <model> --dtype int8 --target rvv --board iris_rtl --harts 2 --run spike --verify \
    --matrix-unit saturn_vme --matrix-config IrisConfig --matrix-support-target iris
```

`merlin-target-compose --check .` (run by `tests/`) fails when the rvv base this layer pins has moved.
`tests/test_matrix_lowering.py` checks the adapter, the contract, the kernel's refusals and the stand-in's
tiling against a reference on the host; with `MERLIN_CHIPYARD_IRIS` set it also runs the derivation.
