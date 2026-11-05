# kodiak-mlir

Merlin support for the **Kodiak** tapeout: Saturn vector units plus the int8 **outer-product unit
(OPU)**. This repository is a layer on top of [`rvv-mlir`](https://github.com/ucb-bar/rvv-mlir). It
inherits all of RVV and adds only what Kodiak has:

| what | where |
|---|---|
| the delta to the rvv contract (OPU compute unit, RTL config, measurement) | `contracts/overlay.yaml` |
| the effective contract Merlin reads (generated, do not edit) | `contracts/target_contract.yaml`, `contracts/composed.lock.yaml` |
| where the OPU's facts are read from (Chisel sources, config files, kernel roles) | `contracts/matrix_units.yaml` |
| the OPU C shim: operand packing, partial-N padding, OpenMP tile split, scalar twin | `opu_shim.py` |
| the whole-model hook: which contractions go to the OPU, and building the object | `matrix_lowering.py` |
| the boards: Kodiak silicon, the 2-core OPU elaboration, FireSim 1-core | `boards.yaml` |
| how to run on the chip (shared rig: check it is idle first) | [`RUNNING.md`](RUNNING.md) |
| how to build the Kodiak+OPU FireSim bitstream (config patches, FPGA clock-gate model) | [`firesim/`](firesim/README.md) |

```
rvv-mlir   (RVV contract, schedules)
 └── kodiak-mlir   (+ OPU unit, + Kodiak boards)    <- this repo
```

`provider.yaml` pins the exact rvv-mlir commit this layer was composed against. CI runs
`merlin-target-compose --check .`, so a change to the base shows up as a stale contract here and is
never inherited silently.

## Compile for Kodiak

Use a Merlin checkout (it vendors this package at `examples/kodiak/support` and selects it by default):

```sh
git clone https://github.com/ucb-bar/merlin.git && cd merlin/examples/kodiak
./run.sh setup        # Merlin venv + model capture env + .env; prints the commands for anything missing
                      # (LLVM/MLIR 23, chipyard with the OPU, Zephyr workspace + SDK)
```

**Test one operator** -- the quickest loop for an OPU kernel:

```sh
./run.sh op gemm                  # 64x64x64: RVV vs RVV+OPU, both gated on spike
./run.sh op gemm 96 80 200        # M N K (partial tiles, odd K)
./run.sh op conv2d 16 64 16 3     # CIN COUT HW K
./run.sh op path/to/loader.py     # your own one-op model
```

**A whole model** -- anything model2MLIR can capture:

```sh
./run.sh capture my_model --loader path/to/my_model/loader.py   # get_model_and_inputs()
./run.sh build --opu --models my_model                          # RVV and RVV+OPU, 1 and 2 harts
./run.sh package --opu --models my_model                        # zip for the chip (see RUNNING.md)
```

One model at one hart count, without packaging:

```sh
merlin-compile --workload <model> --dtype int8 --target rvv --board chipyard_kodiak --harts 2 \
  --matrix-unit saturn_opu --matrix-config KodiakOPUConfig --matrix-support-target kodiak \
  --run spike --verify
```

No simulator here executes the OPU, so the run is the image's **stand-in twin**: the same lowering
(checked by digest) with the OPU tile replaced by a scalar loop. That checks the routing, the K-major
pack, the call ABI and the requant epilogue; the OPU datapath itself is certified on its RTL
(`merlin.kernels.opu_cert`). The shipped ELF is audited to contain the OPU's instructions, and the twin
to contain none.

Every build writes `routing_report.json`, which lists which contractions went to the OPU and why the
others did not (too small for a 64×64 tile, transposed operand, dtype...). Read it before expecting a
speedup: a model whose matmuls are smaller than one tile runs entirely on RVV.

## Add a kernel or an instruction

* **A different OPU microkernel** (blocking, a fused epilogue): the C is emitted by `opu_shim.py` from
  Merlin's `merlin.kernels.opu_kernel.emit_microkernel`. Its instruction words are derived from the RTL,
  never written down. `merlin.kernels.opu_cert` certifies a kernel against the scalar reference
  corpus.
* **A new matrix instruction or unit**: add a `units:` block to `contracts/matrix_units.yaml` naming
  the Chisel sources, the declarations the derivation reads and the kernel roles; pin the RTL revision
  in Merlin's `hardware_pins.yaml`; give `matrix_lowering.py` a `build_object` for it.
* **Which contractions go to the unit**: `matrix_lowering.selector` (default: both `M` and `N` at least
  one tile edge). Profitability belongs in Merlin's cost model, not in a per-model rule.

The newer VME/Xsfmm encodings (saturn-vectors `vme` branch) belong to Iris, in `iris-mlir`.

## Status

* RVV images: h1 and h2 pass on the chip (2026-08). h3 (scalar third hart) is unexplained on silicon.
* OPU, h1 on the chip (2026-08-19): spectformer int8 bit-exact, 1.055× whole-model over RVV on the rig
  (OPU region 1.71×); FireSim 1-core 1.58×. Partial-N tiles need padding (on by default).
* OPU, h2: built; no chip result yet.
* Merlin flow (2026-11-04, spike stand-in twin, W8A8 golden): single GEMMs (64^3, 96x80x200, 64x200x72,
  128x128x64 at 2 harts), a conv2d and a 2-layer encoder all route to the OPU and pass.

## Checks

```sh
python -m pytest tests     # with Merlin installed; the rvv base must be selected or vendored
```
