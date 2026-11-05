# Iris: RVV today, the VME matrix unit next

Iris is chipyard `iris-cy`'s `IrisConfig`: two Shuttle tiles with Saturn vector units (VLEN 256 / DLEN 128)
carrying the **VME/Xsfmm** matrix unit, and one Rocket tile with a Saturn DMA vector unit (VLEN 256).
Its support is the **Iris chip package**, vendored at [`support/`](support/) from `ucb-bar/iris-mlir`. It is
a layer on the RVV base: see [`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md).

```bash
./run.sh setup                    # same one-step setup as the Kodiak example
./run.sh op gemm                  # one GEMM, RVV, gated on spike at the board's VLEN
./run.sh op gemm 128 128 64 --harts 2
./run.sh capture my_model --loader path/to/loader.py
```

Board `iris_rtl` (in `support/boards.yaml`) describes the Verilator elaboration: three vector harts,
serial-TL DRAM at `0x80000000`. No Iris silicon or bitstream facts are described yet.

**The VME unit is not routed yet.** Its instructions (`sf.mm.s.s`, `sf.vtzero.t`, `sf.vtmv.v.t`,
`vsettnt/vsettm/vsettk`) are not Kodiak's BME OPU. Adding it is a `units:` block in the iris package
(encodings derived from saturn `vme`, pinned as `saturn_vme`, cross-checked against
`benchmarks/common/vme.h`), a `matrix_lowering.py`, and a kernel modeled on
`benchmarks/vme-fused-lm2-gemm-transpose` -- after which `op` gains the same RVV-vs-matrix legs Kodiak has.
saturn's `vme` benchmarks already run on a spike built with the `zvt*` matrix extensions, so the VME
images will be gated on the unit itself rather than on a scalar twin.
