# Iris: RVV on three harts, int8 contractions on the VME matrix unit

Iris is chipyard `iris-cy`'s `IrisConfig`: two Shuttle tiles with Saturn vector units (VLEN 256 / DLEN 128)
carrying the **VME/Xsfmm** matrix unit (harts 0–1), and one Rocket tile with a Saturn DMA vector unit
(VLEN 256, no matrix unit; hart 2). Its support is the **Iris chip package**, vendored at
[`support/`](support/) from `ucb-bar/iris-mlir`. It is a layer on the RVV base: see
[`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md).

```bash
./run.sh setup --llvm <llvm23-install> --chipyard <chipyard> --zephyr-sdk <zephyr-sdk-0.17.0>
                                  # the Kodiak example's setup, plus Iris's two pinned checkouts (below)
./run.sh preflight
./run.sh op gemm                  # one 64x64x64 GEMM: RVV vs RVV+VME, gated on spike
./run.sh op gemm 96 80 200        # M N K: partial tiles in N, a K that is not a multiple of kmax
./run.sh op conv2d 16 64 16 3     # CIN COUT HW K (im2col + matmul: a 64x196x144 contraction)
./run.sh op gemm --harts 2        # the tile loop split over both VME harts
./run.sh op gemm --rvv-only       # the RVV leg alone
./run.sh capture my_model --loader path/to/loader.py
```

## Setup

`./run.sh setup` does everything the Kodiak example's does (the common part, with sizes and times, is
in [`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md#setting-up-a-machine)), and adds
Iris's two pinned sources:

| pin (`merlin/contract/hardware_pins.yaml`) | what | found at | size |
|---|---|---|---|
| `chipyard_iris` (with `saturn_vme`, `rocket_chip_iris` nested in it) | chipyard `iris-cy`: the VME encodings and geometry are derived from it | `$MERLIN_CHIPYARD_IRIS`, else `out/build/ext/chipyard_iris/` | ~0.5 GB (only the two submodules the route reads) |
| `spike_zvt` | riscv-isa-sim with Zvt: runs the shipped VME image | `$MERLIN_SPIKE_ZVT_SRC`, else `out/build/ext/spike_zvt/` | ~60 MB source |

Each is verified by content (commit, branch, required files, and the gitlinks chipyard records for
`generators/saturn` and `generators/rocket-chip`). If one is missing, setup prints the commands that fetch
it into `out/build/ext/`, generated from the pin; run them and run setup again. They are, today:

```bash
git clone https://github.com/ucb-bar/chipyard.git out/build/ext/chipyard_iris
git -C out/build/ext/chipyard_iris checkout -B iris-cy b4542facf6cc9457f6e358e8f80a8b3a0ff5df19
git -C out/build/ext/chipyard_iris submodule update --init generators/rocket-chip generators/saturn
git clone https://github.com/riscv-software-src/riscv-isa-sim.git out/build/ext/spike_zvt
git -C out/build/ext/spike_zvt checkout -B master f90d5af53f50b0feec7a7cbcb4411e1e0410c465
```

(setup prints them with absolute paths). A checkout you already have is used where it is: point
`MERLIN_CHIPYARD_IRIS` / `MERLIN_SPIKE_ZVT_SRC` at it. On the wrong revision it is reported with the
commands that would move it, and not used. From a verified `spike_zvt` source, setup builds spike out of
tree into `out/build/spike_zvt/` (about two minutes on 32 cores; needs `make`, `g++` and `dtc`) and
records `MERLIN_SPIKE_ZVT`, `MERLIN_SPIKE_ZVT_SRC` and `MERLIN_CHIPYARD_IRIS` in `.env`, which is all
`op` needs. The build is redone only when the source revision or the binary changes.

## Boards

Board `iris_rtl` (in `support/boards.yaml`) describes the Verilator elaboration: three vector harts, of
which `matrix_hart_ids: [0, 1]` carry the VME unit; serial-TL DRAM at `0x80000000`. No Iris silicon or
bitstream facts are described yet.

## What `op` builds and checks

The RVV leg is the plain build, gated on spike at the board's VLEN. The VME leg routes every int8
contraction that fills at least one tile (TE = 32) to the unit and produces, through
`merlin.runtime.backends.matrix_image.build_matrix_pair`:

* **the shipped image**, whose contractions call the VME kernel (`support/vme_kernel.py`, modeled on saturn
  `vme`'s `vme_gemm.h` / `vme-fused-lm2-gemm-transpose`: 2x2 super-tile, `sf.vtzero.t`, a `kmax`-row
  `sf.mm.s.s` loop, `sf.vtmv.v.t` readout). Audited: the VME instructions (decoded from the words by
  their derived MATCH/MASK, not by mnemonic) must be present;
* **the stand-in twin**: same lowering (checked by digest), scalar tile body, none of the VME
  instructions (audited), gated on stock spike: this checks routing, pack, descriptor ABI and epilogue;
* **the shipped image on spike with Zvt**, when `$MERLIN_SPIKE_ZVT` points at a riscv-isa-sim built with
  the Zvt extensions (pin `spike_zvt`; build recipe in `support/README.md`). This executes the unit's
  words under the Zvt draft's semantics; without it the leg reports `not_run` and why.

Every VME encoding, the tile edge, the operand alignment and the reduction depth per instruction are
derived from `$MERLIN_CHIPYARD_IRIS` (a chipyard `iris-cy` checkout with `generators/saturn` at pin
`saturn_vme` and `generators/rocket-chip` at pin `rocket_chip_iris`), cross-checked against saturn's
`benchmarks/common/vme.h` and, when present, the simulator's `riscv/encoding.h`. A disagreement refuses
to build. A matrix image over 3 harts is refused: hart 2 has no matrix unit.

## Measured (2026-11-04, merlin worktree of `cc/chip-packages`)

Pins verified: `chipyard_iris` b4542fa, `saturn_vme` 536774b, `rocket_chip_iris` fc745a8, `spike_zvt`
f90d5af. All legs graded against the W8A8 golden; spike counts are retired-instruction counts, not Iris
cycles.

| op | harts | RVV | twin | shipped on spike+Zvt | routed (M N K) | spike count RVV / shipped |
|---|---|---|---|---|---|---|
| gemm 64 64 64 | 1 | verified | verified | verified, cos 0.99999994 | 64 64 64 | 615 k / 500 k |
| gemm 64 64 64 | 2 | verified | verified | verified, cos 0.99999994 | 64 64 64 | 535 k / 335 k |
| gemm 96 80 200 | 1 | verified | verified | verified, cos 1.0 | 96 80 200 | 2.53 M / 1.86 M |
| gemm 96 80 200 | 2 | verified | verified | verified, cos 1.0 | 96 80 200 | 2.64 M / 1.24 M |
| conv2d 16 64 16 3 | 1 | verified | verified | verified, cos 1.0 | 64 196 144 | 8.19 M / 3.81 M |
| conv2d 16 64 16 3 | 2 | verified | verified | verified, cos 1.0 | 64 196 144 | 5.59 M / 3.64 M |
| encoder_block | 1 | verified | verified | verified, cos 0.99999988 | 12 contractions, widest 64 384 128 | 54.1 M / 38.0 M |
| encoder_block | 2 | verified | verified | verified, cos 0.99999988 | 12 contractions | 41.7 M / 28.5 M |
| gemm 64 64 64 | 3 | verified | refused: hart 2 has no matrix unit | – | – | – |

Each shipped image carries 71 VME instruction words (20 `sf.mm`, 5 `sf.vtzero.t`, 10 `sf.vtmv.v.t`, the
rest configuration); each twin carries none. The VME share of contraction MACs is 1.0 in every case: nothing
was left on the vector unit.

Not measured here: the RTL. The kernel has not run on Iris's Verilator elaboration or on silicon; the
spike verdict says the words and their use are right under the Zvt draft, which the RTL states it follows.
