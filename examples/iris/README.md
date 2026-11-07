# Iris: RVV on three harts, int8 contractions on the VME matrix unit

Iris is chipyard `iris-cy`'s `IrisConfig`: two Shuttle tiles with Saturn vector units (VLEN 256 / DLEN 128)
carrying the **VME/Xsfmm** matrix unit (harts 0–1), and one Rocket tile with a Saturn DMA vector unit
(VLEN 256, no matrix unit; hart 2). Its support is the **Iris chip package**,
[`ucb-bar/iris-mlir`](https://github.com/ucb-bar/iris-mlir), a layer on the single RVV base (`ucb-bar/rvv-mlir`
at Merlin's base pin): see [`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md). Merlin does
not copy it: the RVV pin supplies the capability contract and executable RVV lowering. Merlin supplies
scalar lowering and shared LLVM finalization. Iris pins
one chip-package commit (`build_tools/upstreams/target_support.json`, `pinned_support`), and
`./run.sh setup` fetches exactly that commit into `out/build/generated/pinned/iris@<sha12>/`
(`merlin-target-fetch iris --root` prints it; below, `<iris-mlir>`). Not fetched, every stage but `setup`
stops and names `setup`. Changing the package is a commit in iris-mlir and one sha bumped here
(`python build_tools/scripts/vendor_support.py pin iris <sha>`); to try an unpublished change, select your
clone with `MERLIN_TARGET_PATH=/path/to/iris-mlir` (cloned with `--recurse-submodules`, it carries the same
pinned rvv base at `base/rvv-mlir`).

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

For a compiler-routed convolution that selects a handwritten VME kernel, run
[`examples/conv2d_kernel_demo.py`](https://github.com/ucb-bar/iris-mlir/blob/46524a94372ac4d63bb78f3586170b0901db4f8e/examples/conv2d_kernel_demo.py)
from the pinned Iris package. It checks routing, the selected kernel, and the shipped image on pinned
Zvt Spike. The matching [portable evidence](../kodiak/evidence/compiler_selected_conv_kernel_examples.json)
records artifact hashes and qualification limits.

### First hour: checklist

Times measured on a 48-core host (2026-11-05), after the Kodiak example had set up the same clone.

1. **Have the big three** the Kodiak example needs (LLVM/MLIR 23, chipyard with riscv-tools, the Zephyr
   SDK 0.17.0; setup finds them, never installs them), plus `uv`, `git`, `make`, `g++`, `dtc`.
2. **Iris's two checkouts.** Point `MERLIN_CHIPYARD_IRIS` and `MERLIN_SPIKE_ZVT_SRC` at ones you have,
   or let the first `./run.sh setup` print the clone commands (below) and run them: about 0.5 GB.
3. **`./run.sh setup --llvm <prefix> --chipyard <dir> --zephyr-sdk <dir>`**: about 2.5 min with the
   Zephyr workspace already there (the Zvt spike build is most of it), about 7.5 min on an empty clone.
   Exit 0 means nothing is left. A checkout on another revision is reported with the commands that
   move it, and never used.
4. **`./run.sh preflight`** (1 s): every stage `[ok ]`, `saturn_vme`, `rocket_chip_iris` and
   `spike_zvt` `ok`, and no `!` in the settings list (a `!` is a shell export overriding what setup
   recorded: `unset` it).
5. **`./run.sh op gemm 64 64 64`** (under 1 min): `rvv` and `matrix` `verified`, and "shipped ELF on a
   simulator that implements the unit: verified".

## Setup

`./run.sh setup` fetches iris-mlir at its pinned commit, does everything the Kodiak example's does (the
common part, with sizes and times, is in
[`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md#setting-up-a-machine)), and adds Iris's
two pinned sources:

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

Board `iris_rtl` (in `<iris-mlir>/boards.yaml`) describes the Verilator elaboration: three vector harts, of
which `matrix_hart_ids: [0, 1]` carry the VME unit; serial-TL DRAM at `0x80000000`. No Iris silicon or
bitstream facts are described yet.

## What `op` builds and checks

The RVV leg is the plain build, gated on spike at the board's VLEN. The VME leg routes every int8
contraction that fills at least one tile (TE = 32) to the unit and produces, through
`merlin.runtime.backends.matrix_image.build_matrix_pair`:

* **the shipped image**, whose contractions call the VME kernel (`<iris-mlir>/vme_kernel.py`, modeled on saturn
  `vme`'s `vme_gemm.h` / `vme-fused-lm2-gemm-transpose`: 2x2 super-tile, `sf.vtzero.t`, a `kmax`-row
  `sf.mm.s.s` loop, `sf.vtmv.v.t` readout). Audited: the VME instructions (decoded from the words by
  their derived MATCH/MASK, not by mnemonic) must be present;
* **the stand-in twin**: same lowering (checked by digest), scalar tile body, none of the VME
  instructions (audited), gated on stock spike: this checks routing, pack, descriptor ABI and epilogue;
* **the shipped image on spike with Zvt**, when `$MERLIN_SPIKE_ZVT` points at a riscv-isa-sim built with
  the Zvt extensions (pin `spike_zvt`; build recipe in `<iris-mlir>/README.md`). This executes the unit's
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
