# Examples

Two end-to-end walkthroughs of the thing this repo is actually for: compiling a PyTorch model to int8
multicore-RVV binaries for a **RISC-V tapeout you cannot attach to**, proving as much as possible in
simulation, and packaging it so the one person with the board can run it and mail back a log that
explains itself.

| example | chip | what it exercises that the other does not |
|---|---|---|
| [`kodiak/`](kodiak/) | ucb-bar Kodiak tapeout, 3 harts (2 with a vector unit), 512 MB | A **host-assisted (HTIF)** console, heterogeneous harts, and scalar images for the core with no vector unit |
| [`gemmelos/`](gemmelos/) | Bearly ML 25, 2 harts | A chip with **no Zephyr port and no debug host** — the UART address, its baud divisor and the PLL sequence are all *derived from that chip's own SDK headers*, and there is a variant that raises the clock 10× |

They are deliberately the same five stages, so the difference between "a board Zephyr knows about" and
"a board nothing knows about" is visible as a diff rather than described.

## Run one

```bash
cd examples/kodiak
./run.sh preflight        # says which stages this machine can do, and what would fix the rest
./run.sh probe            # seconds: build the vector probe, self-check it on spike
./run.sh build            # minutes: build + gate one small model at 1 and 2 harts
./run.sh package          # assemble a delivery package and its zip
./run.sh grade <log>      # score a console log the board's owner sent back
```

`--dry-run` prints every command without running it, so the examples are readable as documentation on a
machine with no toolchain at all. `--full` swaps the cheap subset for the matrix that actually ships
(hours of functional simulation — the flag exists so that cost is always a choice).

## What each stage needs

Run `./run.sh preflight` for a verdict on *your* machine; it asks the library the same questions the real
build does rather than re-deriving them. In general:

| stage | needs | who can run it |
|---|---|---|
| `grade` | python + numpy | anyone |
| `probe` | spike + a riscv64 toolchain | anyone who can build the toolchain |
| `build`, `package` | + the Zephyr tree below, + (gemmelos) that chip's SDK | anyone with those checkouts |
| running on the board | the silicon | the two people who have one |

The last row is the honest limit, and it is why the packaging half of this repo exists: everything the
board owner needs — both builds of every image, the references, a numpy-only grader, the expected console
— goes in one zip, because a round trip costs days.

## Inputs, and how to get each one

Nothing here is a binary blob you have to be given. Everything is either in this repo, clonable, or
regenerable with a documented command.

| input | where it comes from |
|---|---|
| **The tuned RVV schedule** — the compiler package these binaries are built with | Tracked in this repo at `out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8/` (3 files). Also published standalone: `git clone -b stable/impr_tuned_wholemodel_vf_int8 https://github.com/ucb-bar/rvv-mlir.git`. The frozen before-picture is branch **`baseline-int8_w8a8`** on the same repo — diffing the two is the clearest statement of what the tuning actually changed. |
| **Zephyr** | `git clone -b dev https://github.com/ucb-bar/zephyr-chipyard-sw.git`, then point `ZEPHYR_BASE` at `zephyr_ws/zephyr`. The commit matters and is checkable from any returned log: the boot banner prints it (`*** Booting Zephyr OS build 852bb170cc56 ***`). |
| **Model capture bundles** (`out/artifacts/recaptures/<model>_int8_full/`) | Not shipped — they are large and regenerable. Generate with [model2MLIR](https://github.com/ucb-bar/model2MLIR): `python $MODEL2MLIR_DIR/workloads/capture.py <model> --formats int8`. See [`docs/guides/model2mlir.md`](../docs/guides/model2mlir.md); quantization happens in that repo, not this one. |
| **spike + riscv64 toolchain** | Any build works; point `MERLIN_SPIKE` at the binary, or `MERLIN_CHIPYARD` at a chipyard checkout whose `.conda-env` provides riscv-tools. |
| **The gemmelos SDK** | A third-party bring-up repo, not ours to redistribute. It supplies only *facts* — a UART base address, two clock rates, a PLL register map. The `facts` stage prints exactly what is read out of it, so if you cannot get the checkout you can see precisely what would have to be supplied another way. The Kodiak example needs none of this. |
| **The boards** | Two tapeouts. You will not have one, which is the premise. |

## Reading a log that came back

Both examples end at the same place, because in practice that is where the work is. `gemmelos/` ships a
**real returned failure** (`returned/whisper_h1_debug_500mhz_fault.txt`) so the diagnosis flow is
reproducible with nothing but python: four lines of console localise the bug to one wrong field in a board
descriptor. `gemmelos/README.md` walks it.

## See also

- [`docs/guides/getting_started.md`](../docs/guides/getting_started.md) — the repo itself
- [`docs/guides/adding_a_target.md`](../docs/guides/adding_a_target.md) — a third board
- [`docs/guides/model2mlir.md`](../docs/guides/model2mlir.md) — where capture bundles come from
- [`docs/guides/reproducibility.md`](../docs/guides/reproducibility.md) — what each result rests on
