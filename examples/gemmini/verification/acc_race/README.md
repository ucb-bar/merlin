# Accumulator load-order race reproducers

Gemmini orders a load behind earlier loads only until they have been *issued*, not until they have
*completed*. Two loads into the same accumulator rows, the first overwriting and the second
accumulating, can therefore land in the wrong order: when the accumulating load's data arrives
first, the overwriting load erases it and the readout is the first operand alone. The measurements,
the hardware revision they belong to and what they mean for verification are in
[the design note](../../../../docs/guides/12-experiments.md).

These programs make the race visible and classify every wrong element by which operand survived.
They are target-local probes, not compiler tests: they need a RISC-V cross compiler and a bare-metal
Gemmini harness tree, and their verdicts come from a simulator or the board.

| File | What it is |
| --- | --- |
| [acc_race_repro.c](acc_race_repro.c) | An overwrite-then-accumulate loop issued three ways (unfenced on two load units, fenced, both loads on one unit) and the vendor `tiled_resadd_auto` at several views of one tensor. Stimulus is generated on the core, so it is cache-resident. |
| [vend_repro.c](vend_repro.c) + [gen_vend_data.py](gen_vend_data.py) | One vendor residual add on operands that are initialised data in the ELF, so they arrive cold from DRAM, the situation of a capsule's inputs and of a layer's activations. |
| [build.sh](build.sh) | Builds either program against a harness tree and writes `build.json` naming every input by sha256. |
| [fence_vendor_resadd.py](fence_vendor_resadd.py) | Copies a harness tree and makes its residual add order its two loads with a fence, leaving scales, activation and readout scale unchanged. The control for "the vendor library's own residual add races". |
| [classify.py](classify.py) | Rolls the consoles' `RACE`/`VEND` lines up per test and reports whether every wrong element is the race's signature (`lhs_only`). |

## Build and run

```sh
export GEMMINI_HARNESS=/path/to/gemmini-rocc-tests   # include/, riscv-tests/benchmarks/common/
export RISCV_CC=/path/to/riscv64-unknown-elf-gcc

# The default build is what the FPGA runs. RTL simulators can afford the small one:
examples/gemmini/verification/acc_race/build.sh out/artifacts/probes/acc_race/small \
  -DRACE_ROWS=64 -DRACE_COLS=128 -DREPS=1 -DSKIP_VENDOR

# Cold-operand vendor residual add (98 rows x 512 columns):
python examples/gemmini/verification/acc_race/gen_vend_data.py out/artifacts/probes/acc_race/vend/vend_data.h 98
REPRO_SRC=examples/gemmini/verification/acc_race/vend_repro.c \
  examples/gemmini/verification/acc_race/build.sh out/artifacts/probes/acc_race/vend

# Run the ELF on a simulator or the board, keep the console, then:
python examples/gemmini/verification/acc_race/classify.py console.log
```

To build the fenced control, build the same program against
`fence_vendor_resadd.py <harness> <harness_fenced>` instead of the stock harness.

## Reading a result

- A clean in-order run proves nothing. Operands the core wrote, or that the host interface loaded,
  sit in the last-level cache, and no request for them reaches a reordering memory model. Load the
  program into the backing store, and judge each perturbed run against a console that already
  passed the golden, not against the in-order run's cycle count.
- FireSim is cycle-deterministic for a given ELF: the same binary fails the same elements on every
  run. What changes the outcome is timing context, such as a different repetition or different code
  around the kernel.
- A sweep of N seeds samples the schedule space; a clean sweep does not prove a program race-free.
