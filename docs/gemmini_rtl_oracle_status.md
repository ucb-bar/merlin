# Gemmini oracle status (Step 0)

Proof that the Gemmini execution oracles exist and run a known-good binary. Re-run with
`python tools/probe_gemmini_oracle.py [--run spike] [--run verilator]`.

## Environment (probed, all present)

| Component | Path | Status |
|---|---|---|
| chipyard | `/scratch2/agustin/chipyard` | OK |
| spike | `…/.conda-env/riscv-tools/bin/spike` | OK |
| libgemmini.so | `…/.conda-env/riscv-tools/lib/libgemmini.so` | OK |
| Verilator sim | `…/sims/verilator/simulator-chipyard.harness-GemminiAndOPUShuttleConfig` | OK (prebuilt) |
| riscv gcc | `…/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc` | OK |
| gemmini-rocc-tests | `…/generators/gemmini/software/gemmini-rocc-tests` | OK |
| linker script | `…/gemmini-rocc-tests/riscv-tests/benchmarks/common/test.ld` | OK |

Env overrides: `MERLIN_CHIPYARD`, `MERLIN_GEMMINI_SPIKE`, `MERLIN_GEMMINI_VERILATOR`,
`MERLIN_RISCV_GCC`, `MERLIN_GEMMINI_HARNESS_DIR`.

## Oracle ladder — proven

- **L1 spike-gemmini (bootstrap, `derived_from_rtl: false`)** — PROVEN.
  `spike --extension=gemmini …/build/bareMetalC/matmul-baremetal` ran to completion
  (exit 0), printed its test output, reported `dim = 16`. `LD_LIBRARY_PATH` must include
  `…/riscv-tools/lib` for `libgemmini.so`.
- **L2 Verilator RTL (`GemminiAndOPUShuttleConfig`, `derived_from_rtl: true`)** — PROVEN.
  `simulator-chipyard.harness-GemminiAndOPUShuttleConfig …/build/bareMetalC/mvin_mvout_zeros-baremetal`
  reached `$finish` (exit 0) in **~91 s** wall. UART0 is stdout (printf reaches the console).

## Build + run recipe (for generated C)

**Compile** (bareMetalC, from `gemmini-rocc-tests/bareMetalC/Makefile`):
```
riscv64-unknown-elf-gcc -DBAREMETAL=1 -mcmodel=medany -march=rv64gc -std=gnu99 -O2 \
  -static -nostdlib -nostartfiles -fno-common -fno-builtin-printf \
  -T <rocc_tests>/riscv-tests/benchmarks/common/test.ld \
  -I<rocc_tests> -I<rocc_tests>/riscv-tests -I<rocc_tests>/riscv-tests/env \
  -I<rocc_tests>/riscv-tests/benchmarks/common \
  <generated.c> <rocc_tests>/riscv-tests/benchmarks/common/*.c \
  <rocc_tests>/riscv-tests/benchmarks/common/*.S -lm -lgcc -o <elf>
```
**Run — spike (L1):** `LD_LIBRARY_PATH=<riscv-tools/lib> spike --extension=gemmini <elf>`
**Run — Verilator (L2):** `simulator-chipyard.harness-GemminiAndOPUShuttleConfig <elf>`

## Gemmini ISA facts (for codegen)

- `DIM = 16`; `elem_t = int8`, `acc_t = int32`.
- Scratchpad address: bit31 = 0. Accumulator address: bit31 = 1 (`1u << (ADDR_LEN-1)` = `0x80000000`);
  bit30 = accumulate-vs-overwrite.
- Sequence (weight-stationary single tile): `gemmini_config_ld(stride)`,
  `gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0)`, `gemmini_mvin(A, …)`,
  `gemmini_mvin(B, …)`, `gemmini_preload(B_spad, acc_addr)`,
  `gemmini_compute_preloaded(A_spad, GARBAGE_ADDR)`, `gemmini_mvout(C, acc_addr)`, `gemmini_fence()`.
- **Full i32 mvout** (C0 needs this): `gemmini_extended_config_st(DIM*sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY)`
  and an `acc_t` output buffer (default mvout emits `elem_t`/i8).
- Cycles: `read_cycles()` = `rdcycle` (`gemmini_testutils.h`). Bracket the Gemmini region only.

## Constraints / gotchas

- **M, K, N must be multiples of DIM (16).** C0 is a single **16×16×16** tile (edge shapes are
  the held-out C4 rung). Row-major; `gemmini_fence()` before reading results.
- Verilator wall time ~90 s for a tiny test → certification runs are out of the fast unit-test
  path (toolchain-gated / manual), not in CI.
- **Spike is bootstrap only.** It is a hand-written functional model (`libgemmini.so`),
  **not derived from RTL** — only Verilator/FireSim results may be marked `rtl_certified`.

## Status: C0 is RTL-certified

Both oracles run, and the C0 path is complete: the generated Gemmini kernel runs on the
Verilator RTL sim with **three-way bit-exact equality** (RTL == reference == simulator),
**cycles = 241** (gemmini_region), in ~3 min wall. See `results/gemmini/certification_c0.yaml`
and `docs/gemmini_target_prototype.md`. The Verilator memory model is not characterized
(recorded as `unknown`; FireSim/FASED is where memory timing becomes realistic).

Build gotcha (cost a debugging cycle): the bareMetalC compile flags **and include order** must
match `gemmini-rocc-tests/bareMetalC/Makefile` exactly — a wrong `-I` order shadows the
`riscv-tests/env` syscall headers and corrupts the tohost protocol ("bad syscall" on spike).
`runtime/backends/gemmini.py::compile_command_buffer` mirrors it.
