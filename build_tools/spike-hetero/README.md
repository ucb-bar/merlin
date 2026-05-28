# `spike-hetero` — Spike with both Gemmini and Saturn-OPU extensions

A single `spike` invocation that decodes **both** Gemmini RoCC ops
(custom-3, 0x7B) and Saturn OPU ops (OP-V, 0x57). Lets us run the same
`merlin_hetero_runner` Zephyr ELFs that target the FireSim Shuttle
`DualSaturnOPUGemmini` bitstream without spinning up FireSim.

Wall-clock for `dronet × Gemmini+OPU` on Spike is ~50× faster than
FireSim, which makes it the right tool for inner-loop correctness
debugging — Gemmini ISA bugs, OPU encoding bugs, runtime-side glue
issues. Use FireSim for the final timing-accurate validation.

## How it works

* `saturn_opu.cc` is vendored verbatim from
  [CobbledSteel/riscv-isa-sim @ saturn-opu-extension](https://github.com/CobbledSteel/riscv-isa-sim/tree/saturn-opu-extension).
  It implements the four custom OP-V instructions
  (`vopacc`, `opmvinbcast`, `vmv.vr`, `vmv.rv`) as a runtime-loadable
  `extension_t` subclass.
* `libgemmini.so` ships with chipyard (built from
  `chipyard/generators/gemmini/software/libgemmini/`).
* This directory's `Makefile` builds `saturn_opu.cc` as a standalone
  `libsaturn_opu.so` against the chipyard-installed spike headers
  (`$RISCV/include/riscv`) plus the source-tree-only macro headers
  (`$SPIKE_SRC/riscv/{insn_macros.h,decode_macros.h}` and
  `$SPIKE_BUILD/config.h`).
* The `spike-hetero` wrapper script invokes
  `spike --extension=gemmini --extension=saturn_opu` so both .so files
  are loaded before the ELF is mapped.

The opcode spaces don't conflict: Gemmini lives in custom-3 (0x7B),
OPU lives in OP-V (0x57). If a hart accidentally hits the wrong
extension Spike just traps the illegal instruction — no silent
corruption.

## Build + install

```bash
export CHIPYARD_ROOT=/scratch2/agustin/chipyard
cd build_tools/spike-hetero
make            # produces libsaturn_opu.so in this directory
make install    # copies it to $RISCV/lib/ alongside libgemmini.so
```

The build needs chipyard's `riscv-isa-sim/build/config.h` to exist
(i.e., spike must have been built at least once via
`./scripts/build-toolchains.sh` in chipyard).

## Run

```bash
# Single-hart, Gemmini-only ELF
./spike-hetero <elf>

# Multi-hart hetero ELF (the FireSim Shuttle default — 2 harts)
SPIKE_HARTS=2 ./spike-hetero <elf>

# Wider VLEN (matching Saturn's 512-bit config)
SPIKE_ISA=rv64gcv_zicntr ./spike-hetero <elf>
```

The wrapper exits with spike's exit code, so it's CI-friendly.

## Smoke test

```bash
make test   # equivalent to: ./tests/smoke.sh
```

Runs three ELFs in increasing complexity:

| # | ELF | Coverage |
|---|---|---|
| 1 | `matmul_1x1x2048_os-baremetal` | Pure Gemmini, single hart, upstream rocc-tests |
| 2 | `bench_gemmini_spike_matmul` | Pure Gemmini, single hart, IREE bare-metal |
| 3 | `build_*/zephyr/zephyr.elf` | Full hetero, 2 harts, both extensions exercised |

ELF paths can be overridden via `GEM_ROCC_ELF`, `IREE_ELF`,
`MERLIN_HETERO_ELF` env vars.

## Why a runtime plugin instead of mirroring dima's customext-builtin path

Two valid integration paths exist:

* **Plugin** (this repo): build `libsaturn_opu.so` standalone against the
  installed chipyard spike, no rebuild of riscv-isa-sim required.
* **Built-in** (used by `/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard`):
  add `saturn_opu.cc` to `customext.mk.in`'s `customext_srcs`, rebuild
  riscv-isa-sim, so the extension factory ships statically in
  `libcustomext.so`.

Both end up at the same `REGISTER_EXTENSION(saturn_opu, ...)` factory
and the same `--extension=saturn_opu` CLI flag. The plugin model is
faster to bootstrap (seconds vs 30-min spike rebuild) and survives
chipyard env churn cleanly, so that's what's wired up here. The
`saturn_opu.cc` file is bit-identical to dima's working tree, so
behaviour is the same.

## Limitations / known gaps

* `saturn_opu.cc` is **functional-only** — it models the architectural
  semantics of the four OPU instructions, not the microarchitectural
  pipeline. Cycle counts emitted by Spike are not meaningful for OPU
  ops (Gemmini RoCC cycles aren't faithful either; both extensions
  execute atomically in Spike's model). Use FireSim for timing.
* The OPU matrix dim follows `VLEN/8`. With Spike's default
  `--isa=rv64gcv_zicntr` this gives dim=16 (vlen=128). Saturn's HW
  config uses vlen=512 → dim=64; pass `--varch=vlen:512,elen:64`
  to match.
* Only the four documented OPU custom insns are modelled (matching the
  set used by `merlin_hetero_runner`). If the OPU plugin ever needs to
  cover new ops the file is short enough to edit directly.

## Provenance

* `saturn_opu.cc` — copied from
  `CobbledSteel/riscv-isa-sim@saturn-opu-extension:customext/saturn_opu.cc`
  (commit `7668165`, single-file branch ahead of upstream `riscv-isa-sim`).
  Bit-identical to the working copy at
  `/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/saturn_opu.cc`.
