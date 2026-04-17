# 2026-03-18: Chipyard Bare-Metal Integration

> **Repro pin:** merlin@[`e18fc562`](https://github.com/ucb-bar/merlin/commit/e18fc562c5c9a9601fc3e34a6d990a0427ddc255) · iree_bar@[`dd293bb513`](https://github.com/ucb-bar/iree_bar/commit/dd293bb513)
> **Status:** Active

## Context and Goal

Merlin compiles ML models for RISC-V accelerator targets (Saturn OPU, Gemmini MX,
NPU). Chipyard (separate repo at `CHIPYARD_ROOT`) generates the hardware those
models run on. There are two execution paths: FireSim FPGA simulation (Linux
workloads) and Chipyard VCS/Verilator bare-metal simulation.

The goal of this workstream is to bring up bare-metal IREE runtime execution on
Chipyard RTL simulation targets and make the workflow seamless for users who are
not Chipyard experts.

Prior art: the ucb-bar/iree `OPU_benchmark` branch
(`samples/simple_embedding_ukernel`) demonstrated bare-metal IREE on Saturn OPU
in an earlier fork. This workstream bridges that prior art into the Merlin repo
with proper patch management, configurable toolchains, and documented recipes.

## Implementation Changes

### Bare-Metal IREE Runtime Changes

Committed on the `ucb-bar/main` branch of our IREE fork (`github.com/ucb-bar/iree`).
The `third_party/iree_bar` submodule points at this branch — no separate patch
files to apply. Changes:

**`runtime/src/iree/base/alignment.h`** — Replaced `memcpy`-based unaligned
load/store with byte-by-byte implementations in the
`IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_16/32/64` code paths. Newlib's `memcpy`
on bare-metal RISC-V can emit unaligned word loads that trap.

**`runtime/src/iree/vm/bytecode/dispatch.c`** — Wrapped `IREE_RETURN_IF_ERROR`
macro calls in `#ifdef IREE_PLATFORM_GENERIC` guards with explicit status checks.
The macro expansion interacts badly with bare-metal newlib headers. Also wrapped
`memcpy` register marshaling in `#ifdef IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED`.

**`runtime/src/iree/vm/bytecode/dispatch_util.h`** — Added brace scoping to
`IREE_VM_ISA_DISPATCH_OP` macro bodies for both computed-goto and switch dispatch
modes.

**`runtime/src/iree/vm/bytecode/module.c`** — Same `IREE_RETURN_IF_ERROR` to
explicit status check pattern under `IREE_PLATFORM_GENERIC`.

**`runtime/src/iree/vm/bytecode/verifier.c`** — Wrapped `IREE_VM_VERIFY_PC_RANGE`
macro in `do { ... } while(0)` for proper hygiene. Extracted opcode into named
variable before switch.

**`runtime/src/iree/vm/invocation.c`** — Same `IREE_RETURN_IF_ERROR` and
`IREE_RETURN_AND_END_ZONE_IF_ERROR` to explicit status check pattern.

**`runtime/src/iree/vm/ref.c`** — Added byte-by-byte struct copy for
`iree_vm_ref_assign` under `IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED`, since direct
struct assignment can generate unaligned accesses.

### FireSim/Bare-Metal Toolchain

**`build_tools/firesim/riscv_firesim.toolchain.cmake`** — Combined Clang
(compile) + GCC (link via specs) toolchain for bare-metal RISC-V. Key features:

- Compiles with `--target=riscv64-unknown-elf`, `-march=rv64imafdc`,
  `-mstrict-align`
- Links with GCC via `htif-nano.spec` and `htif.ld`
- Key defines: `-DIREE_PLATFORM_GENERIC=1`,
  `-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1`,
  `-DIREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED`
- **Fixed:** Removed hardcoded chipyard sysroot path. Now reads `CHIPYARD_ROOT`
  or `RISCV_NEWLIB_SYSROOT` env vars.
- **Fixed:** `SCRIPTS_DIR` now uses `CMAKE_CURRENT_LIST_DIR` instead of
  hardcoded path.

**`build_tools/firesim/htif.ld`** — Linker script: base address `0x80000000`,
1GB heap, 128MB stack, HTIF section for host-target communication.

### Chipyard Bare-Metal Execution

Run bare-metal ELFs on Chipyard VCS simulation:

```bash
cd $CHIPYARD_ROOT/sims/vcs
make CONFIG=RadianceGemminiOnlyConfig BINARY=path/to/elf LOADMEM=1 run-binary
```

`LOADMEM=1` bypasses DRAM initialization, loading the ELF directly into
simulated memory.

### Gemmini MX Integration

- Chipyard `graphics` branch contains the Radiance + Gemmini MX configuration
- Config class: `RadianceGemminiOnlyConfig` with MX Gemmini (dim=16,
  accSize=32KB, tile=8x8x8)
- Reference bare-metal kernel: `gemmini-rocc-tests` repo,
  `bareMetalC/matmul_ws_mx_generic.c`

### `merlin chipyard` Tool

New `tools/chipyard.py` registered as `merlin chipyard` subcommand. Reads
recipe YAMLs from `build_tools/hardware/` and automates all Chipyard
interactions so users never manually edit Chipyard config files.

**Bare-metal flow:**

```bash
merlin chipyard set-path /path/to/chipyard
merlin chipyard validate gemmini_mx
merlin chipyard build-sim gemmini_mx
merlin chipyard run gemmini_mx path/to/elf
```

**FireSim flow:**

```bash
merlin chipyard build-firemarshal
merlin chipyard configure-firesim saturn_opu_u250
merlin chipyard build-bitstream saturn_opu_u250
merlin chipyard register-hwdb saturn_opu_u250
merlin chipyard stage-workload saturn_opu_u250
merlin chipyard status saturn_opu_u250
```

The FireSim commands automatically write:

- `config_build.yaml` — selects the build recipe
- `config_runtime.yaml` — selects hardware config and workload
- `config_hwdb.yaml` — registers built bitstream tarballs
- `workloads/<name>.json` + `workloads/<name>/overlay/` — workload definition
  and Merlin binary overlay

Shell scripts in `build_tools/hardware/scripts/` handle the Chipyard-side
operations (`configure_firesim.sh`, `register_hwdb.sh`,
`stage_firesim_workload.sh`, `build_firemarshal_base.sh`).

### Hardware Manifests

New `build_tools/hardware/*.yaml` manifests pin exact Chipyard SHAs per recipe
and contain enough data for the tool to generate all FireSim configs:

- `gemmini_mx.yaml` — mode: bare-metal, Chipyard graphics branch,
  RadianceGemminiOnlyConfig
- `saturn_opu_u250.yaml` — mode: firesim, includes full build recipe fields,
  runtime config, and workload definition
- `spacemit_x60.yaml` — mode: board, no Chipyard

### Hardware Backends Documentation

New `docs/hardware_backends/` section:

- Overview (including full `merlin chipyard` subcommand reference)
- Compatibility matrix, Chipyard concepts for Merlin users
- Recipes: Saturn OPU on U250, Gemmini MX bare-metal, SpacemiT X60

## What Worked

- The conditional compilation approach (`#ifdef IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED`
  and `#ifdef IREE_PLATFORM_GENERIC`) cleanly separates bare-metal workarounds
  from normal builds without penalizing Linux targets
- The Clang-compile + GCC-link toolchain split works reliably for bare-metal
  newlib targets
- Chipyard's `LOADMEM=1` flag makes VCS simulation fast (no DRAM initialization
  sequence)
- The existing patch infrastructure (`series.iree`, `apply_all.sh`) handles the
  new patch cleanly alongside the existing RISC-V/stream patch
- Env var cascade for `CHIPYARD_ROOT` makes the toolchain portable across
  different developer setups

## What Did Not Work

- Initial approach of unconditionally replacing `memcpy` with byte-by-byte copies
  penalized all targets, including Linux targets where `memcpy` is fast and
  aligned
- Hardcoded chipyard sysroot path in the toolchain cmake broke portability and
  required each developer to edit the cmake file
- `IREE_RETURN_IF_ERROR` macro issues are subtle — they only manifest on
  bare-metal newlib where certain header definitions differ from glibc
- The IREE codebase has evolved significantly since the ucb-bar/iree
  `OPU_benchmark` fork; the original diffs could not be applied as-is and
  required adaptation to the current `iree_bar` tree

## Debugging Notes

- Bare-metal traps from unaligned access show as `Store/AMO access fault`
  (cause 7) or `Load access fault` (cause 5) in spike/VCS
- To reproduce without RTL: `spike --isa=rv64imafdc <elf>`
- To trace `IREE_RETURN_IF_ERROR` macro expansion issues:
  `clang -E -DIREE_PLATFORM_GENERIC=1 dispatch.c` to see preprocessor output
- The `-mstrict-align` flag in the toolchain is critical — without it, Clang may
  generate unaligned accesses even for aligned data
- When debugging VCS simulation hangs, check `tohost` memory value — a nonzero
  value means the program terminated (value 1 = success)

## Test Coverage and Commands

```bash
# --- One-time setup ---
git submodule update --init third_party/iree_bar
conda run -n merlin-dev uv run tools/merlin.py chipyard set-path /path/to/chipyard
conda run -n merlin-dev uv run tools/merlin.py patches verify

# --- Bare-metal (Gemmini MX) ---
conda run -n merlin-dev uv run tools/merlin.py chipyard validate gemmini_mx
conda run -n merlin-dev uv run tools/merlin.py chipyard build-sim gemmini_mx
conda run -n merlin-dev uv run tools/build.py --profile firesim --config release
conda run -n merlin-dev uv run tools/compile.py \
  models/mlp/mlp.q.int8.mlir --target gemmini_mx --quantized
conda run -n merlin-dev uv run tools/merlin.py chipyard run gemmini_mx /path/to/elf

# --- FireSim (Saturn OPU) ---
conda run -n merlin-dev uv run tools/merlin.py chipyard validate saturn_opu_u250
conda run -n merlin-dev uv run tools/merlin.py chipyard build-firemarshal
conda run -n merlin-dev uv run tools/merlin.py chipyard configure-firesim saturn_opu_u250
conda run -n merlin-dev uv run tools/merlin.py chipyard build-bitstream saturn_opu_u250
conda run -n merlin-dev uv run tools/merlin.py chipyard register-hwdb saturn_opu_u250
conda run -n merlin-dev uv run tools/compile.py \
  models/mlp/mlp.q.int8.mlir --target saturn_opu --hw OPU --quantized
conda run -n merlin-dev uv run tools/build.py --profile firesim --config release
conda run -n merlin-dev uv run tools/merlin.py chipyard stage-workload saturn_opu_u250
conda run -n merlin-dev uv run tools/merlin.py chipyard status saturn_opu_u250
cd $CHIPYARD_ROOT/sims/firesim/deploy && firesim infrasetup && firesim runworkload
```

## Follow-Up Tasks

- Upstream the `alignment.h` byte-by-byte fix to IREE (benefits all bare-metal
  RISC-V targets)
- Add CI bare-metal smoke test using spike simulator
- Integrate `gemmini-rocc-tests` build into the Merlin `firesim` profile for
  automated Gemmini MX validation
- Add support for Verilator simulation (VCS alternative for teams without VCS
  licenses)

---

*Dev-blog written by:* Agustin Coppari Hollmann

*Project Members:* Agustin Coppari Hollmann, Hansung Kim
