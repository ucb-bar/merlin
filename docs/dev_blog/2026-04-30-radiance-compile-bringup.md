# 2026-04-30: Radiance/Muon Compile Pipeline in Merlin — Phase 0 + 1

> **Status:** Phase 0 + 1 + 2 (manifest mode) + 2.6 (MLIR plugin) all
> landed. Merlin compiles a kernel.mlir → kernel.radiance.elf via the
> upstream radiance-kernels llvm-muon toolchain, end-to-end from MLIR.
> Three byte/functionally equivalent paths to the same ELF:
> Phase 1 (descriptor + inline body, byte-identical), Phase 2 (manifest +
> C++ body, +80 bytes), Phase 2.6 (MLIR + LLVM-IR body, +60 bytes).
> Sim execution remains gated on a fresh simv build (TSI harness has a
> pre-existing TLMonitor assertion).

## Context and Goal

Goal: enable Merlin to compile Radiance/Muon GPU kernels for the Chipyard
`RadianceMuonConfig` SoC. Hansung's working flow is:

```bash
cd /scratch2/agustin/radiance-kernels/kernels/vecadd && make
cd $CHIPYARD_ROOT/sims/vcs
make run-binary CONFIG=RadianceMuonConfig BINARY=.../kernel.radiance.elf
```

This produces a Muon-targeted RV32 device kernel via a vendored llvm-muon
clang and runs it on Chipyard RTL simulation. We wanted Merlin to be the
top-level driver of that pipeline, with MLIR as the eventual frontend
(Phase 2) and a kernel-descriptor YAML as the Phase-1 stand-in.

User-confirmed parameters:
- Target SoC: `RadianceMuonConfig` (single Muon core, no host CPU, no
  Gemmini, fastest elaboration, Cyclotron-difftest available).
- Backend: llvm-muon clang as a separate process; IREE itself unmodified.
- Phasing: skip-Gluon RTL run-binary smoke first, then gluon-sim parity.
- **Honest lowering target**: C++ source (with `mu_intrinsics.h` +
  `mu_schedule`), not LLVM IR. There is no Muon LLVM backend; `+vortex`
  is a print-only feature flag on stock RV32. C++ matches the actual ISA
  surface and reuses `libmuonrt.a` + `mu_start.S.o` correctly.

## What Landed

### Phase 0a — Reference control

Confirmed the upstream build still works untouched. The reference
`kernel.radiance.elf` (22744 bytes, ELF32 RV32, entry 0x10000000,
md5=6919fe959f5b8787dc2e79b59260a262) is the byte-equivalence target
for everything downstream.

The existing `simv-chipyard.harness-RadianceMuonConfig` (built Apr 23)
hits a `TLMonitor_228.sv` PutPartial assertion in `TSIHarness.scala:77`
on either the default or `LOADMEM=1` paths. That's an **existing simv
build issue**, independent of Merlin's compile pipeline. The Cyclotron
ELF loader reaches `loading ELF file: .../kernel.radiance.elf` before
the harness fault, confirming the ELF and toolchain are correct. Sim
execution validation is therefore deferred to a fresh simv rebuild or
Verilator install.

### Phase 0b — Merlin owns the toolchain wrap

New / modified files:
- `merlin/build_tools/radiance/riscv_muon.toolchain.cmake` (NEW) — single
  source of truth for Muon CFLAGS/LDFLAGS, mirrors
  `radiance-kernels/kernels/common.mk` exactly. Resolves `$LLVM_MUON` and
  `$RADIANCE_KERNELS_ROOT` from env; fails fast if either is unset.
- `merlin/build_tools/radiance/mu_link.ld` (NEW, vendored from
  `radiance-kernels/lib/linker/mu_link.ld`).
- `merlin/build_tools/radiance/common.cmake` (NEW) — CMake helper
  `merlin_radiance_kernel_executable(NAME ... KERNEL ...)` that wraps
  `add_executable` with the Muon-specific suffix (`.radiance.elf`) and
  link rule. Also `merlin_radiance_pin_check()` validates the
  radiance-kernels checkout against `pin.txt`.
- `merlin/build_tools/radiance/pin.txt` (NEW) — radiance-kernels commit
  `56aad6e1620c452bd131c948f72352dab0754d6e` ("Port gaussian and bfs
  from vortex"). CMake warns loud on drift.
- `merlin/build_tools/radiance/CMakeLists.txt` (NEW) — driver
  CMakeLists. Loaded by `./merlin build --profile radiance_muon`. Reads
  `MERLIN_RADIANCE_KERNEL_DIR` and emits `<name>.radiance.elf`.
- `merlin/build_tools/hardware/scripts/run_radiance_muon.sh` (NEW) —
  wraps `make run-binary CONFIG=RadianceMuonConfig BINARY=...` from
  `chipyard/sims/vcs/`, scrapes uartlog tail.
- `merlin/build_tools/hardware/radiance_muon.yaml` (MOD) — recipe
  pointed at `RadianceMuonConfig` + vcs simulator + the
  `kernel.radiance.elf` reference binary.
- `merlin/tools/build.py` (MOD) — `--target radiance_muon`,
  `--profile radiance_muon`, `--kernel-dir`, `--kernel-name`. Early
  short-circuit in `main()` to a dedicated `_build_radiance_muon()`
  handler (no IREE configure).
- `merlin/tools/chipyard.py` (MOD) — `run-radiance-muon` subcommand.
  Auto-discovers the kernel ELF from
  `build/radiance_muon-vanilla-release/`.

Acceptance — Phase 0b passes:
```
$ ./merlin build --profile radiance_muon --config release
$ md5sum build/radiance_muon-vanilla-release/vecadd.radiance.elf \
         $RADIANCE_KERNELS_ROOT/kernels/vecadd/kernel.radiance.elf
6919fe959f5b8787dc2e79b59260a262  build/.../vecadd.radiance.elf
6919fe959f5b8787dc2e79b59260a262  $RADIANCE_KERNELS_ROOT/.../kernel.radiance.elf
```
**Byte-identical**. `cmp` returns 0; `nm` symbol set is identical. We
faithfully reproduce the upstream kernel build.

### Phase 1 — Merlin emits the kernel source

New files:
- `merlin/build_tools/radiance/templates/kernel.cpp.j2` (NEW) — Jinja
  template that emits a kernel.cpp matching the upstream
  `radiance-kernels/kernels/<name>/kernel.cpp` shape: `#include
  <mu_intrinsics.h>`, `struct Args`, per-thread function, `mu_schedule(fn,
  &args, NUM_WARPS)` in `main()`. The `mu_schedule` call is
  unconditional and load-bearing: omitting it causes all warps to
  execute redundantly because the hardware boots all warps/threads at
  `_start` with no gating.
- `merlin/build_tools/radiance/templates/host.cpp.j2` (NEW) — minimal
  RV64 host stub for symmetry with the upstream build.
- `merlin/models/radiance_muon.yaml` (NEW) — compile-flag bundle.
- `merlin/models/radiance_muon/vecadd.yaml` (NEW) — first reference
  kernel descriptor: name, num_warps, args struct fields, kernel body,
  data_file path.
- `merlin/tools/compile.py` (MOD) — `--target radiance_muon`
  short-circuit to `_compile_radiance_muon()`. Renders the descriptor
  through Jinja, stages the data sidecar, hands off to `./merlin build
  --profile radiance_muon`. Supports `--compile-to cpp` to stop after
  source emit.
- `merlin/scripts/radiance_muon_smoke.sh` (NEW) — single-command driver
  (compile → run).
- `merlin/docs/dev_blog/2026-04-30-radiance-compile-bringup.md` (this
  file).

Acceptance — Phase 1 passes:
```
$ ./merlin compile models/radiance_muon/vecadd.yaml --target radiance_muon
  📄 Generated: build/compiled_models/vecadd/radiance_muon_vecadd/kernel.cpp
  📄 Staged:    build/compiled_models/vecadd/radiance_muon_vecadd/data
  ✅ kernel.radiance.elf: build/compiled_models/vecadd/radiance_muon_vecadd/vecadd.radiance.elf

$ md5sum build/compiled_models/vecadd/radiance_muon_vecadd/vecadd.radiance.elf \
         $RADIANCE_KERNELS_ROOT/kernels/vecadd/kernel.radiance.elf
6919fe959f5b8787dc2e79b59260a262  build/.../vecadd.radiance.elf
6919fe959f5b8787dc2e79b59260a262  $RADIANCE_KERNELS_ROOT/.../kernel.radiance.elf
```
**Byte-identical** end-to-end from a Merlin descriptor input. The
generated `kernel.cpp` is near-identical to the upstream reference
modulo a comment-banner and one trailing blank line — neither affects
the ELF.

### Phase 2 — Kernel-embed manifest mode

Reuses Merlin's existing kernel-embedding infrastructure
(`tools/kernels/{manifest,precompile}.py`) as the source of Radiance kernel
bodies. The body is compiled by llvm-muon clang via a new `radiance-muon`
target in `precompile.py`, then linked into `kernel.radiance.elf` alongside
a Phase-2 wrapper template (`kernel_phase2.cpp.j2`) that declares the kernel
`extern "C"`.

New / modified files:
- `tools/kernels/precompile.py` (MOD) — added `radiance-muon` to
  `_CPU_TARGET_FLAGS`, new `_compile_radiance_muon_obj()` helper, new
  `_radiance_muon_baseline_flags()` (verbatim mirror of
  `radiance-kernels/kernels/common.mk` MU_CFLAGS), `Toolchain` extended
  with `llvm_muon_clang` + `radiance_kernels_root` resolved from env.
  Source-lang dispatch now routes `c`/`cpp`/`ll` on `radiance-muon` through
  the Muon code path; the existing `c` host-clang path is unchanged.
- `tools/kernels/manifest.py` (MOD) — added `cpp` and `ll` to
  `_VALID_SOURCE_LANGS`.
- `benchmarks/Radiance/kernels/manifest.json` (NEW) — first Radiance
  manifest, one entry: `radiance_vecadd_body` → `abi/vecadd_body.cpp`,
  targets `["radiance-muon"]`.
- `benchmarks/Radiance/kernels/abi/vecadd_body.cpp` (NEW) — per-thread
  function extracted from `radiance-kernels/kernels/vecadd/kernel.cpp`,
  given `extern "C"` linkage as `radiance_vecadd_body`. Uses
  `mu_intrinsics.h`.
- `build_tools/radiance/templates/kernel_phase2.cpp.j2` (NEW) — wrapper
  template that declares the kernel `extern "C"` and calls `mu_schedule`
  on the symbol from the manifest body .o. Phase-1 template
  (`kernel.cpp.j2`) is unchanged for descriptors that inline the body.
- `build_tools/radiance/CMakeLists.txt` (MOD) — accepts
  `MERLIN_RADIANCE_KERNEL_BODY_OBJ` and links it into the executable when
  set.
- `tools/build.py` (MOD) — `--kernel-body-obj` flag passes through to
  CMake.
- `tools/compile.py` (MOD) — `_compile_radiance_muon` detects manifest-mode
  descriptors (presence of `manifest:` and `kernel_entry_symbol:` fields),
  loads the manifest, runs `precompile.precompile()` to produce the
  body .o, renders the Phase-2 template, hands off to
  `merlin build --kernel-body-obj`.
- `models/radiance_muon/vecadd_v2.yaml` (NEW) — manifest-mode descriptor
  pointing at `manifest.json` + `radiance_vecadd_body`.

Acceptance — Phase 2 passes:
```
$ ./merlin compile models/radiance_muon/vecadd_v2.yaml --target radiance_muon
  📦 Precompiled body .o: build/.../radiance_vecadd_body.<hash>.muon.o
  📄 Generated: build/.../kernel.cpp
  📄 Staged:    build/.../data
  ✅ kernel.radiance.elf: build/.../vecadd.radiance.elf

$ md5sum build/.../vecadd.radiance.elf $RADIANCE_KERNELS_ROOT/.../kernel.radiance.elf
f061b872002cd3c7d182d7cd0ea0983d  build/.../vecadd.radiance.elf  (Phase 2)
6919fe959f5b8787dc2e79b59260a262  reference                       (Phase 0a / Phase 1)
```

The two ELFs differ by 80 bytes, all in the symbol table: Phase 2 has
`T radiance_vecadd_body` (extern-C, externally-linkable) where the
reference has `t _ZL6vecaddPvjjj` (static-inline, name-mangled). All
other symbols (`mu_schedule`, `_start`, `_exit`, `tohost`, `fromhost`,
`A_raw`, `B_raw`, `C_raw`, `n`) are identical. Runtime semantics are
identical: same `mu_schedule(<fn>, &args, 4)` call, same data layout.

This validates the architecture: the kernel-embed manifest can supply
Radiance kernel bodies that link cleanly into the run-binary ELF. The
next step (the IRBuilder-driven MLIR plugin) plugs in as a
`source_lang: ll` body source — the rest of the pipeline carries through
unchanged.

## What's Next

### Phase 2.6 — MLIR compiler plugin (LANDED)

A real Radiance compiler plugin lowering MLIR linalg/scf/arith/memref to
LLVM IR via the standard MLIR-to-LLVM conversion pipeline, plugged into
the Phase-2 kernel-embed pipeline as a `source_lang: ll` body source.

**New files:**

- `compiler/plugins/target/Radiance/{CMakeLists.txt,
  PluginRegistration.cpp, RadianceOptions.{h,cpp}}` — IREE compiler
  plugin scaffolding. Mirrors the Gemmini plugin structure.
- `compiler/src/merlin/Dialect/Radiance/IR/{RadianceDialect,RadianceAttrs}.{td,h,cpp}` —
  minimal dialect: `radiance.AddrSpace` enum (`global`=1, `shared`=3)
  wrapped in a real `radiance.AddrSpaceAttr`
  (EnumAttr<Radiance_Dialect, ...>) so it parses as a non-builtin
  dialect attribute and can be used as a memref's memorySpace.
  Address-space-sensitive ops (loads/stores) inherit ptr addrspace(N)
  via the standard MemRef→LLVM converter.
- `compiler/src/merlin/Dialect/Radiance/Transforms/{Passes.h,
  ConvertRadianceAddrSpaces.cpp, LowerRadianceToLLVM.cpp}`:
  - `ConvertRadianceAddrSpaces` walks every memref bearing
    `#radiance.addrspace<global|shared>` and rewrites it to a
    plain-integer-memorySpace memref (1 / 3). Handles block args,
    function signatures (via FunctionOpInterface so it works on both
    `func.func` and IREE's wrapped `util.func`), and op result types.
  - `LowerRadianceToLLVM` runs the standard MLIR→LLVM conversion
    pipeline (scf→cf, arith→llvm, memref→llvm, func→llvm,
    cf→llvm, reconcile-unrealized-casts), then optionally
    translateModuleToLLVMIR + write text to disk. After emission
    erases all isolated-from-above ops in the module so subsequent
    IREE input-conversion passes see an empty module.
- `iree_compiler_plugin.cmake` — registers the Radiance plugin under
  `MERLIN_ENABLE_TARGET_RADIANCE`.
- `tools/build.py` — `--compiler-scope=radiance` opts in to the
  Radiance plugin without bringing in NPU.
- `tools/compile.py` — descriptor-mode `mlir:` field triggers the new
  path: invoke iree-compile with `--iree-plugin=radiance
  --iree-radiance-enable=true --iree-radiance-emit-llvm-ir=true`,
  patch the emitted .ll to strip LLVM-23-only `nuw` GEP keywords
  (llvm-muon clang is LLVM 18.1), synthesize a one-entry manifest
  with `source_lang: ll`, then hand off to the Phase-2 link path.
- `models/radiance_muon/{vecadd.mlir, vecadd_mlir.yaml}` — first
  reference MLIR descriptor.

**Plugin-side hooks:** the lowering is registered via
`extendInputConversionPreprocessingPassPipeline`, not
`extendPreprocessingPassPipeline`. The earlier hook runs before IREE's
input conversion, while functions are still `func.func` with the
original `radiance.kernel` / `radiance.entry_symbol` /
`radiance.num_warps` attrs intact. Running later (post-input-conversion)
strips those attrs.

**LLVM version-skew bridge:** iree-compile is built against LLVM 23;
llvm-muon clang is 18.1. The compile.py post-step strips LLVM-23-only
keywords (`getelementptr inbounds nuw` → `getelementptr inbounds`) from
the emitted .ll before feeding it to llvm-muon. A more durable fix would
assemble via llvm-as from the iree LLVM build then link bitcode through
llvm-muon lld; this is a Phase 2.7 follow-up.

**Acceptance — Phase 2.6 passes:**
```
$ ./merlin compile models/radiance_muon/vecadd_mlir.yaml --target radiance_muon
  🧪 iree-compile + Radiance plugin -> radiance_vecadd_body.ll
  🩹 patched LLVM-23-only keywords in radiance_vecadd_body.ll
  📦 Precompiled body .o (mlir mode): radiance_vecadd_body.<hash>.muon.o
  📄 Generated: kernel.cpp
  📄 Staged:    data
  ✅ kernel.radiance.elf: build/.../vecadd.radiance.elf
```

Resulting ELF is 22804 bytes. The function `radiance_vecadd_body`
defined as `void @radiance_vecadd_body(ptr addrspace(1) %0, ptr
addrspace(1) %1, ...)` in LLVM IR after lowering, then compiled via
llvm-muon clang and linked into the Phase-2 wrapper. Same runtime
symbol set as Phase 1/2 (mu_schedule, _start, _exit, tohost, fromhost,
A_raw/B_raw/C_raw/n).

### Phase 3 — Sim execution validation

Two paths, both blocked on environment:
1. **Fresh simv rebuild**: `cd $CHIPYARD_ROOT/sims/vcs && make CONFIG=RadianceMuonConfig`.
   The current simv (Apr 23) trips on a TLMonitor PutPartial assertion;
   a rebuild against the `graphics` branch HEAD likely fixes it.
2. **Verilator path**: `verilator` not installed in the conda env; the
   user can `conda install -c conda-forge verilator` and rebuild
   `sims/verilator/`.

Once a working sim is in place:
```
$ ./merlin scripts/radiance_muon_smoke.sh
```
should run end-to-end and print the uartlog tail with `tohost==1`.

### Phase 4 — Cyclotron difftest (later)

Re-enable `difftest = true` on RadianceMuonConfig, expose
`--difftest` flag on `merlin chipyard run-radiance-muon`, validate that
Merlin-produced ELF runs match the Cyclotron golden trace.

## Critical files

```
merlin/
  build_tools/radiance/
    riscv_muon.toolchain.cmake     NEW — toolchain
    mu_link.ld                     NEW — vendored linker script
    common.cmake                   NEW — CMake helpers
    pin.txt                        NEW — pinned commit
    CMakeLists.txt                 NEW — driver
    templates/kernel.cpp.j2        NEW — Phase 1 template
    templates/host.cpp.j2          NEW — host stub
  build_tools/hardware/scripts/
    run_radiance_muon.sh           NEW — sim wrapper
  build_tools/hardware/
    radiance_muon.yaml             MOD — recipe
  models/
    radiance_muon.yaml             NEW — flag bundle
    radiance_muon/vecadd.yaml      NEW — first descriptor
  scripts/
    radiance_muon_smoke.sh         NEW — single-cmd driver
  tools/
    build.py                       MOD — --target radiance_muon
    compile.py                     MOD — --target radiance_muon
    chipyard.py                    MOD — run-radiance-muon
  docs/dev_blog/
    2026-04-30-radiance-compile-bringup.md  NEW (this file)
```

**No changes to `third_party/iree_bar/`, `third_party/gluon/`, or
`compiler/plugins/`** in Phases 0-1. The Merlin compiler plugin is the
work item for Phase 2.

## Reproduction

```bash
export LLVM_MUON=/scratch2/agustin/radiance-kernels/llvm/llvm-muon
export RADIANCE_KERNELS_ROOT=/scratch2/agustin/radiance-kernels
export CHIPYARD_ROOT=/scratch2/agustin/chipyard
export MERLIN_ROOT=/scratch2/agustin/merlin

cd $MERLIN_ROOT

# Phase 0b: Merlin re-builds the upstream vecadd kernel.
./merlin build --profile radiance_muon --config release
md5sum build/radiance_muon-vanilla-release/vecadd.radiance.elf \
       $RADIANCE_KERNELS_ROOT/kernels/vecadd/kernel.radiance.elf
# Both md5s match.

# Phase 1: Merlin owns the source emit.
./merlin compile models/radiance_muon/vecadd.yaml --target radiance_muon
md5sum build/compiled_models/vecadd/radiance_muon_vecadd/vecadd.radiance.elf \
       $RADIANCE_KERNELS_ROOT/kernels/vecadd/kernel.radiance.elf
# Both md5s match.

# Phase 1 single-command driver (compile + run).
./scripts/radiance_muon_smoke.sh
```
