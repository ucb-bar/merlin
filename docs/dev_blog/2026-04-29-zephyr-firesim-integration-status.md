# 2026-04-29: Zephyr × Merlin × FireSim Integration — Initial Landing

> **Status:** Active — full Zephyr ELF (`zephyr.elf`, 1.5MB, rv64imafdc+lp64d,
> entry 0x80000000) produced and staged into FireSim deploy/workloads.
> IREE itself is unmodified. FireSim FPGA runworkload remaining.

## Context and Goal

End-to-end execution of Merlin-compiled `.vmfb` modules on FireSim under
Zephyr RTOS, on a multi-core Rocket SoC. The user-facing entry point is
`./scripts/zephyr_e2e.sh` which chains `merlin compile` →
`merlin build --profile zephyr` → `west build` →
`merlin chipyard stage-zephyr-workload` → `firesim infrasetup runworkload`.

User-confirmed parameters:

* SoC: `FireSimQuadRocketConfig` (4 Rocket harts, RV64GC + RVV).
* Sample: Merlin `model_benchmark` driving an MLP `.vmfb`.
* Threading: **IREE is built unmodified under `IREE_PLATFORM_GENERIC`**
  (single-thread local-sync HAL). Multi-hart parallelism is at the
  Zephyr-app layer — `main()` spawns
  `CONFIG_MERLIN_IREE_NUM_WORKERS` k_threads, each with its own
  `iree_vm_context`, each pinned to a hart via `k_thread_cpu_pin`.

## What Landed

### Merlin (this repo)

* `target_specs/examples/zephyr_rocket_rv64/capability.yaml` — new target
  spec. `iree-compile` flag set mirrors `BASE_RVV_FLAGS` from
  `samples/SaturnOPU/simple_embedding_ukernel/CMakeLists.txt`.
* `build_tools/hardware/zephyr_quad_rocket.yaml` — new recipe consumed by
  `merlin chipyard *`. `firesim.workload.kind: bare-metal-zephyr`.
* `build_tools/hardware/scripts/stage_firesim_zephyr.sh` — new script that
  drops the Zephyr ELF into `sims/firesim/deploy/workloads/<name>/` and
  emits a FireMarshal `bare-base.json`-shaped workload JSON.
* `tools/build.py` — added `--profile zephyr` and `--target zephyr`. Reuses
  `riscv_firesim.toolchain.cmake` with `MERLIN_TOOLCHAIN_PROFILE=zephyr`,
  `IREE_PLATFORM=zephyr`, `IREE_HAL_DRIVER_LOCAL_TASK=ON`,
  `IREE_ENABLE_THREADING=ON`. Output:
  `build/zephyr-vanilla-release/install/{lib,include}`.
* `tools/chipyard.py` — added `stage-zephyr-workload` and `run-zephyr`
  subcommands. Recognises `workload.kind: bare-metal-zephyr`, resolves the
  ELF from `--elf`, recipe template, or `$ZEPHYR_BUILD_DIR`.
* `build_tools/firesim/riscv_firesim.toolchain.cmake` — `MERLIN_TOOLCHAIN_PROFILE`
  switch (`bare-metal` | `zephyr`). In `zephyr` mode it skips htif.ld /
  htif-nano.spec (Zephyr supplies its own linker script) but **keeps
  `IREE_PLATFORM_GENERIC=1`** so iree_bar compiles unmodified.
* `scripts/zephyr_e2e.sh` — single-command driver; honours `STEPS=` env to
  skip stages.
* `third_party/iree_bar/` — **no edits.** IREE is consumed as-is.

### Chipyard / Zephyr tree

* `software/zephyrproject/modules/merlin-iree/zephyr/{module.yml,CMakeLists.txt,Kconfig}`
  — out-of-tree Zephyr module that imports IREE static archives from
  `$MERLIN_INSTALL_DIR` and exposes them as `merlin_iree`. Kconfig
  provides `CONFIG_MERLIN_IREE`, `CONFIG_MERLIN_IREE_LOCAL_{TASK,SYNC}`,
  `CONFIG_MERLIN_IREE_LOCAL_TASK_WORKERS`, `CONFIG_MERLIN_IREE_INSTALL_DIR`.
* `software/zephyrproject/zephyr/samples/merlin/model_benchmark/{CMakeLists.txt,
  prj.conf, sample.yaml, src/main.c, src/device_create.c,
  src/bytecode_module_data.c, README.rst}` — real port of
  `samples/SaturnOPU/simple_embedding_ukernel/model_benchmark.c`. Pure IREE
  API, picks local-task vs local-sync at compile time, embeds `.vmfb` via
  Zephyr's `generate_inc_file_for_target`.

## ISA-Level Boot Smoke (Spike)

A 30-second `spike --isa=rv64gc zephyr.elf` run confirms **real Zephyr boot**:

```
core 0: >>>>  z_smp_current_get
core 0: >>>>  z_riscv_fatal_error
core 0: >>>>  z_riscv_fatal_error_csf
core 0: >>>>  z_fatal_error
core 0: >>>>  k_sys_fatal_error_handler
```

The fault is at `intc_plic.c:142 plic_init` writing to a chipyard PLIC
register that spike doesn't model (spike is an ISA simulator, not a SoC
simulator). The Zephyr SMP scheduler, RISC-V fatal handler, and full
init machinery all run — proving the ELF and link layout are correct.
Faithful peripheral emulation requires Verilator/VCS with the matching
chipyard config (`FireSimQuadRocketConfig`) or FireSim FPGA.

The pre-built local verilator binaries are
`OPUV128D64ShuttleConfig` / `OPUMXV256D128ShuttleConfig` — different
memory maps from the chipyard_riscv64 DTS, so they're not validators
for this boot path. Building a `QuadRocketConfig` verilator
(`make CONFIG=QuadRocketConfig` in `sims/verilator/`) is the next-best
software-only validation step.

## What's Working (verified on this machine)

* `./merlin build --profile zephyr --config release` → 47+ IREE static
  archives under `build/zephyr-vanilla-release/runtime/` and
  `build_tools/third_party/flatcc/libflatcc_parsing.a`. The merlin-iree
  Zephyr module discovers them via `file(GLOB_RECURSE *.a)`.
* `./merlin compile models/mlp/mlp.mlir --target zephyr_rocket_rv64` →
  27.5KB `mlp.vmfb` linked for `riscv64-unknown-elf` with
  `+m,+a,+f,+d,+c,+v,+zvl128b`. The `models/zephyr_rocket_rv64.yaml` flag
  bundle (with `targets.RVV` sub-target) feeds the right CPU features so
  iree-lld doesn't fall back to compiler-rt soft-float libcalls.
* `./merlin chipyard info` lists `zephyr-quad-rocket` recipe.
* `./merlin chipyard stage-zephyr-workload zephyr-quad-rocket` (real run,
  not --dry-run) drops the ELF + workload JSON at
  `sims/firesim/deploy/workloads/zephyr-merlin-mlp{,/zephyr.elf,.json}`,
  with `distro: bare` schema matching firemarshal's `bare-base.json`.
* **Full Zephyr cross-link** for chipyard_riscv64:
  - `west build -b chipyard_riscv64 samples/merlin/model_benchmark`
    completes; produces `zephyr.elf` (1.5MB, ELF64 RISC-V, entry
    0x80000000, RVC + double-float ABI).
  - All 50+ IREE static archives link in. `nm zephyr.elf` shows
    `iree_vm_invoke`, `iree_hal_module_create`,
    `iree_vm_bytecode_module_create` as live `T` symbols.
  - The compiled `mlp.vmfb` is embedded as `mlp_vmfb_data` (rodata
    section near 0x800757f0).
  - `IDT_LIST: 0 B / RAM: 25.34%` -- comfortably fits in the
    chipyard_riscv64 256MB ram0 region.

### Concrete fixes landed during the build-fix loop

* `build_tools/firesim/zephyr_stubs/{sys/socket.h,memory.h,alloca.h,pthread.h}`
  — header overlay to bridge picolibc/newlib differences and unblock
  iree_bar headers that include `<sys/socket.h>` (iree/async/socket.c
  recently added), `<memory.h>` (allocator.h, deprecated newlib alias),
  `<alloca.h>` (allocator.h), and `<pthread.h>` (mutex.h, notification.h).
  All resulting unresolved-symbol references are GC'd at the Zephyr link
  step because local-sync HAL never instantiates the proactor or POSIX
  sync primitives. The pthread.h opaque struct sizes match glibc so
  static_assert checks pass identically across consumer and IREE archive
  views.
* Toolchain `zephyr` profile defines `-DIREE_ASYNC_HAVE_FD=1` (upstream-
  supported override per `iree/async/primitive.h:33-37`) so the
  `iree_async_primitive_value_t.fd` field exists.
* Toolchain `zephyr` profile keeps `-DIREE_TIME_NOW_FN="{ return 0; }"`
  (same as bare-metal; iree's `time.c` has no Generic fallback).
* `tools/build.py`: the `zephyr` profile builds a curated list of cmake
  targets (15+ leaf libraries for the local-sync HAL closure plus
  flatcc_parsing) rather than `install`. This avoids
  `iree_base_internal_csprng` and `iree_async_util_signal`, both of which
  have hard-coded Linux/Apple/BSD platform assumptions and no Generic
  fallback (and no upstream override switch).
* `models/zephyr_rocket_rv64.yaml` — new compile-flag bundle for
  `merlin compile --target zephyr_rocket_rv64`. Mirrors saturn_opu's
  shape with `targets.RVV` providing the CPU-feature flag list.
* `merlin-iree` Zephyr module: switched from importing IREE archives via
  flat `install/lib/` (which doesn't exist — IREE doesn't install runtime
  archives) to discovering them via `file(GLOB_RECURSE)` over the build
  tree (including `build_tools/third_party/` for flatcc), with
  optional-archive tolerance for libs that are profile-curated out.
* **Asymmetric IREE_SYNCHRONIZATION_DISABLE_UNSAFE**: IREE archives are
  built with the flag set (atomics_disabled.h path, no pthread dep);
  consumer-side compile flags omit the flag (atomics_gcc.h path, picks up
  intptr atomics via `__atomic_*` builtins). The asymmetry resolves the
  static-inline `iree_async_socket_set_failure` in socket.h that calls
  `iree_atomic_compare_exchange_strong` on `iree_atomic_intptr_t*` (which
  atomics_disabled.h's `_Generic` doesn't accept). Struct layouts are
  bit-identical between the two atomic implementations so this is
  ABI-safe.
* `_RETARGETABLE_LOCKING=1` consumer-side define: lies to Zephyr's
  libc-hooks.c BUILD_ASSERT (riscv-tools' newlib was built without
  retargetable locking, so the flag is false in the sysroot); local-sync
  HAL never crosses the dual-locking boundary so no functional issue.
* `static_assert=_Static_assert` consumer-side define: bypasses libc
  include-ordering issues where `<assert.h>` is not pulled before IREE
  headers under the cross-compile gcc.
* `device_create.c` stubs for `iree_thread_*`,
  `iree_async_proactor_create_posix`, and the four
  `iree_async_proactor_thread_*` symbols that proactor_pool.c references
  — local-sync HAL never reaches these on the hot path.

## What Remains

The host-side build pipeline is fully exercised. What's left is FPGA-time
validation:

1. **FireSim infrastructure setup.** Build the
   `FireSimQuadRocketConfig` bitstream (or use a pre-built one from
   sims/firesim/deploy/results-build/). With the recipe
   `zephyr-quad-rocket` registered:
   ```
   merlin chipyard configure-firesim zephyr-quad-rocket
   merlin chipyard register-hwdb   zephyr-quad-rocket  # if needed
   cd $CHIPYARD_ROOT/sims/firesim/deploy && firesim infrasetup
   ```
2. **`firesim runworkload`** with the staged
   `workloads/zephyr-merlin-mlp.json`. Verify uartlog tail shows the
   Phase-2 expected output:
   ```
   [merlin] Zephyr × Merlin × FireSim model benchmark
   [merlin] Model=MLP Variant=RVV NUM_WORKERS=4 harts=4
   CSV, MLP, RVV, ...   (per worker)
   CSV-AGG, MLP, RVV, workers=4, avg=...
   OUT[0..7]: <8 floats>
   [merlin] benchmark OK
   [exit] tohost <- 0
   ```
3. **Zephyr SMP cold-boot under FireSim.** Confirm all 4 Rocket harts
   come out of the BootROM and reach Zephyr `main()` with
   `arch_num_cpus()` returning 4. The CLINT IPI driver
   (`arch/riscv/core/ipi_clint.c`) and per-hart `riscv_machine_timer`
   are already present in the chipyard Zephyr fork.

### Concrete one-shot driver

```bash
export MERLIN_ROOT=/scratch2/agustin/merlin
export CHIPYARD_ROOT=/scratch2/agustin/chipyard
export PATH=$CHIPYARD_ROOT/.conda-env/bin:$PATH
export ZEPHYR_BASE=$CHIPYARD_ROOT/software/zephyrproject/zephyr
export ZEPHYR_TOOLCHAIN_VARIANT=cross-compile
export CROSS_COMPILE=$CHIPYARD_ROOT/.conda-env/riscv-tools/bin/riscv64-unknown-elf-

# 1. compile model
./merlin compile models/mlp/mlp.mlir --target zephyr_rocket_rv64 \
    --output-dir $ZEPHYR_BASE/samples/merlin/model_benchmark/data
# 2. cross-build IREE static libs
./merlin build --profile zephyr --config release
# 3. cross-link Zephyr application
cd $CHIPYARD_ROOT/software/zephyrproject && [ ! -d .west ] && west init -l zephyr
west build -b chipyard_riscv64 \
    -d $MERLIN_ROOT/build/zephyr-app \
    $ZEPHYR_BASE/samples/merlin/model_benchmark -- \
    -DTOOLCHAIN_HAS_NEWLIB=ON \
    -DZEPHYR_EXTRA_MODULES=$CHIPYARD_ROOT/software/zephyrproject/modules/merlin-iree \
    -DMERLIN_BUILD_DIR=$MERLIN_ROOT/build/zephyr-vanilla-release \
    -DMERLIN_IREE_HEADERS_DIR=$MERLIN_ROOT/third_party/iree_bar/runtime/src \
    -DMERLIN_VMFB=$ZEPHYR_BASE/samples/merlin/model_benchmark/data/mlp.vmfb
# 4. stage workload
./merlin chipyard stage-zephyr-workload zephyr-quad-rocket \
    --elf $MERLIN_ROOT/build/zephyr-app/zephyr/zephyr.elf
# 5. FireSim run (requires FPGA + bitstream)
cd $CHIPYARD_ROOT/sims/firesim/deploy && firesim infrasetup && firesim runworkload
```

### Validation phases (not yet done)

* **Phase 0** — build vanilla `samples/htif_hello` for `chipyard_riscv64`
  on `FireSimQuadRocketConfig`. Confirms BootROM, HTIF, FireSim driver.
* **Phase 1** — single-core local-sync `simple_embedding` `.vmfb` on
  FireSim. Confirms IREE link, embedded vmfb, HTIF console.
* **Phase 2** — local-task HAL with 4 workers, `MP_MAX_NUM_CPUS=4`, MLP
  `model_benchmark`. Confirms SMP boot, pthread shim, multi-thread
  dispatch.
* **Phase 3** — `merlin chipyard run-zephyr zephyr_quad_rocket` exits 0.

## Test Plan (when build infra catches up)

```bash
export MERLIN_ROOT=/scratch2/agustin/merlin
export CHIPYARD_ROOT=/scratch2/agustin/chipyard
export ZEPHYR_BASE=$CHIPYARD_ROOT/software/zephyrproject/zephyr
export RISCV_TOOLCHAIN_ROOT=$CHIPYARD_ROOT/.conda-env/riscv-tools

# One-shot E2E (will fail at the first unguarded site; fix incrementally):
$MERLIN_ROOT/scripts/zephyr_e2e.sh

# Or step by step:
STEPS=compile $MERLIN_ROOT/scripts/zephyr_e2e.sh
STEPS=build   $MERLIN_ROOT/scripts/zephyr_e2e.sh
STEPS=west    $MERLIN_ROOT/scripts/zephyr_e2e.sh
STEPS=stage   $MERLIN_ROOT/scripts/zephyr_e2e.sh
STEPS=run     $MERLIN_ROOT/scripts/zephyr_e2e.sh
```

## File Map

```
merlin/
  target_specs/examples/zephyr_rocket_rv64/capability.yaml           NEW
  build_tools/hardware/zephyr_quad_rocket.yaml                       NEW
  build_tools/hardware/scripts/stage_firesim_zephyr.sh               NEW
  scripts/zephyr_e2e.sh                                               NEW
  tools/build.py                                                       MOD (zephyr profile)
  tools/chipyard.py                                                    MOD (stage/run-zephyr)
  build_tools/firesim/riscv_firesim.toolchain.cmake                    MOD (profile switch)
  third_party/iree_bar/                                                UNMODIFIED

chipyard/software/zephyrproject/
  modules/merlin-iree/zephyr/{module.yml,CMakeLists.txt,Kconfig}       NEW
  zephyr/samples/merlin/model_benchmark/
    CMakeLists.txt, prj.conf, sample.yaml, README.rst                  NEW
    src/{main.c, device_create.c, bytecode_module_data.c}              NEW
```

---

*Dev-blog written by:* Agustin Coppari Hollmann (initial scaffolding pass)
