---
title: Zephyr runtime backend
kind: guide
status: current
owner: runtime
last_verified: 2026-09-29
related: [getting_started, reproducibility, runtime, tinyllama_int8_rvv_zephyr]
code_refs: [src/merlin/runtime/backends/zephyr_model.py, src/merlin/runtime/boards.py]
---

# Zephyr runtime backend

> For the whole-model **multicore** RVV path on Zephyr (the OpenMP shim over pinned harts,
> sustained inference, and the multicore Saturn SoC) see
> [TinyLlama int8 on multicore RVV under Zephyr](tinyllama_int8_rvv_zephyr.md).

Zephyr is a **runtime backend** for Merlin, not part of the core compiler model. Merlin owns
the generic runtime API; the target provides a Zephyr driver that implements it.

```
runtime dialect
  -> Merlin generic runtime API  (merlin_submit / merlin_wait / merlin_get_metrics)
  -> target-specific Zephyr driver (MMIO / DMA / interrupts / counters)
```

## Prerequisites

**Shared base:** complete the base install + `.env` setup in [Getting started](getting_started.md)
first. `check_repro_env.py` checks the combined `zephyr_spike` capability when
running an image on Spike; building an image does not require Spike.

**Workflow-specific prerequisites** (only for building/running the whole-model `build_app` path; the
generic module-layout description below needs none):

- **Required — selected Zephyr tree + SDK 0.17.0**: `ZEPHYR_BASE` (the zephyr tree)
  and `ZEPHYR_SDK_INSTALL_DIR`. `MERLIN_ZEPHYR_SW` is only an optional shell
  convenience for commands that navigate the containing workspace; Merlin does not read it.
- **Required — RISC-V cross compiler** via `MERLIN_RISCV_GCC` or `MERLIN_CHIPYARD`
  for the image build. `zephyr_model.build_available()` checks build prerequisites.
- **Spike only for a Spike run** via `MERLIN_CHIPYARD` or `MERLIN_SPIKE`.
  `zephyr_model.available()` checks the combined build-and-Spike path;
  `build_app` uses the build-only check.
- **Optional — FireSim** (2-tile SMP) is board/FPGA-gated and **not fresh-machine reproducible** (see
  [Getting started §5](getting_started.md)). It also needs a separately installed,
  external operator through [FireSim prepare/collect](firesim.md); **spike substitutes** for the functional whole-model run.
- **Required — board catalog and selection**: set `MERLIN_BOARD_CATALOG` to a
  target-owned YAML catalog and pass its board name to `build_app(board=...)` or
  `merlin-compile --board ...`. The [example catalog](../../examples/board-catalog.yaml)
  is for demonstrations; it is not bundled into the Merlin wheel. No board or
  unknown board fails before build output is created.

For a Zephyr board, the catalog must also declare `zephyr_default_ram_bytes`
(the selected port's unmodified device-tree region) and
`zephyr_link_limit_bytes` (the model-object relocation window). If weights and
arena would exceed that window, Merlin uses a separate weights region **only**
when the board declares `zephyr_external_ram_bytes` and
`zephyr_external_tail_reserve_bytes`. The weights base is computed as
`dram_base + zephyr_external_ram_bytes`; the aligned blob and reserved tail
must fit inside `dram_bytes`. No 16 GiB board, fixed `ram0` address, or Spike
memory size is assumed by the runtime. A missing or impossible layout is an
error before the Zephyr link; the build result records the low RAM region,
simulator span, and weights base for inspection. Mark a simulator descriptor
with `simulator: spike` if the certification runner should select it as its
functional substrate.

## Generated module layout

`merlin.targetgen.generate.zephyr_module` produces, from a `zephyr_plan.yaml`:

```
zephyr/
├── module.yml                 # Zephyr module manifest
├── CMakeLists.txt
├── Kconfig                    # MERLIN_RUNTIME[_PROFILING], rsources driver Kconfig
├── dts/bindings/accelerator/ucb,<target>.yaml
├── drivers/accelerator/<short>_driver.c   # implements merlin_driver_api (blocking)
├── include/merlin/{runtime.h,command_buffer.h,metrics.h,<short>.h}
├── samples/<target>_repeated_rhs_matmul/{CMakeLists.txt,prj.conf,app.overlay,src/main.c}
└── tests/<short>_driver/
```

The generated C is structurally plausible but **non-building** placeholder scaffold.

## Ownership split

- **Merlin owns**: the generic runtime API surface (`merlin_submit`/`merlin_wait`/
  `merlin_get_metrics`), the command-buffer ABI, and the metrics/trace schemas.
- **Target owns**: the driver body — MMIO/RoCC/interrupt/DMA mechanics, counter readout, and
  command-packet decoding.

## Backend modes

1. **Blocking driver** (MVP, generated first): `submit` / `wait` / `get_metrics`.
2. **Interrupt-driven completion**: ISR completion event + kernel-object wakeup + latency
   metrics.
3. **RTIO backend**: submission/completion queues, operation chains, async command batches.
   Add only after command batching and DMA overlap matter — never first.

Devicetree describes hardware instances (base addresses, interrupts, DMA channels,
resident-store bytes, accumulator entries, queue depth); Kconfig gates the feature at build
time. The generic binding `ucb,merlin-accelerator` can be extended by target bindings.
