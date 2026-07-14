---
title: Zephyr runtime backend
kind: guide
status: current
owner: runtime
last_verified: 2026-07-14
related: [runtime]
code_refs: [merlin/python/merlin/runtime/backends/zephyr_model.py]
---

# Zephyr runtime backend

Zephyr is a **runtime backend** for Merlin, not part of the core compiler model. Merlin owns
the generic runtime API; the target provides a Zephyr driver that implements it.

```
runtime dialect
  -> Merlin generic runtime API  (merlin_submit / merlin_wait / merlin_get_metrics)
  -> target-specific Zephyr driver (MMIO / DMA / interrupts / counters)
```

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
