# Radiance HAL Driver (Merlin Runtime Plugin)

Development of this section is only possible by basing ourselves on the development from the authors of the following repositories:

- [Radiance](https://github.com/ucb-bar/radiance/tree/main)
- [Gluon](https://github.com/Sh-Anand/gluon)
- [Radiance-Kernels](https://github.com/ucb-bar/radiance-kernels)

For ISA information and documentation and proper attribution to the work of this authors please check their work using the link above.

This directory is the out-of-tree IREE HAL driver integration point for Radiance/Muon in Merlin.

The design split is intentional:

```text
IREE HAL (driver/device/cmd buffer/executable/semaphore)
  -> radiance transport interface (this repo)
    -> Gluon-compatible backend(s)
```

The HAL should own IREE semantics. Transport should own "how bytes/packets get to Gluon".

## Current Status

- External HAL registration is wired via Merlin runtime plugin CMake.
- `iree_hal_radiance_device_create(...)` now constructs a concrete
  `iree_hal_device_t` with allocator/semaphore/queue vtable wiring.
- Radiance transport backends exist for:
  - `direct_submit` (bring-up stub)
  - `rpc_compat` (bring-up stub)
  - `kmod` (placeholder, unimplemented)
- Unit/smoke tests exist for the transport seam.
- Queue transfer paths are currently host-emulated via command-buffer replay;
  dispatch/executable paths are still `UNIMPLEMENTED`.

## Key Files and Ownership

- HAL API and driver/device entry:
  - `api.h`
  - `driver.c`
  - `device.c`
- Radiance target model:
  - `target_caps.{h,c}`
  - `occupancy.{h,c}`
- Dispatch/ABI shaping:
  - `dispatch_builder.{h,c}`
  - `executable.{h,c}`
  - `executable_cache.{h,c}`
- Submission and command recording seam:
  - `command_buffer.{h,c}`
  - `submission.{h,c}`
- Transport seam:
  - `transport/transport.{h,c}`
  - `transport/rpc_compat.c`
  - `transport/direct_submit.c`
  - `transport/kmod.c`
- Registration:
  - `registration/driver_module.c`
- Test seam:
  - `testing/fake_transport.{h,c}`
  - `testing/transport_smoke_test.cc`

## Build/Setup Through Merlin Scripts

Do not pass ad hoc CMake flags manually for normal development; use Merlin scripts.

Environment setup:

```bash
python tools/setup.py env --env-name merlin-dev
```

Minimal runtime-plugin check (avoids common network fetch dependencies):

```bash
python tools/build.py \
  --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

Run the smoke test:

```bash
./build/host-merlin-debug/runtime/plugins/merlin/runtime/iree/hal/drivers/radiance/testing/transport_smoke_test
```

## Plugin Knobs (build.py / iree_runtime_plugin.cmake)

Compiler/runtime split:

- `--plugin-compiler` / `--no-plugin-compiler`
- `--plugin-runtime` / `--no-plugin-runtime`

Runtime slice controls:

- `--plugin-runtime-radiance` / `--no-plugin-runtime-radiance`
- `--plugin-runtime-samples` / `--no-plugin-runtime-samples`
- `--plugin-runtime-benchmarks` / `--no-plugin-runtime-benchmarks`
- `--plugin-runtime-radiance-tests` / `--no-plugin-runtime-radiance-tests`
- `--plugin-runtime-radiance-rpc` / `--no-plugin-runtime-radiance-rpc`
- `--plugin-runtime-radiance-direct` / `--no-plugin-runtime-radiance-direct`
- `--plugin-runtime-radiance-kmod` / `--no-plugin-runtime-radiance-kmod`

## Where We Need Input From Gluon/Radiance Maintainers

### From Gluon maintainers

Please validate and/or provide:

1. Current command packet wire layout and required alignment constraints.
2. Definitive command completion/error packet format (command-id semantics).
3. RPC compatibility contract we should freeze for bring-up (`rad_rpc` behavior).
4. Direct-submit boundary expectations (which fields belong in kernel/mem/wait packets).
5. Stream/event semantics (ordering, fence behavior, wait behavior).

### From Radiance / radiance-kernels maintainers

Please validate and/or provide:

1. Launch ABI contract (`_start`, kernel symbol, param packing, TLS/stack/printf fields).
2. Stable executable metadata exported by toolchain (regs/thread, shmem usage, occupancy hints).
3. Canonical target capability values we should expose in HAL (warp size, shmem limits, etc.).
4. Expected behavior for shared/global address-space usage in kernels.
5. Recommended kernel corpus and expected outputs for bring-up CI.

Tracked question list (maintained in Merlin):

- [`docs/architecture/radiance_author_questions.md`](../../../../../../docs/architecture/radiance_author_questions.md)

## Non-Goals for This Layer

- This HAL should not mirror RTL block internals directly.
- It should not call `rad.h` APIs as the primary HAL abstraction.
- It should not encode Neutrino/Gemmini orchestration in v1 unless ABI is stabilized for runtime use.
