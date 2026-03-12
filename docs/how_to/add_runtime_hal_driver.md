# Add Or Modify A HAL Driver

This guide describes the Merlin runtime-plugin path for HAL drivers, using
Radiance as the concrete example.

## 1) Driver Layout

Put driver sources under:

- `runtime/src/iree/hal/drivers/<driver_name>/`

Radiance reference:

- `runtime/src/iree/hal/drivers/radiance`

Typical files:

- `api.h` (public options/types)
- `driver.c` (HAL driver)
- `device.c` (HAL device + queue behavior)
- `registration/driver_module.c` (factory registration)
- transport/backend seam (optional but recommended)
- `testing/` smoke tests

## 2) Register Driver In Runtime Plugin CMake

Use `iree_register_external_hal_driver` in:

- `iree_runtime_plugin.cmake`

Radiance registration uses:

- `DRIVER_TARGET iree::hal::drivers::radiance::registration`
- `REGISTER_FN iree_hal_radiance_driver_module_register`

## 3) Add Build Toggles

Expose toggles in:

- `iree_runtime_plugin.cmake`
- `tools/build.py`

Radiance examples:

- `MERLIN_RUNTIME_ENABLE_HAL_RADIANCE`
- `MERLIN_HAL_RADIANCE_BUILD_TESTS`
- backend toggles for `rpc`, `direct`, `kmod`
- CLI flags:
  - `--plugin-runtime-radiance`
  - `--plugin-runtime-radiance-tests`
  - `--plugin-runtime-radiance-rpc`
  - `--plugin-runtime-radiance-direct`
  - `--plugin-runtime-radiance-kmod`

## 4) Think About Compiler/HAL Interaction

Compiler and HAL responsibilities should stay separate:

- compiler decides executable formats and target backend flags
- HAL runtime decides device/queue/transport behavior at execution time

For integration planning:

1. define expected executable format/backend flags during compilation
2. define device URI + driver registration path at runtime
3. stage functionality:
   - transfer/copy/fill
   - queue execution semantics
   - dispatch/executable support
   - sync/semaphores/channels/events

## 5) Add Smoke Tests Early

Keep at least one transport seam smoke test while dispatch path is evolving.

Radiance example:

- `runtime/src/iree/hal/drivers/radiance/testing/transport_smoke_test.cc`

Build and run:

```bash
conda run -n merlin-dev python tools/build.py \
  --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

```bash
./build/host-merlin-debug/runtime/plugins/merlin/runtime/iree/hal/drivers/radiance/testing/transport_smoke_test
```

## 6) Current Development Caveat

If your driver is in staged bring-up (like current Radiance), document clearly:

- what works (registration, basic transfer seam, smoke tests)
- what is still unimplemented (`dispatch`, `kmod`, events, etc.)
- that passing tests are seam validation, not taped-out hardware validation
