# 2026-03-11: Radiance HAL Workstream Log

## Context and Goal

Radiance in Merlin is an out-of-tree IREE runtime HAL driver integration
focused on a transport seam:

`IREE HAL -> radiance transport interface -> Gluon-compatible backend(s)`

Current status: **active development; no validation yet on simulated/programmed
or taped-out hardware in this repo flow**.

## Implementation Changes (Current In-Tree State)

Implemented runtime surfaces under `runtime/src/iree/hal/drivers/radiance`:

- HAL driver/device registration
- allocator, buffer, semaphore, command-buffer and submission scaffolding
- target capability and occupancy helpers
- transport abstraction with backend selection

Transport backends:

- `direct_submit` (bring-up stub)
- `rpc_compat` (bring-up stub)
- `kmod` (currently `UNIMPLEMENTED`)

Runtime plugin registration is wired through:

- `iree_runtime_plugin.cmake`
- `registration/driver_module.c`

Build knobs include:

- `MERLIN_RUNTIME_ENABLE_HAL_RADIANCE`
- `MERLIN_HAL_RADIANCE_BUILD_TESTS`
- `MERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT`
- `MERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT`
- `MERLIN_HAL_RADIANCE_ENABLE_KMOD`

## What Worked

- Driver factory registration for `radiance` works at runtime plugin level.
- Device creation path is implemented with concrete HAL vtable wiring and
  transport instantiation.
- Queue transfer-related flows have host-emulated behavior:
  - `queue_fill`, `queue_update`, `queue_copy` via IREE queue emulation helpers
  - deferred command buffers can be replayed through inline command buffers
    and synchronized via transport.
- Transport seam smoke testing exists:
  - direct-submit transport smoke
  - fake-transport call-recording tests
  - device create smoke test

## What Did Not Work / Current Limitations

- Dispatch execution path is not implemented in HAL queue APIs:
  - `queue_dispatch` returns `UNIMPLEMENTED`
  - dispatch-category command buffers in `queue_execute` are rejected.
- Collectives/channels and events are not implemented.
- Executable cache creation is not implemented.
- `kmod` transport backend is explicitly `UNIMPLEMENTED`.
- `direct_submit` and `rpc_compat` are bring-up stubs:
  - currently validate arguments and bump internal counters
  - no real kernel submission or packet IO path is wired yet.

## Debugging Notes

Bring-up is easiest when forcing backend explicitly through device path:

- `radiance://direct`
- `radiance://rpc`
- `radiance://kmod` (expected to fail until implemented)

For queue behavior debugging, focus first on non-dispatch command buffers and
transport synchronization ordering, since those paths are currently active.

## Test Coverage and Commands

Primary runtime smoke target:

- `iree_hal_drivers_radiance_testing_transport_smoke_test`

Build (Merlin scripts):

```bash
conda run -n merlin-dev uv run tools/build.py \
  --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

Run:

```bash
./build/host-merlin-debug/runtime/plugins/merlin/runtime/iree/hal/drivers/radiance/testing/transport_smoke_test
```

Current tests validate transport seam and device creation only; they are not
evidence of real kernel dispatch on hardware.

## Reproduce Latest Stage (Checklist)

1. Build Radiance runtime plugin test target:
   - `conda run -n merlin-dev uv run tools/build.py --profile radiance --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test`
2. Run transport smoke test binary.
3. Confirm passing sub-tests:
   - direct-submit smoke
   - fake-transport stats
   - device-create smoke
4. (Optional) toggle transport backends via build flags:
   - `--plugin-runtime-radiance-rpc`
   - `--plugin-runtime-radiance-direct`
   - `--plugin-runtime-radiance-kmod`
5. Rebuild and rerun smoke test for each backend combination under investigation.

Note: this is runtime-driver seam validation, not full dispatch-on-hardware
validation.

## Follow-Up Tasks

- Implement executable + dispatch path end-to-end in HAL.
- Replace direct/rpc stubs with real packet/transport interactions.
- Implement `kmod` backend or remove it from default bring-up paths until ready.
- Add targeted runtime tests for semaphore/event/channel semantics.
- Add simulator/hardware-facing validation once backend path is available.
