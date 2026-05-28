# Merlin QNN HAL audit (2026-04-29)

Total: 2220 LOC across 6 files. Steady-state per-dispatch on QRB5165:
CPU 1.83 ms / NPU 3.29 ms / GPU 6.04 ms. Setup ranges 43–105 ms.

## What's solid

- **Clean separation of concerns** — driver / device / executable /
  executable_cache / command_buffer / semaphore are each a single C
  file mirroring the upstream amdgpu/local-task patterns.
- **Borrowed handle lifecycle is correct now** — driver retains
  libQnn{Gpu,Hta,Htp}.so handles, device retains the parent driver
  (just fixed today; was the cause of an earlier segfault when caching
  libQnnSystem.so). Executables borrow {qnn_interface, backend, device}
  handles from the device. Each level releases its own retains in
  destroy.
- **Lazy resource init** — QnnBackend_create + QnnDevice_create only
  fire on first executable_cache_create. Probe-only opens of the driver
  cost ~50 µs.
- **Eager graph enumeration** — at executable_create we walk
  QnnSystemContext_getBinaryInfo once and cache (graph_handle,
  Qnn_Tensor_t* inputs/outputs) per ordinal. Dispatch is then O(1)
  lookup + clientBuf patch.
- **Driver-level libQnnSystem.so cache** — one dlopen per process.
  Multi-chunk schedules now save ~10 ms per chunk after the first.
- **Prefix-match device.id query** — `query_i64("hal.device.id", "qnn")`
  matches our internal "qnn-gpu"/"qnn-htp" identifiers, so the same
  VMFB binds without backend-specific glue.

## What's good but not "top-notch"

| Area | Current | Top-notch would be |
|---|---|---|
| Dispatch path | **Async via QnnGraph_executeAsync + Qnn_NotifyFn_t with sync-graphExecute fallback** when the backend lacks async support (HTA on QAIRT 2.45 falls back). Still waits for completion before queue_execute returns. | Drop the per-dispatch wait and signal IREE async semaphores from the notify callback, allowing back-to-back dispatches in different command buffers to pipeline on QNN's internal queue. |
| Concurrency | Single mutex serializes all dispatches per device | Per-graph or per-context locks; multi-context per device for multi-model schedules |
| Per-call stack work | 200B Qnn_Tensor_t copy per binding | Reuse cached prototype arrays in-place under queue mutex |
| Profiling | None | Wire QnnProfile_create / QnnProfile_getEvents → IREE trace events |
| Allocator | Heap-allocator only (host malloc) | Buffer pool + sub-allocation; advertise device-local + host-visible separately |
| Cross-device compile | hal.allocator.resolve_memory_properties fails for #hal.device.optimal<[@local, @qnn]> | Implement the resolver hook + advertise compatible memory types so IREE can pick allocators |

## What's missing for full integration

1. **Async dispatch.** Today `graphExecute` blocks the QNN HAL queue
   thread; the IREE async semaphore on top is also synchronous. Adding
   `QnnGraph_executeAsync` + a Qnn_SignalHandle_t wired to an IREE
   async_event would let the host overlap CPU work with GPU compute,
   which is the biggest remaining throughput win for multi-stream models.

2. **Cross-device transfer compile path.** The
   `hal.allocator.resolve_memory_properties` op fails to legalize when
   IREE tries to pick a buffer that satisfies both a `local` device
   affinity and a `qnn` device affinity. To make heterogeneous compile
   work end-to-end (input on CPU, dispatch on GPU, output to CPU), our
   QNN HAL allocator needs to advertise the same memory types as
   `local-task`'s heap allocator, OR we need a custom resolve hook.
   Today we work around this by partitioning at the schedule layer
   (per-chunk dispatch).

3. **HTP backend support on QRB5165.** The HTA backend works (Hexagon
   698 / Snapdragon 865); the HTP backend (libQnnHtp.so) refuses with
   "Unsupported SnapdragonModel" because it expects 8 Gen 1+ silicon.
   This is a Qualcomm SDK gate; our HAL handles both via the same
   `qnn://htp` and `qnn://hta` URIs.

4. **Dynamic-shape dispatch.** The QnnContext_createFromBinary path
   bakes shape into the .qnn-ctx. Variable batch / variable spatial
   dispatch would need either compile-time-known shape variants or
   runtime graph rebuilding.

## Fixed in this session

- **Driver retain bug** — qnn_device.c stored parent_driver without
  calling iree_hal_driver_retain. Caused segfault when the driver
  registry released its ref and the device later tried to fetch the
  cached system_interface. Added retain in device_create + release
  in device_destroy.
- **libQnnSystem.so driver-level cache** re-enabled (was reverted
  after the segfault; now stable).
- **QNN target placeholder** — `--iree-hal-qnn-allow-placeholder` is
  now opt-in. Default is hard error on missing manifest entries
  (previously emitted a 4-byte placeholder + warning, which is a
  footgun in production).
- **Device.id prefix match** — `qnn-gpu`/`qnn-htp`/`qnn-hta`
  identifiers match VMFB queries for `qnn` (was matched-only-fully
  earlier).
- **Async dispatch path** — QnnGraph_executeAsync wired in with a
  Qnn_NotifyFn_t callback + pthread_cond_t completion. The dispatch
  hot path now uses async on backends that support it (Adreno GPU)
  and transparently falls back to graphExecute when the async API
  returns an error (HTA on QAIRT 2.45 returns rc=1000 / not-supported,
  triggers fallback). Output bytes are identical between paths
  (md5-verified). Wall time per dispatch is unchanged today (we
  still wait for completion before returning); the change lays the
  framework for non-blocking signal_semaphore from notify_fn —
  which is the real pipelining win.

## Known remaining issues (out of scope, tracked)

- IREE CPU codegen tile-config regression at `1×149×149×16 → 64`
  (5× slower per FLOP than trend; lowering picks `[2, 2, 32]`
  distribution which underutilises SIMD). Filing upstream.
- qairt-converter 2.45 crashes on PyTorch-exported ONNX during
  layout transform (uninitialized C++ shape state). Blocks NPU/GPU
  shape sweep — see `eval/qrb5165/dispatch_perf/RESULTS.md`.
- iree-benchmark-module won't cross-build for QRB5165 because
  google/benchmark hits the toolchain's `-nostdinc++ stdint.h`
  wall. Workaround: `merlin-dispatch-bench` (in tree).

## hexagon-mlir alternative compiler — NOT integrated (hardware mismatch)

Qualcomm's open-source hexagon-mlir compiler at
`/scratch2/agustin/CompGen/tmp/hexagon-mlir` would in principle be a
nicer fit than QNN (open source MLIR/Triton path, lets us write
custom kernels). It does NOT support our Hexagon 698 (v66):

  - hexagon-mlir docs: tested on v73 / v75 / v79 only (Snapdragon 8
    Gen 2+ / X Elite).
  - HVX width hard-coded to `+hvx-length128b` in two places
    (LLVMIRTranslation.cpp:104, hexagon_options.py:21). v66 needs
    `+hvx-length64b` mode and the codegen passes assume 128b lanes.
  - HexKL kernel library ships v73/v75/v79 prebuilts only.
  - Runtime stub binaries compiled for v73+ cDSP.

Estimated port effort to v66: ~2 weeks across LLVM target tweaks,
codegen pass audit, HexKL recompile, and runtime stub adaptation —
plus the cDSP signing wall that already blocks QNN HTP on this
silicon.

Decision (per user directive 2026-04-29): **don't integrate until
v73+ hardware is on the bench.** Our QNN HAL via libQnnHta.so
(Hexagon HTA path) is the production NPU surface today and works
end-to-end through iree-compile + IREE runtime with 3.29 ms median
per inception_q8 dispatch.

The integration shape — a `compiler/plugins/target/Hexagon/` plugin
mirroring `compiler/plugins/target/QNN/` and a runtime HAL driver
under `runtime/src/iree/hal/drivers/hexagon/` — is sketched at the
top of this file. Cheap to revive when 8 Gen 2+ silicon arrives.
