# QNN ctxbin generation: cross-compile vs on-board, and why

The reasonable question — *can we just cross-compile from this machine?* —
has a backend-specific answer.

## TL;DR

| QNN backend | x86_64 host cross-compile? | Why |
|---|---|---|
| **CPU** (`libQnnCpu.so`) | ✅ Yes | Pure validation, no hardware codegen. |
| **HTP** (`libQnnHtp.so`, v68+) | ✅ Yes | Qualcomm's supported offline-prepare path; pass `dsp_arch` + `soc_id` via backend extensions JSON. |
| **GPU** (`libQnnGpu.so`, Adreno) | ❌ No (segfaults) | Adreno shader codegen is in the on-device driver, not the desktop lib. Even with `TuningMode=true`, libQnnGpu.so on x86_64 segfaults inside graph-compose. |
| **HTA** (`libQnnHta.so`, Hexagon v66) | ❌ No | HTA codegen lives in the on-device driver only; the x86_64 lib is a stub. |

For our QRB5165 — Hexagon 698 = **v66 HTA**, *pre*-HTP — Qualcomm does not
ship a cross-compile path. **We have to go through the board.**

If we ever had access to v68+ silicon (Snapdragon 8 Gen 1, RB6, X Elite, …)
the entire HTP flow would compile on host without any board involvement.

## Concrete repro of the host-side limit

```
$ /scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --model libqnn_add_f32_x86.so \
    --backend /…/lib/x86_64-linux-clang/libQnnGpu.so \
    --binary_file add_f32_host
[ ERROR ] GPU ERROR: GPU_ERROR_INVALID_ARG(10008) - TuningMode must be enabled on x86_64-linux-clang platforms
[ ERROR ] Could not initialize backend
```

With `tuning_mode: true` plus `graph_names` per the
`libQnnGpuNetRunExtensions.so` schema, we get past initialization and
then segfault inside graph compose:

```
Backend (QnnContext_createFromBinary): 9110 us
[…some progress…]
Segmentation fault (core dumped)
```

Same hand-authored kernel, same SDK, same call site, but with
`libQnnHtp.so` + a v68 `dsp_arch` config: **45 KB ctxbin in ~50 µs.**
That's the supported offline-prepare path Qualcomm expects from desktop
hosts.

## What `kernels/qnn_build.py` does today

`build_qnn_kernel_on_board(...)`:
1. scp's the `.qnn.cpp` source + Qualcomm's `QnnModel*.cpp` wrappers + the
   QAIRT include tree to `/tmp/qnn_kernel_build/<sha>/` on `qdev`.
2. Cross-compile? **No** — board has g++ 9.4 native; we just `g++ -shared`
   the pieces *on the board* into `libqnn_<kernel>.so`.
3. Run `qnn-context-binary-generator --backend lib/libQnn{Gpu,Hta}.so`
   *on the board* against that .so. The backend lib's real (Adreno or
   Hexagon HTA) codegen runs at this step, producing a target-native blob.
4. scp the resulting `.qnn-ctx` back into `build/qnn_cache/board_*/...`.

The host (developer machine) supplies the source and aggregates the
output; the board runs the steps that need the on-device driver.

`QNN_USE_BOARD_BUILD=1` toggles whether we use this path — when off, we
fall back to host-side libQnnCpu validation, which produces a CPU-format
ctxbin that's smaller and faster to build but only runs on libQnnCpu.so
(useful as a correctness oracle, not as a deployable artifact).

## Could we avoid the SCP roundtrip in build_tools/?

Two practical wins are possible:

1. **Cache aggressively** — the on-board tarball stage (Qualcomm
   wrappers + QNN headers) is identical across kernel builds. We could
   stage it once per session and let kernel rebuilds only ship the
   `.qnn.cpp` source. `qnn_build.py` already keys the cache on
   `(source_hash, sdk_root, target_backend)`, so repeats short-circuit;
   the only roundtrip is the first-time-this-session cost.

2. **Cross-compile the .so**, run only the ctxbin step on-board.
   `aarch64-linux-gnu-g++` builds the wrapper .so just fine on the host
   (we already do this for the IREE QRB5165 runtime). Running a
   cross-built .so through the on-board `qnn-context-binary-generator`
   would skip the on-board g++ step (~600 ms) but keep the on-board
   serialize step (~few s). The .so is small (~80 KB), so scp dominates
   either way.

Neither saves more than a couple seconds per kernel because the GPU/HTA
backend's compose step is the long pole. For a session with N kernels
and M repeats, the total cost is roughly:

```
   total_seconds ≈ (per-kernel-compose-seconds) × N
                 + ssh-control-master-setup        ; ~250 ms once
                 + scp-source                     ; ~50 ms × N
                 + scp-ctxbin-back                ; ~50 ms × N
```

so the roundtrip overhead is ~10% on top of the irreducible compose time.
Not worth a major refactor.

## What about HTP for future boards?

If a future hardware target uses HTP (Hexagon v68+, e.g., Snapdragon
8 Gen 2/3, X Elite, RB6), the build path can be **fully host-side**:

```python
# Sketch of the refactor when HTP is in scope:
def build_htp_kernel(source, kernel_name, cache_dir, dsp_arch="v73",
                     soc_id=43):
    # All steps below run locally on x86_64 — no board involvement.
    so_path = compile_kernel_so_x86_64(source, kernel_name)
    config_path = write_htp_extensions_json(dsp_arch, soc_id)
    ctx_path = run_qnn_context_binary_generator(
        backend=QAIRT_HOST_LIB / "libQnnHtp.so",
        model=so_path, config_file=config_path)
    return ctx_path  # ready to ship to a v68+ device
```

For QRB5165 specifically the on-board path stays. The build_tools/
toolchains we already pin (`aarch64-linux-gnu-gcc-10` cross-compiler,
`qrb5165_board_sysroot/`) are the right toolchains for cross-compiling
the *runtime* (IREE + QNN HAL driver); they just don't help with
QNN-backend ctxbin generation because that's a closed-source
hardware-driver step.
