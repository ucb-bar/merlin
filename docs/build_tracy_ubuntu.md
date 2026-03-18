# Tracy Profiling in Merlin

## Overview

[Tracy](https://github.com/wolfpld/tracy) is a hybrid instrumentation and
sampling profiler. IREE's runtime is instrumented with Tracy zones throughout
its task executor, HAL, and VM layers. Merlin samples add additional
application-level zones for dispatch scheduling.

## 1. Build the Tracy Profiler GUI (Host Machine)

```bash
cd third_party/iree_bar/third_party/tracy
sudo apt-get install build-essential cmake libglfw3-dev libdbus-1-dev
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON
cmake --build profiler/build --parallel --config Release
./profiler/build/tracy-profiler
```

## 2. Compile Models with Tracy Debug Info

Use the `--tracy` flag in `tools/compile.py`:

```bash
conda run -n merlin-dev uv run tools/compile.py \
  models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --tracy
```

This adds:

* `--iree-hal-executable-debug-level=3` (source info in VMFBs)
* `--iree-llvmcpu-link-embedded=false` (system linking for symbol resolution)
* `--iree-llvmcpu-debug-symbols=true`

## 3. Build Runtime with Tracy Instrumentation

Use the `--enable-tracy` flag in `tools/build.py`:

```bash
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit --config release --enable-tracy \
  --cmake-target merlin_benchmark_baseline_dual_model_async
```

This sets:

* `IREE_ENABLE_RUNTIME_TRACING=ON`
* `IREE_TRACING_MODE=1` (instrumentation zones + log messages)
* `TRACY_NO_POINTER_COMPRESSION=ON` (required for RISC-V 64-bit)

## 4. Build `iree-tracy-capture` for the Board

```bash
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit --config release --enable-tracy \
  --cmake-arg=-DIREE_BUILD_TRACY=ON \
  --cmake-target iree-tracy-capture
```

The binary is at `build/spacemit-merlin-release/tracy/iree-tracy-capture`.

## 5. Capture a Trace on the Board

### Option A: On-board capture with `iree-tracy-capture`

```bash
# Deploy binary + capture tool to the board
scp build/spacemit-merlin-release/.../benchmark-baseline-dual-model-async-run root@BOARD:/home/baseline/
scp build/spacemit-merlin-release/tracy/iree-tracy-capture root@BOARD:/home/baseline/

# On the board:
TRACY_NO_EXIT=1 ./benchmark-baseline-dual-model-async-run \
  dronet.q.int8_dispatch_graph.json local-task 3 1 1 --parallelism=6 &
sleep 1
./iree-tracy-capture -f -o trace.tracy

# Copy back and open
scp root@BOARD:/home/baseline/trace.tracy .
./tracy-profiler trace.tracy
```

### Option B: Live connection from host

```bash
# Terminal 1: SSH with port forwarding
ssh -L 8086:localhost:8086 root@BOARD

# On the board (through SSH):
TRACY_NO_EXIT=1 ./benchmark-baseline-dual-model-async-run \
  dronet.q.int8_dispatch_graph.json local-task 3 1 1 --parallelism=6

# Terminal 2: Open tracy-profiler, connect to localhost:8086
```

## 6. RISC-V Limitations

| Tracing Mode | What it does | RISC-V status |
|---|---|---|
| `IREE_TRACING_MODE=1` | Instrumentation zones + log messages | Works |
| `IREE_TRACING_MODE=2` | + allocation tracking | Crashes (`PackPointer` assert) |
| `IREE_TRACING_MODE=3` | + allocation callstacks | Crashes (callstack collection) |
| `IREE_TRACING_MODE=4` | + all callstacks | Crashes (callstack collection) |

Root causes:

* **`PackPointer` crash**: Tracy's pointer compression assumes x86-style 48-bit
  virtual addresses. RISC-V 64-bit uses different high-bit patterns. Even
  `TRACY_NO_POINTER_COMPRESSION=ON` only helps the client; the server's
  `PackPointer` in `TracyWorker.cpp` has a hardcoded assert.
* **Callstack crash**: `ProcessZoneBeginCallstack` fails because RISC-V callstack
  unwinding produces addresses that Tracy's server can't process.

## 7. Automation Script

An end-to-end script is available:

```bash
bash samples/SpacemiTX60/b_baseline_dual_model_async/analysis/run_tracy_on_board.sh
```

This compiles models with `--tracy`, builds the runtime with `--enable-tracy`,
deploys to the board, and guides you through the capture process.
