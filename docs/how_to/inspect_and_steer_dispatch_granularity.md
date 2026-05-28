# Inspect and Steer Dispatch Granularity

When you compile a model, IREE forms **dispatches** — units of work that the
runtime schedules onto a HAL device. The number, size, and shape of those
dispatches is determined by a chain of fusion and tiling decisions made during
the `dispatch-creation` phase. This guide shows how to see those decisions in
the IR and how to nudge them.

The worked example throughout is `models/dronet/dronet.q.int8.mlir` compiled
for `spacemit_x60` with `--hw RVV`.

## 1. Dump every compilation phase

`./merlin compile` exposes the relevant flags directly — no need to pass
`--mlir-print-ir-after-all` by hand.

```bash
./merlin compile models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized \
  --dump-phases \
  --dump-graph \
  --dump-artifacts \
  --output-dir build/dronet_granularity/default/
```

Outputs of interest in `build/dronet_granularity/default/`:

| Path | What it shows |
|---|---|
| `phases/*.3.preprocessing.mlir` | After Merlin/IREE preprocessing (im2col, conv-to-channels-last, etc.). Still linalg ops. |
| `phases/*.4.global-optimization.mlir` | After global opts (fuse, fold, data-tiling). |
| `phases/*.5.dispatch-creation.mlir` | **Dispatches as `flow.dispatch.region` blocks** — this is the file to read for granularity. |
| `phases/*.6.flow.mlir` | After dispatch outlining: `flow.dispatch @<exec>::@<entry>` call sites. |
| `phases/*.10.executable-targets.mlir` | Final `hal.executable.export` count — the runtime-visible dispatches. |
| `dronet.q.int8_dispatch_graph.dot` | Dataflow DAG; render with `dot -Tpng`. |
| `sources/*.mlir` | One MLIR file per dispatch (after dispatch creation). |
| `binaries/`, `files/`, `configs/` | Per-dispatch executable artifacts. |

## 2. Knobs that change granularity

These are forwarded to `iree-compile` via repeated `--iree-compile-arg=` on the
`./merlin compile` command line.

| Flag | Effect on dispatch graph |
|---|---|
| `--iree-opt-data-tiling=true` (default) | Inserts `pack` / `unpack` ops as **separate dispatches** so matmul inputs land in mmt4d tile layout. Adds 1-2 dispatches per quantized matmul. |
| `--iree-opt-data-tiling=false` | Removes the pack/unpack dispatches; matmuls consume row-major operands directly. |
| `--iree-dispatch-creation-data-tiling=true` | Asks dispatch creation to consume the data-tiling decisions. Used together with the global flag for full effect. |
| `--iree-llvmcpu-enable-ukernels=mmt4d,pack,...` | Routes the matched op classes to IREE's built-in ukernel bitcode rather than codegen. Doesn't change the dispatch *count*, only its body. |
| `--iree-llvmcpu-enable-ukernels=none` | Forces all dispatches through codegen — useful when you want the IR to be self-contained for inspection. |
| `--iree-flow-fuse-multi-use=true` | More aggressive fusion across multi-use values; tends to merge dispatches. |
| `--iree-preprocessing-pass-pipeline=...` | Run preprocessing passes (e.g. `iree-global-opt-convert-conv2d-to-img2col`) before dispatch creation; converts conv into matmul + reshape so it becomes a different (often single) dispatch. |

## 3. Worked comparison on dronet

Three configurations of `dronet.q.int8.mlir` on `spacemit_x60 --hw RVV`,
captured via the commands below:

```bash
# default (data tiling on, mmt4d/pack/query_tile_sizes ukernels off as per yaml)
./merlin compile models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --dump-phases \
  --output-dir build/dronet_granularity/default/

# data tiling off
./merlin compile models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --dump-phases \
  --output-dir build/dronet_granularity/no_data_tiling/ \
  --iree-compile-arg='--iree-opt-data-tiling=false'

# data tiling off + ukernels disabled (matmul codegen self-contained)
./merlin compile models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --dump-phases \
  --output-dir build/dronet_granularity/no_ukernels/ \
  --iree-compile-arg='--iree-opt-data-tiling=false' \
  --iree-compile-arg='--iree-llvmcpu-enable-ukernels=none'
```

| Configuration    | `flow.dispatch.region` (phase 5) | `hal.executable.export` (phase 10) | vmfb bytes |
|---|---:|---:|---:|
| default          | 20 | 22 | 391,542 |
| no_data_tiling   | 20 | 20 | 382,984 |
| no_ukernels      | 20 | 20 | 382,984 |

The +2 dispatches in the default config are the `pack` / `unpack` ops that
data tiling inserts around the matmuls. Disabling ukernels does not change the
dispatch count; it only changes how each dispatch's body is lowered (visible
in `phases/*.10.executable-targets.mlir` — the matmul body is no longer a
`call @iree_uk_mmt4d` import).

## 4. Inspecting a single dispatch

After `--dump-artifacts`, every dispatch lives as a standalone MLIR file under
`sources/`. To recompile any one of them in isolation (useful when isolating a
codegen bug):

```bash
build/host-vanilla-release/tools/iree-compile \
  build/dronet_granularity/default/sources/main_dispatch_5.mlir \
  -o /tmp/dispatch_5.vmfb \
  <same flags the main compile used>
```

For a per-dispatch breakdown with shapes and dependencies (used by the
XPU-RT scheduling tools), see `tools/breakdown_vmfb.py`.

## 5. Visualizing the dispatch graph

```bash
dot -Tpng build/dronet_granularity/default/dronet.q.int8_dispatch_graph.dot \
    -o /tmp/dronet_dispatch_graph.png
```

Each node is a dispatch; edges are tensor dataflow. Comparing the DOT files
between configurations is the fastest way to see what fusion did or didn't
happen.

## 6. Where to look in the source

- `tools/compile/cli.py` — CLI surface (`--dump-phases`, `--dump-graph`,
  `--dump-artifacts`, `--compile-to`, `--iree-compile-arg`).
- `tools/breakdown_vmfb.py` — per-dispatch VMFB extraction + dependency
  manifest.
- `models/spacemit_x60.yaml`, `models/saturn_opu.yaml` — the per-target flag
  bundles (each `--hw` is a knob preset).
