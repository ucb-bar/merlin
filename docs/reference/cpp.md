# C/C++ API

Auto-generated from Doxygen comments in `samples/common/` headers.


## Core Utilities

`samples/common/core/` — Generic utilities (no IREE dependency)

- [`cli_utils.h`](generated/cpp/core/cli_utils.md) — C utility functions for CLI flag parsing.
- [`json_parser.h`](generated/cpp/core/json_parser.md) — Minimal recursive-descent JSON parser (header-only, C++).
- [`path_utils.h`](generated/cpp/core/path_utils.md) — Lightweight path manipulation and file-existence utilities (C++).
- [`stats.h`](generated/cpp/core/stats.md) — Log2-bucketed histogram and running statistics for latency tracking.

## Dispatch Scheduling

`samples/common/dispatch/` — Dispatch graph types, parsing, and output

- [`dispatch_graph.h`](generated/cpp/dispatch/dispatch_graph.md) — JSON schedule parsing, predecessor expansion, and topological sort.
- [`dispatch_output.h`](generated/cpp/dispatch/dispatch_output.md) — Trace CSV writer, DOT graph output, and JSON serialization helpers.
- [`dispatch_types.h`](generated/cpp/dispatch/dispatch_types.md) — Core types and scheduling policies for dispatch graph execution.
- [`vmfb_resolve.h`](generated/cpp/dispatch/vmfb_resolve.md) — VMFB path resolution for dispatch schedulers.

## Runtime Utilities

`samples/common/runtime/` — IREE runtime helpers

- [`fatal_state.h`](generated/cpp/runtime/fatal_state.md) — Atomic fatal-error tracking for multi-threaded IREE workloads.
- [`iree_module_utils.h`](generated/cpp/runtime/iree_module_utils.md) — Heuristic entry-function picker for IREE VMFB modules.
- [`module_cache.h`](generated/cpp/runtime/module_cache.md) — VMFB session caching and entry-function invocation helpers.
- [`pinned_device.h`](generated/cpp/runtime/pinned_device.md) — Pinned local-task IREE device creation with per-device core affinity.
