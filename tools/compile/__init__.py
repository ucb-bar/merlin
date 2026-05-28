"""Implementation package for the `compile` subcommand (`./merlin compile`).

The registered shim is `tools/compile/cli.py`; this package owns the
helper modules.

Extension points (where new behavior plugs in):

- `breakdown_vmfb.py` — one VMFB per `flow.dispatch.region`, with a
  manifest of shapes + SSA dependencies. Extend when adding a new
  dispatch-level analysis or per-dispatch emit-format.
- `chunk_extractor.py` — aggregates dispatches into LAYER / MEGAKERNEL /
  TILE chunks; uses `breakdown_vmfb` parsers. Build on the existing
  `aggregate_<level>` helpers — don't duplicate the SSA walker.
- `chunk_compile.py` — compiles each extracted chunk to its own VMFB.
- `dispatch_matrix.py` — compiles a model for each (dispatch, target)
  cell; emits `matrix.json`. Extend when adding a new target column.
- `benchmark_dispatches.py` — compiles per-dispatch benchmark MLIRs into
  runnable VMFBs.
- `qnn.py` — QNN per-chunk compiler: qairt-converter → model.cpp →
  aarch64 .so → .qnn-ctx. Extend for a new QNN backend or SDK version.

Toolchain paths in this package come from env vars (`QNN_SDK_ROOT`,
`QNN_BOARD_SYSROOT`, `QNN_CROSS_TOOLCHAIN`) — see the no-overfit rule in
`AGENTS.md`. Sibling imports use the package (e.g., `from compile.breakdown_vmfb
import parse_dispatch_creation`), not sys.path hacks.
"""
