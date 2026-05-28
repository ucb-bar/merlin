"""Qualcomm QNN kernel embedding.

Modules:

- `emit.py`              — MLIR → QNN graph emitter (the production v1 path).
- `ir.py`                — small intermediate representation for QNN graphs.
- `partition.py`         — subgraph partitioner.
- `route.py`             — per-(island, target) routing.
- `gates.py`             — validation gates for the heterogeneous QNN pipeline.
- `build.py`             — toolchain orchestration (qairt-converter → .qnn-ctx).
- `precompile_extras.py` — QNN-specific dispatch invoked from `core.precompile`.
- `recognizers/`         — pattern matchers (NCHW int8 conv, NHWC conv, etc.).
- `headers/`             — C++ headers exposed to runtime code.
- `tests/`               — QNN test suite.

Status: v1 (`emit.py`) is the active path. A v2 attempt was archived to
`tools/archive/qnn_v2/` — see that folder's README for context.
"""
