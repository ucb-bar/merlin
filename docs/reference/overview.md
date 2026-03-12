# Reference Overview

Reference pages in this section are generated directly from source code and build metadata:

- Python APIs are extracted from `tools/`, `models/`, `benchmarks/`, and `samples/`.
- CLI docs are emitted from argparse parser definitions in `tools/`.
- MLIR pages are generated from `.td` files via `mlir-tblgen`.
- CMake target inventories are generated from repository `CMakeLists.txt`.
- C/C++ API docs are produced through Doxygen + `mkdoxy`.

Generators run via `docs/hooks.py` before `zensical build` and are validated in CI.
