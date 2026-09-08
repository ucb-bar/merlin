"""An out-of-tree MLIR target backend for the gemmini systolic accelerator, built with xDSL.

Layout
------
* `frontend/`  — the `merlin_iface` input dialect (IRDL) and a structural reader.
* `ir/`        — the `gemmini` target dialect (IRDL ops + verifiers).
* `lowering/`  — semantic normalisation (`plan`) and the tile schedule (`schedule`).
* `codegen/`   — the gemmini-dialect module and the LLVM-dialect target artifact.
* `tables/`    — RTL-derived hardware facts and the RoCC instruction encoding.
* `cmdbuf.py`  — the ABI command-buffer writer.
"""
__version__ = "0.1"
