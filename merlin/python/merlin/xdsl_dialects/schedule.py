"""Prototype merlin.schedule dialect (xDSL).

xDSL is the default plane for prototyping merlin IR before committing to stable
MLIR/C++ under merlin/compiler/. No real ops/types defined yet.

See docs/dialects.md and docs/xdsl.md.
"""
from __future__ import annotations

DIALECT_NAME = "merlin.schedule"

# TODO: define ops/types using xdsl.irdl once xdsl is installed
# (pip install -e '.[xdsl]'). Keep names aligned with
# merlin/compiler/include/merlin/Dialect/Schedule/ so stable promotion is mechanical.
