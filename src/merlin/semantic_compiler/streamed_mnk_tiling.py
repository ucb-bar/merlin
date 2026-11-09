"""Exact compatibility alias for shared panels tiling."""

import sys

from merlin.compiler.transforms import panels as _implementation

sys.modules[__name__] = _implementation
