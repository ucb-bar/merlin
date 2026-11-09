"""Exact compatibility alias for shared contraction k tiling."""

import sys

from merlin.compiler.transforms import contraction_k as _implementation

sys.modules[__name__] = _implementation
