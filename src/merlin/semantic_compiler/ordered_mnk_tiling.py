"""Exact compatibility alias for shared contraction mnk tiling."""

import sys

from merlin.compiler.transforms import contraction_mnk as _implementation

sys.modules[__name__] = _implementation
