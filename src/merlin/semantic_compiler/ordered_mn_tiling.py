"""Exact compatibility alias for shared contraction mn tiling."""

import sys

from merlin.compiler.transforms import contraction_mn as _implementation

sys.modules[__name__] = _implementation
