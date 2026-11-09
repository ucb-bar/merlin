"""Exact compatibility alias for the shared compiler pointwise tiling owner."""

import sys

from merlin.compiler.transforms import pointwise as _implementation

sys.modules[__name__] = _implementation
